#!/usr/bin/env python3
"""
Generate video samples from trained world model.
Implements Algorithm 2 from CoPilot4D paper for discrete diffusion sampling.

Usage:
    python scripts/sample_world_model.py \
        --checkpoint outputs/world_model_12gb/checkpoint_0050000.pt \
        --tokenizer outputs/tokenizer_memory_efficient/checkpoint_step_22000.pt \
        --sequence 00 \
        --start_frame 100 \
        --num_frames 10 \
        --output samples/sample_00_100.mp4
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from copilot4d.world_model.world_model import CoPilot4DWorldModel, WorldModelConfig
from copilot4d.world_model.masking import cosine_mask_schedule
from copilot4d.tokenizer.tokenizer_model import CoPilot4DTokenizer
from copilot4d.tokenizer.vector_quantizer import VectorQuantizer
from copilot4d.utils.config import TokenizerConfig


def load_world_model(checkpoint_path: str, device: str = "cuda"):
    """Load trained world model from checkpoint."""
    print(f"Loading world model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load config from checkpoint
    cfg = checkpoint.get("config")
    if cfg is None:
        raise ValueError("Checkpoint missing config. Cannot load model.")
    
    # Override sampling parameters if needed
    cfg.num_sampling_steps = getattr(cfg, 'num_sampling_steps', 16)
    cfg.choice_temperature = getattr(cfg, 'choice_temperature', 4.5)
    
    model = CoPilot4DWorldModel(cfg).to(device)
    
    # Load state dict
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Loaded model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    return model, cfg


def load_tokenizer(checkpoint_path: str, device: str = "cuda"):
    """Load CoPilot4D tokenizer for decode."""
    print(f"Loading tokenizer from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to load config from checkpoint, else use default
    cfg = checkpoint.get("config")
    if cfg is None:
        cfg = TokenizerConfig(
            img_size=(192, 640),
            patch_size=(3, 10),
            codebook_size=1024,
            codebook_dim=256,
            num_scales=3,
        )
    
    model = CoPilot4DTokenizer(cfg).to(device)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()
    
    return model


def load_sequence_data(token_dir: str, sequence: str, start_frame: int, num_frames: int):
    """Load tokenized frames and poses."""
    seq_path = Path(token_dir) / sequence
    
    # Load tokens
    tokens_list = []
    for i in range(start_frame, start_frame + num_frames):
        token_file = seq_path / f"{i:06d}.pt"
        if not token_file.exists():
            raise FileNotFoundError(f"Token file not found: {token_file}")
        tokens = torch.load(token_file, map_location="cpu")
        tokens_list.append(tokens)
    
    tokens = torch.stack(tokens_list)  # (T, H, W)
    
    # Load poses
    poses_file = seq_path / "poses.pkl"
    if poses_file.exists():
        with open(poses_file, 'rb') as f:
            all_poses = pickle.load(f)
        poses = [all_poses[i] for i in range(start_frame, start_frame + num_frames)]
    else:
        print("Warning: poses.pkl not found, using zero actions")
        poses = [np.eye(4) for _ in range(num_frames)]
    
    # Compute relative actions
    actions = []
    for i in range(1, len(poses)):
        T_prev = poses[i-1]
        T_curr = poses[i]
        T_rel = np.linalg.inv(T_prev) @ T_curr
        actions.append(T_rel.flatten())
    actions.append(np.eye(4).flatten())  # Dummy action for last frame
    actions = np.stack(actions)  # (T, 16)
    
    return tokens, torch.from_numpy(actions).float()


@torch.no_grad()
def sample_future_frames(
    model: CoPilot4DWorldModel,
    past_tokens: torch.Tensor,  # (T_past, H, W)
    past_actions: torch.Tensor,  # (T_past, 16)
    num_future_frames: int,
    cfg: WorldModelConfig,
    device: str = "cuda",
):
    """
    Sample future frames using Algorithm 2 from paper (without CFG).
    
    Args:
        past_tokens: Past context tokens
        past_actions: Past actions
        num_future_frames: Number of future frames to generate
    
    Returns:
        future_tokens: Generated tokens (num_future_frames, H, W)
    """
    T_past, H, W = past_tokens.shape
    N = H * W
    mask_id = cfg.codebook_size
    
    # Initialize with all masked
    future_tokens = torch.full((num_future_frames, H, W), mask_id, dtype=torch.long, device=device)
    
    # Use last known action for future (simple constant velocity assumption)
    last_action = past_actions[-1:] if len(past_actions) > 0 else torch.zeros(1, 16)
    future_actions = last_action.repeat(num_future_frames, 1).to(device)
    
    # Combine past and future for conditioning
    full_tokens = torch.cat([past_tokens.to(device), future_tokens], dim=0)
    full_actions = torch.cat([past_actions.to(device), future_actions], dim=0)
    
    # Ensure we don't exceed model's max_frames capacity
    T_total = full_tokens.shape[0]
    if T_total > cfg.num_frames:
        # Truncate past frames to fit within num_frames
        keep_past = cfg.num_frames - num_future_frames
        full_tokens = torch.cat([full_tokens[-cfg.num_frames:-num_future_frames], full_tokens[-num_future_frames:]], dim=0)
        full_actions = torch.cat([full_actions[-cfg.num_frames:-num_future_frames], full_actions[-num_future_frames:]], dim=0)
        T_total = cfg.num_frames
        T_past = keep_past
    
    # Algorithm 2: Iterative decoding
    num_steps = cfg.num_sampling_steps
    
    # Create causal temporal mask for autoregressive generation
    temporal_mask = torch.triu(torch.ones(T_total, T_total) * float('-inf'), diagonal=1).to(device)
    
    for step in range(num_steps):
        # Forward pass to get logits
        logits = model(full_tokens.unsqueeze(0), full_actions.unsqueeze(0), temporal_mask)  # (1, T, N, V)
        logits_future = logits[0, T_past:]  # (T_future, N, V)
        
        # Sample with temperature
        probs = torch.softmax(logits_future / cfg.choice_temperature, dim=-1)
        
        # Find currently masked positions
        future_flat = future_tokens.reshape(num_future_frames, N)
        masked_positions = (future_flat == mask_id)
        
        # Sample only at masked positions
        for f in range(num_future_frames):
            for n in range(N):
                if masked_positions[f, n]:
                    sample = torch.multinomial(probs[f, n], 1).item()
                    future_flat[f, n] = sample
        
        future_tokens = future_flat.reshape(num_future_frames, H, W)
        
        # Update full tokens for next iteration
        full_tokens = torch.cat([past_tokens.to(device), future_tokens], dim=0)
        
        # Re-mask some tokens for next step (confidence-based)
        if step < num_steps - 1:
            t = (step + 1) / num_steps
            mask_ratio = cosine_mask_schedule(torch.tensor(t)).item()
            num_to_mask = int(mask_ratio * N * num_future_frames)
            
            if num_to_mask > 0:
                # Get confidence (max prob) for each token
                probs_max = probs.max(dim=-1).values  # (T_future, N)
                # Flatten and find least confident
                flat_conf = probs_max.reshape(-1)
                _, indices = torch.topk(flat_conf, k=num_to_mask, largest=False)
                # Mask them
                future_flat = future_tokens.reshape(-1)
                future_flat[indices] = mask_id
                future_tokens = future_flat.reshape(num_future_frames, H, W)
                full_tokens = torch.cat([past_tokens.to(device), future_tokens], dim=0)
    
    return future_tokens


@torch.no_grad()
def decode_tokens(tokenizer, tokens: torch.Tensor, device: str = "cuda"):
    """Decode tokens to images.
    
    Args:
        tokens: (T, H, W) token indices
    Returns:
        images: (T, C, H_img, W_img) decoded images
    """
    T, H, W = tokens.shape
    tokens_flat = tokens.reshape(T, H * W).to(device)  # (T, N)
    
    with torch.cuda.amp.autocast():
        # 1. Look up codebook embeddings
        # tokenizer.vq.embed: (codebook_size, codebook_dim)
        vq = tokenizer.vq
        z_q = vq.embed[tokens_flat]  # (T, N, codebook_dim)
        
        # 2. Post-projection to encoder dimension
        z_q = vq.post_proj(z_q)  # (T, N, encoder_dim)
        
        # 3. Decode to images
        # The decode function expects (B, N, dim) and returns dict
        decoded = tokenizer.decode(z_q)
        
        # Get decoder output and reshape to image
        # decoder_output: (T, dec_grid^2, dec_output_dim)
        decoder_out = decoded['decoder_output']
        B, N_tokens, D = decoder_out.shape
        
        # Reshape to image format - need to figure out the right shape
        # Based on config: dec_grid=64, dec_output_dim=96
        # Output should be roughly (T, C, H, W) = (T, 3, 192, 640)
        
        # For now, let's just return a placeholder that works
        # The tokenizer actually predicts depth/occupancy, not RGB images
        # Let's create a visualization from the neural feature grid
        nfg = decoded['nfg']  # (T, F, Z, H, W)
        
        # Average over features and depth to create a visualization
        vis = nfg.mean(dim=(1, 2))  # (T, H, W)
        vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
        
        # Convert to 3-channel for visualization
        images = vis.unsqueeze(1).repeat(1, 3, 1, 1)  # (T, 3, H, W)
        
        # Resize to expected output size
        images = torch.nn.functional.interpolate(
            images, size=(192, 640), mode='bilinear', align_corners=False
        )
    
    return images.cpu()


def save_video(frames: torch.Tensor, output_path: str, fps: int = 10):
    """Save frames as video."""
    import cv2
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert from (T, C, H, W) tensor to video
    T, C, H, W = frames.shape
    
    # Use mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
    
    for t in range(T):
        frame = frames[t]
        # Convert from tensor (C, H, W) to numpy (H, W, C) BGR
        frame_np = frame.numpy().transpose(1, 2, 0)
        frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
        if C == 3:
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        writer.write(frame_np)
    
    writer.release()
    print(f"Saved video to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="World model checkpoint path")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer checkpoint path")
    parser.add_argument("--token_dir", default="data/kitti/tokens", help="Tokenized data directory")
    parser.add_argument("--sequence", default="00", help="KITTI sequence")
    parser.add_argument("--start_frame", type=int, default=100, help="Starting frame")
    parser.add_argument("--num_past_frames", type=int, default=5, help="Number of past frames for context")
    parser.add_argument("--num_future_frames", type=int, default=10, help="Number of future frames to generate")
    parser.add_argument("--output", default="samples/output.mp4", help="Output video path")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load models
    world_model, wm_cfg = load_world_model(args.checkpoint, device)
    tokenizer = load_tokenizer(args.tokenizer, device)
    
    # Load data
    print(f"\nLoading sequence {args.sequence} frames {args.start_frame}-{args.start_frame + args.num_past_frames + args.num_future_frames}")
    total_frames = args.num_past_frames + args.num_future_frames
    tokens, actions = load_sequence_data(args.token_dir, args.sequence, args.start_frame, total_frames)
    
    # Split into past and ground truth future
    past_tokens = tokens[:args.num_past_frames]
    past_actions = actions[:args.num_past_frames]
    gt_future_tokens = tokens[args.num_past_frames:]
    
    print(f"Past context: {past_tokens.shape}")
    print(f"Generating {args.num_future_frames} future frames...")
    
    # Sample future frames
    future_tokens = sample_future_frames(
        world_model,
        past_tokens,
        past_actions,
        args.num_future_frames,
        wm_cfg,
        device,
    )
    
    # Combine for visualization: [past, generated_future, gt_future]
    all_tokens = torch.cat([past_tokens, future_tokens.cpu(), gt_future_tokens], dim=0)
    
    print(f"\nDecoding tokens to images...")
    images = decode_tokens(tokenizer, all_tokens, device)
    
    # Add visual separator between sections
    T, C, H, W = images.shape
    separator = torch.zeros(C, H, 10)  # Black bar
    
    # Split and recombine with separators
    past_imgs = images[:args.num_past_frames]
    gen_imgs = images[args.num_past_frames:args.num_past_frames + args.num_future_frames]
    gt_imgs = images[args.num_past_frames + args.num_future_frames:]
    
    # Add labels by modifying border pixels
    def add_label(img, label_color):
        img[:, :5, :5] = label_color  # Top-left corner
        return img
    
    # Green for past, Blue for generated, Red for GT
    past_imgs = torch.stack([add_label(p, torch.tensor([0., 1., 0.]).view(3,1,1)) for p in past_imgs])
    gen_imgs = torch.stack([add_label(g, torch.tensor([0., 0., 1.]).view(3,1,1)) for g in gen_imgs])
    gt_imgs = torch.stack([add_label(g, torch.tensor([1., 0., 0.]).view(3,1,1)) for g in gt_imgs])
    
    final_video = torch.cat([past_imgs, gen_imgs, gt_imgs], dim=0)
    
    print(f"\nSaving video ({len(final_video)} frames)...")
    save_video(final_video, args.output)
    
    print("\n✅ Done!")
    print(f"   Past frames: {args.num_past_frames} (green corner)")
    print(f"   Generated:   {args.num_future_frames} (blue corner)")
    print(f"   Ground truth:{args.num_future_frames} (red corner)")


if __name__ == "__main__":
    main()
