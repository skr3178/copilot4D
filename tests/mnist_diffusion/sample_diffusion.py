"""Reverse diffusion sampling for Moving MNIST.

Implements Algorithm 2 from the paper (iterative parallel decoding):
1. Start with all tokens masked
2. Iteratively unmask the most confident predictions
3. Generate video from pure noise/mask
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.mnist_diffusion.simple_model import SimpleVideoTransformer
from tests.mnist_diffusion.moving_mnist_precomputed import MovingMNISTPrecomputed


def cosine_schedule(u):
    """Cosine schedule for reverse diffusion."""
    import math
    return torch.cos(u * math.pi / 2)


@torch.no_grad()
def generate_video_iterative(
    model,
    actions,
    num_steps=12,
    device='cuda',
    temperature=1.0,
    cfg_scale=0.0,  # Classifier-free guidance scale
):
    """Generate video using iterative parallel decoding (Algorithm 2).
    
    Args:
        model: Trained transformer model
        actions: (T, action_dim) action sequence
        num_steps: Number of decoding steps
        device: Device to run on
        temperature: Sampling temperature
        cfg_scale: Classifier-free guidance scale (0 = disabled)
        
    Returns:
        tokens: (T, H, W) generated token sequence
    """
    model.eval()
    T = actions.shape[0]
    H, W = model.height, model.width
    N = H * W
    vocab_size = model.vocab_size
    mask_token_id = model.mask_token_id
    
    # Initialize with all tokens masked
    tokens = torch.full((T, H, W), mask_token_id, dtype=torch.long, device=device)
    
    # Expand actions for batch
    actions = actions.unsqueeze(0).to(device)  # (1, T, action_dim)
    
    # Iterative decoding
    for step in range(num_steps):
        # Current ratio of decoded tokens
        ratio = (step + 1) / num_steps
        
        # Number of tokens to unmask at this step
        # Use cosine schedule for unmasking
        u = torch.tensor(ratio)
        mask_ratio = cosine_schedule(u).item()
        
        # Count currently masked tokens
        masked_positions = (tokens == mask_token_id)
        num_masked = masked_positions.sum().item()
        
        # Number to unmask
        num_to_unmask = num_masked - int(mask_ratio * T * N)
        if num_to_unmask <= 0:
            continue
        
        # Forward pass to get predictions
        tokens_batch = tokens.unsqueeze(0)  # (1, T, H, W)
        
        # Create temporal mask (causal for generation)
        causal_mask = torch.full((T, T), float('-inf'), device=device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        
        logits = model(tokens_batch, actions, causal_mask)  # (1, T, N, vocab_size+1)
        logits = logits[0]  # (T, N, vocab_size+1)
        
        # Get predictions for currently masked positions only
        logits_masked = logits[masked_positions.reshape(T, N)]  # (num_masked, vocab_size+1)
        
        # Sample from the distribution
        probs = F.softmax(logits_masked / temperature, dim=-1)
        
        # Don't sample mask token (last index)
        probs_no_mask = probs[:, :vocab_size]
        probs_no_mask = probs_no_mask / (probs_no_mask.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Sample tokens
        sampled_tokens = torch.multinomial(probs_no_mask, num_samples=1).squeeze(-1)
        
        # Get confidence scores (probability of sampled token)
        confidence = probs_no_mask[torch.arange(len(sampled_tokens)), sampled_tokens]
        
        # Select top-k most confident to unmask
        k = min(num_to_unmask, len(sampled_tokens))
        top_k_indices = torch.topk(confidence, k).indices
        
        # Create flat token view
        tokens_flat = tokens.reshape(-1).clone()
        masked_flat = masked_positions.reshape(-1)
        masked_indices = torch.where(masked_flat)[0]
        
        # Unmask the selected tokens
        unmask_positions = masked_indices[top_k_indices]
        tokens_flat[unmask_positions] = sampled_tokens[top_k_indices]
        tokens = tokens_flat.reshape(T, H, W)
    
    return tokens.cpu()


@torch.no_grad()
def generate_video_future_prediction(
    model,
    context_frames,
    future_actions,
    device='cuda',
    temperature=1.0,
):
    """Generate future frames given context (future prediction mode).
    
    Args:
        model: Trained transformer model
        context_frames: (T_context, H, W) context token sequence
        future_actions: (T_future, action_dim) future actions
        device: Device to run on
        temperature: Sampling temperature
        
    Returns:
        tokens: (T_context + T_future, H, W) complete sequence
    """
    model.eval()
    T_context = context_frames.shape[0]
    T_future = future_actions.shape[0]
    T_total = T_context + T_future
    H, W = model.height, model.width
    N = H * W
    mask_token_id = model.mask_token_id
    
    # Initialize with context + masked future
    tokens = torch.full((T_total, H, W), mask_token_id, dtype=torch.long, device=device)
    tokens[:T_context] = context_frames.to(device)
    
    # Expand actions for batch
    # For context frames, use zero actions
    context_actions = torch.zeros(T_context, future_actions.shape[-1], device=device)
    all_actions = torch.cat([context_actions, future_actions.to(device)], dim=0)
    all_actions = all_actions.unsqueeze(0)  # (1, T_total, action_dim)
    
    # Iterative decoding (fewer steps since we have context)
    num_steps = 8
    
    for step in range(num_steps):
        ratio = (step + 1) / num_steps
        u = torch.tensor(ratio)
        mask_ratio = cosine_schedule(u).item()
        
        # Only consider future positions for unmasking
        future_mask = torch.zeros(T_total, H, W, dtype=torch.bool, device=device)
        future_mask[T_context:] = True
        
        masked_positions = (tokens == mask_token_id) & future_mask
        num_masked = masked_positions.sum().item()
        
        if num_masked == 0:
            break
        
        num_to_unmask = num_masked - int(mask_ratio * T_future * N)
        if num_to_unmask <= 0:
            continue
        
        tokens_batch = tokens.unsqueeze(0)
        
        # Causal mask
        causal_mask = torch.full((T_total, T_total), float('-inf'), device=device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        
        logits = model(tokens_batch, all_actions, causal_mask)[0]
        
        # Only consider masked future positions
        logits_masked = logits[masked_positions.reshape(T_total, N)]
        
        probs = F.softmax(logits_masked / temperature, dim=-1)
        probs_no_mask = probs[:, :model.vocab_size]
        probs_no_mask = probs_no_mask / (probs_no_mask.sum(dim=-1, keepdim=True) + 1e-8)
        
        sampled_tokens = torch.multinomial(probs_no_mask, num_samples=1).squeeze(-1)
        confidence = probs_no_mask[torch.arange(len(sampled_tokens)), sampled_tokens]
        
        k = min(num_to_unmask, len(sampled_tokens))
        top_k_indices = torch.topk(confidence, k).indices
        
        tokens_flat = tokens.reshape(-1).clone()
        masked_flat = masked_positions.reshape(-1)
        masked_indices = torch.where(masked_flat)[0]
        
        unmask_positions = masked_indices[top_k_indices]
        tokens_flat[unmask_positions] = sampled_tokens[top_k_indices]
        tokens = tokens_flat.reshape(T_total, H, W)
    
    return tokens.cpu()


def tokens_to_video(tokens, num_levels=16):
    """Convert discrete tokens back to video frames.
    
    Args:
        tokens: (T, H, W) token indices
        num_levels: Number of quantization levels
        
    Returns:
        frames: (T, H, W) normalized frames [0, 1]
    """
    return tokens.float() / (num_levels - 1)


def save_video_comparison(real_frames, generated_frames, save_path, num_show=5):
    """Save a comparison of real vs generated videos.
    
    Args:
        real_frames: (T, H, W) real frames
        generated_frames: (T, H, W) generated frames
        save_path: Path to save visualization
        num_show: Number of timesteps to show
    """
    import matplotlib.pyplot as plt
    
    T = min(len(real_frames), len(generated_frames), num_show)
    
    fig, axes = plt.subplots(2, T, figsize=(T * 2, 4))
    
    for t in range(T):
        # Real
        axes[0, t].imshow(real_frames[t], cmap='gray', vmin=0, vmax=1)
        axes[0, t].set_title(f'Real t={t}')
        axes[0, t].axis('off')
        
        # Generated
        axes[1, t].imshow(generated_frames[t], cmap='gray', vmin=0, vmax=1)
        axes[1, t].set_title(f'Gen t={t}')
        axes[1, t].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Sample from trained discrete diffusion model")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, default="mnist_test_seq.1.npy")
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--frame_size", type=int, default=32)
    parser.add_argument("--num_token_levels", type=int, default=16)
    
    # Model config (should match training)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    
    # Sampling config
    parser.add_argument("--num_samples", type=int, default=5, help="Number of videos to generate")
    parser.add_argument("--num_steps", type=int, default=12, help="Number of denoising steps")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="outputs/mnist_samples")
    parser.add_argument("--mode", type=str, default="generation", 
                       choices=["generation", "future_prediction"],
                       help="Generation mode: pure generation or future prediction")
    parser.add_argument("--num_context", type=int, default=10, 
                       help="Number of context frames for future prediction")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = SimpleVideoTransformer(
        vocab_size=args.num_token_levels,
        mask_token_id=args.num_token_levels,
        num_frames=args.seq_len,
        height=args.frame_size,
        width=args.frame_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Load dataset for reference
    dataset = MovingMNISTPrecomputed(
        data_path=args.data_path,
        seq_len=args.seq_len,
        num_sequences=args.num_samples,
        start_idx=0,
        num_token_levels=args.num_token_levels,
        frame_size=args.frame_size,
    )
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nGenerating {args.num_samples} videos using {args.mode} mode...")
    
    for i in tqdm(range(args.num_samples)):
        # Get reference data
        sample = dataset[i]
        real_tokens = sample["tokens"]  # (T, H, W)
        actions = sample["actions"]  # (T, action_dim)
        
        if args.mode == "generation":
            # Pure generation from noise
            generated_tokens = generate_video_iterative(
                model, actions,
                num_steps=args.num_steps,
                device=device,
                temperature=args.temperature,
            )
        else:
            # Future prediction
            context = real_tokens[:args.num_context]
            future_actions = actions[args.num_context:]
            generated_tokens = generate_video_future_prediction(
                model, context, future_actions,
                device=device,
                temperature=args.temperature,
            )
        
        # Convert to video frames
        real_video = tokens_to_video(real_tokens, args.num_token_levels).numpy()
        gen_video = tokens_to_video(generated_tokens, args.num_token_levels).numpy()
        
        # Save comparison
        save_path = Path(args.output_dir) / f"sample_{i:03d}.png"
        save_video_comparison(real_video, gen_video, save_path)
        
        # Save individual videos as numpy arrays for further analysis
        np.save(Path(args.output_dir) / f"sample_{i:03d}_real.npy", real_video)
        np.save(Path(args.output_dir) / f"sample_{i:03d}_gen.npy", gen_video)
    
    print(f"\nSaved {args.num_samples} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
