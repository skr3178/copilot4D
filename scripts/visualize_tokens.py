#!/usr/bin/env python3
"""
Visualize token sequences as heatmaps for world model evaluation.

This creates a visualization comparing:
- Past context tokens
- Generated future tokens
- Ground truth future tokens
- Difference/error maps

Usage:
    python scripts/visualize_tokens.py \
        --checkpoint outputs/world_model_12gb/checkpoint_0050000.pt \
        --sequence 00 \
        --start_frame 500 \
        --num_past_frames 2 \
        --num_future_frames 3 \
        --output figures/token_comparison.png
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent.parent))

from copilot4d.world_model.world_model import CoPilot4DWorldModel, WorldModelConfig
from copilot4d.world_model.masking import cosine_mask_schedule


def load_world_model(checkpoint_path: str, device: str = "cuda"):
    """Load trained world model from checkpoint."""
    print(f"Loading world model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    cfg = checkpoint.get("config")
    if cfg is None:
        raise ValueError("Checkpoint missing config. Cannot load model.")
    
    cfg.num_sampling_steps = getattr(cfg, 'num_sampling_steps', 16)
    cfg.choice_temperature = getattr(cfg, 'choice_temperature', 4.5)
    
    model = CoPilot4DWorldModel(cfg).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Loaded model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    return model, cfg


def load_tokens(token_dir: str, sequence: str, start_frame: int, num_frames: int):
    """Load tokenized frames."""
    import pickle
    seq_path = Path(token_dir) / sequence
    
    tokens_list = []
    for i in range(start_frame, start_frame + num_frames):
        token_file = seq_path / f"{i:06d}.pt"
        if not token_file.exists():
            raise FileNotFoundError(f"Token file not found: {token_file}")
        tokens = torch.load(token_file, map_location="cpu")
        tokens_list.append(tokens)
    
    tokens = torch.stack(tokens_list)  # (T, H, W)
    
    # Load poses for actions
    poses_file = seq_path / "poses.pkl"
    if poses_file.exists():
        with open(poses_file, 'rb') as f:
            all_poses = pickle.load(f)
        poses = [all_poses[i] for i in range(start_frame, start_frame + num_frames)]
    else:
        poses = [np.eye(4) for _ in range(num_frames)]
    
    # Compute relative actions
    actions = []
    for i in range(1, len(poses)):
        T_prev = poses[i-1]
        T_curr = poses[i]
        T_rel = np.linalg.inv(T_prev) @ T_curr
        actions.append(T_rel.flatten())
    actions.append(np.eye(4).flatten())
    actions = np.stack(actions)
    
    return tokens, torch.from_numpy(actions).float()


@torch.no_grad()
def sample_future_tokens(model, past_tokens, past_actions, num_future, cfg, device, greedy=True):
    """Generate future tokens using Algorithm 2 (with greedy option)."""
    T_past, H, W = past_tokens.shape
    N = H * W
    mask_id = cfg.codebook_size
    
    # Initialize future tokens as all masked
    future_tokens = torch.full((num_future, H, W), mask_id, dtype=torch.long, device=device)
    
    # Use last action for future prediction
    last_action = past_actions[-1:] if len(past_actions) > 0 else torch.zeros(1, 16)
    future_actions = last_action.repeat(num_future, 1).to(device)
    
    # Combine past and future
    full_tokens = torch.cat([past_tokens.to(device), future_tokens], dim=0)
    full_actions = torch.cat([past_actions.to(device), future_actions], dim=0)
    
    # Ensure we fit within model capacity
    if full_tokens.shape[0] > cfg.num_frames:
        keep = cfg.num_frames - num_future
        full_tokens = torch.cat([
            full_tokens[-cfg.num_frames:-num_future], 
            full_tokens[-num_future:]
        ], dim=0)
        full_actions = torch.cat([
            full_actions[-cfg.num_frames:-num_future], 
            full_actions[-num_future:]
        ], dim=0)
        T_past = keep
    
    T_total = full_tokens.shape[0]
    
    # Create causal temporal mask
    temporal_mask = torch.triu(torch.ones(T_total, T_total) * float('-inf'), diagonal=1).to(device)
    
    # Iterative decoding
    num_steps = cfg.num_sampling_steps
    
    for step in range(num_steps):
        # Forward pass
        logits = model(full_tokens.unsqueeze(0), full_actions.unsqueeze(0), temporal_mask)
        logits_future = logits[0, T_past:]  # (T_future, N, V)
        
        future_flat = future_tokens.reshape(num_future, N)
        
        if greedy:
            # Greedy: take argmax
            predictions = logits_future.argmax(dim=-1)  # (T_future, N)
            # Only update masked positions
            mask = (future_flat == mask_id)
            future_flat = torch.where(mask, predictions, future_flat)
        else:
            # Sampling with temperature
            probs = torch.softmax(logits_future / cfg.choice_temperature, dim=-1)
            # Only sample at masked positions
            for f in range(num_future):
                for n in range(N):
                    if future_flat[f, n] == mask_id:
                        sample = torch.multinomial(probs[f, n], 1).item()
                        future_flat[f, n] = sample
        
        future_tokens = future_flat.reshape(num_future, H, W)
        full_tokens = torch.cat([full_tokens[:T_past], future_tokens], dim=0)
        
        # Remask for next step
        if step < num_steps - 1:
            t = (step + 1) / num_steps
            mask_ratio = cosine_mask_schedule(torch.tensor(t)).item()
            num_to_mask = int(mask_ratio * N * num_future)
            
            if num_to_mask > 0:
                if greedy:
                    # For greedy, use logit confidence
                    confidences = logits_future.max(dim=-1).values  # (T_future, N)
                else:
                    confidences = probs.max(dim=-1).values
                
                flat_conf = confidences.reshape(-1)
                _, indices = torch.topk(flat_conf, k=num_to_mask, largest=False)
                future_flat = future_tokens.reshape(-1)
                future_flat[indices] = mask_id
                future_tokens = future_flat.reshape(num_future, H, W)
                full_tokens = torch.cat([full_tokens[:T_past], future_tokens], dim=0)
    
    return future_tokens.cpu()


def create_token_colormap():
    """Create a colormap for token visualization."""
    # Use tab20 for discrete tokens, with black for masked (if present)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    # Add more colors for up to 1024 tokens
    extended_colors = np.vstack([
        plt.cm.tab20(np.linspace(0, 1, 20)),
        plt.cm.tab20b(np.linspace(0, 1, 20)),
        plt.cm.tab20c(np.linspace(0, 1, 20)),
        plt.cm.Set1(np.linspace(0, 1, 9)),
        plt.cm.Set2(np.linspace(0, 1, 8)),
        plt.cm.Set3(np.linspace(0, 1, 12)),
    ])
    # Pad to 1024 with random colors
    while len(extended_colors) < 1024:
        extended_colors = np.vstack([
            extended_colors,
            np.random.rand(128, 4)
        ])
    return extended_colors[:1024]


def visualize_tokens(past_tokens, gen_tokens, gt_tokens, output_path):
    """Create comprehensive token visualization."""
    T_past = past_tokens.shape[0]
    T_future = gen_tokens.shape[0]
    
    # Create figure with grid
    fig = plt.figure(figsize=(16, 4 * (T_past + T_future)))
    gs = gridspec.GridSpec(T_past + T_future, 4, figure=fig, 
                           width_ratios=[1, 1, 1, 0.3],
                           wspace=0.3, hspace=0.4)
    
    # Colormap
    token_cmap = create_token_colormap()
    
    def plot_tokens(ax, tokens, title, cmap=None):
        """Plot token heatmap."""
        tokens_np = tokens.numpy()
        im = ax.imshow(tokens_np, cmap='tab20' if cmap is None else cmap, 
                      vmin=0, vmax=1023, interpolation='nearest')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        return im
    
    def plot_diff(ax, tokens1, tokens2, title):
        """Plot difference between two token sets."""
        diff = (tokens1 != tokens2).float().numpy()
        im = ax.imshow(diff, cmap='RdYlGn_r', vmin=0, vmax=1, interpolation='nearest')
        accuracy = (1 - diff.mean()) * 100
        ax.set_title(f"{title}\nAcc: {accuracy:.1f}%", fontsize=10)
        ax.axis('off')
        return im, accuracy
    
    accuracies = []
    
    # Plot past frames
    for i in range(T_past):
        # Past tokens
        ax = fig.add_subplot(gs[i, 0])
        plot_tokens(ax, past_tokens[i], f"Past Frame {i}")
        
        # Leave other columns empty for past
        ax = fig.add_subplot(gs[i, 1])
        ax.axis('off')
        ax = fig.add_subplot(gs[i, 2])
        ax.axis('off')
        ax = fig.add_subplot(gs[i, 3])
        ax.axis('off')
    
    # Plot future frames
    for i in range(T_future):
        row = T_past + i
        
        # GT tokens
        ax = fig.add_subplot(gs[row, 0])
        plot_tokens(ax, gt_tokens[i], f"GT Future {i}")
        
        # Generated tokens
        ax = fig.add_subplot(gs[row, 1])
        plot_tokens(ax, gen_tokens[i], f"Generated {i}")
        
        # Difference
        ax = fig.add_subplot(gs[row, 2])
        _, acc = plot_diff(ax, gen_tokens[i], gt_tokens[i], f"Difference {i}")
        accuracies.append(acc)
        
        # Legend/colorbar
        ax = fig.add_subplot(gs[row, 3])
        ax.axis('off')
    
    # Add overall title
    fig.suptitle(f"Token Generation Comparison\n"
                f"Mean Accuracy: {np.mean(accuracies):.1f}% "
                f"(Min: {np.min(accuracies):.1f}%, Max: {np.max(accuracies):.1f}%)",
                fontsize=14, fontweight='bold')
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()
    
    return accuracies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--token_dir", default="data/kitti/tokens")
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--start_frame", type=int, default=500)
    parser.add_argument("--num_past_frames", type=int, default=2)
    parser.add_argument("--num_future_frames", type=int, default=3)
    parser.add_argument("--output", default="figures/token_comparison.png")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load model
    model, cfg = load_world_model(args.checkpoint, device)
    
    # Load data
    print(f"\nLoading sequence {args.sequence} frames {args.start_frame}-{args.start_frame + args.num_past_frames + args.num_future_frames}")
    total_frames = args.num_past_frames + args.num_future_frames
    tokens, actions = load_tokens(args.token_dir, args.sequence, args.start_frame, total_frames)
    
    past_tokens = tokens[:args.num_past_frames]
    past_actions = actions[:args.num_past_frames]
    gt_future = tokens[args.num_past_frames:]
    
    print(f"Past context: {past_tokens.shape}")
    print(f"Ground truth future: {gt_future.shape}")
    
    # Generate with greedy decoding
    print(f"\nGenerating {args.num_future_frames} future frames (greedy decoding)...")
    gen_future = sample_future_tokens(model, past_tokens, past_actions, 
                                      args.num_future_frames, cfg, device, greedy=True)
    
    print(f"Generated: {gen_future.shape}")
    
    # Compute accuracy directly
    direct_acc = (gen_future == gt_future).float().mean() * 100
    print(f"Direct token accuracy: {direct_acc:.2f}%")
    
    # Visualize
    print("\nCreating visualization...")
    accuracies = visualize_tokens(past_tokens, gen_future, gt_future, args.output)
    
    print(f"\n✅ Done!")
    print(f"   Frame accuracies: {[f'{a:.1f}%' for a in accuracies]}")
    print(f"   Mean accuracy: {np.mean(accuracies):.1f}%")


if __name__ == "__main__":
    main()
