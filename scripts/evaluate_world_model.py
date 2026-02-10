#!/usr/bin/env python3
"""
Evaluate world model by comparing predicted future tokens with ground truth.
Creates visualizations similar to tokenizer evaluation.

Usage:
    python scripts/evaluate_world_model.py \
        --checkpoint outputs/world_model_12gb/checkpoint_0050000.pt \
        --sequence 00 \
        --start_frame 500 \
        --output_dir outputs/world_model_eval
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

from copilot4d.world_model.world_model import CoPilot4DWorldModel, WorldModelConfig
from copilot4d.world_model.masking import cosine_mask_schedule


def load_world_model(checkpoint_path: str, device: str = "cuda"):
    """Load trained world model."""
    print(f"Loading world model: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    cfg = checkpoint.get("config")
    if cfg is None:
        raise ValueError("Checkpoint missing config")
    
    cfg.num_sampling_steps = getattr(cfg, 'num_sampling_steps', 16)
    cfg.choice_temperature = getattr(cfg, 'choice_temperature', 4.5)
    
    model = CoPilot4DWorldModel(cfg).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    return model, cfg


def load_sequence_data(token_dir: str, sequence: str, start_frame: int, num_frames: int):
    """Load tokens and poses."""
    seq_path = Path(token_dir) / sequence
    
    tokens_list = []
    for i in range(start_frame, start_frame + num_frames):
        token_file = seq_path / f"{i:06d}.pt"
        tokens = torch.load(token_file, map_location="cpu")
        tokens_list.append(tokens)
    
    tokens = torch.stack(tokens_list)
    
    # Load poses
    poses_file = seq_path / "poses.pkl"
    if poses_file.exists():
        with open(poses_file, 'rb') as f:
            all_poses = pickle.load(f)
        poses = [all_poses[i] for i in range(start_frame, start_frame + num_frames)]
    else:
        poses = [np.eye(4) for _ in range(num_frames)]
    
    # Compute actions
    actions = []
    for i in range(1, len(poses)):
        T_prev = poses[i-1]
        T_curr = poses[i]
        T_rel = np.linalg.inv(T_prev) @ T_curr
        actions.append(T_rel.flatten())
    actions.append(np.eye(4).flatten())
    actions = np.stack(actions)
    
    return tokens, torch.from_numpy(actions).float(), poses


@torch.no_grad()
def predict_future(model, past_tokens, past_actions, num_future, cfg, device):
    """Predict future tokens using greedy decoding."""
    T_past, H, W = past_tokens.shape
    N = H * W
    mask_id = cfg.codebook_size
    
    future_tokens = torch.full((num_future, H, W), mask_id, dtype=torch.long, device=device)
    
    last_action = past_actions[-1:] if len(past_actions) > 0 else torch.zeros(1, 16)
    future_actions = last_action.repeat(num_future, 1).to(device)
    
    full_tokens = torch.cat([past_tokens.to(device), future_tokens], dim=0)
    full_actions = torch.cat([past_actions.to(device), future_actions], dim=0)
    
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
    temporal_mask = torch.triu(torch.ones(T_total, T_total) * float('-inf'), diagonal=1).to(device)
    
    # Store confidences
    all_confidences = []
    
    num_steps = cfg.num_sampling_steps
    
    for step in range(num_steps):
        logits = model(full_tokens.unsqueeze(0), full_actions.unsqueeze(0), temporal_mask)
        logits_future = logits[0, T_past:]  # (T_future, N, V)
        
        # Get predictions and confidences
        probs = torch.softmax(logits_future, dim=-1)
        confidences = probs.max(dim=-1).values  # (T_future, N)
        all_confidences.append(confidences.cpu())
        
        predictions = logits_future.argmax(dim=-1)
        future_flat = future_tokens.reshape(num_future, N)
        mask = (future_flat == mask_id)
        future_flat = torch.where(mask, predictions, future_flat)
        future_tokens = future_flat.reshape(num_future, H, W)
        full_tokens = torch.cat([full_tokens[:T_past], future_tokens], dim=0)
        
        if step < num_steps - 1:
            t = (step + 1) / num_steps
            mask_ratio = cosine_mask_schedule(torch.tensor(t)).item()
            num_to_mask = int(mask_ratio * N * num_future)
            
            if num_to_mask > 0:
                flat_conf = confidences.reshape(-1)
                _, indices = torch.topk(flat_conf, k=num_to_mask, largest=False)
                future_flat = future_tokens.reshape(-1)
                future_flat[indices] = mask_id
                future_tokens = future_flat.reshape(num_future, H, W)
                full_tokens = torch.cat([full_tokens[:T_past], future_tokens], dim=0)
    
    # Average confidences across steps
    avg_confidences = torch.stack(all_confidences).mean(dim=0)
    
    return future_tokens.cpu(), avg_confidences


def compute_token_metrics(pred_tokens, gt_tokens):
    """Compute comprehensive token-level metrics."""
    T, H, W = pred_tokens.shape
    N = H * W
    
    metrics = {
        'overall_accuracy': (pred_tokens == gt_tokens).float().mean().item() * 100,
        'per_frame_accuracy': [],
        'spatial_accuracy_map': (pred_tokens == gt_tokens).float().mean(dim=0).numpy(),
    }
    
    for t in range(T):
        acc = (pred_tokens[t] == gt_tokens[t]).float().mean().item() * 100
        metrics['per_frame_accuracy'].append(acc)
    
    return metrics


def visualize_results(past_tokens, pred_tokens, gt_tokens, confidences, metrics, output_dir):
    """Create comprehensive visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    T_future = pred_tokens.shape[0]
    T_past = past_tokens.shape[0]
    
    # 1. Main comparison grid
    fig = plt.figure(figsize=(18, 4 * (T_past + T_future)))
    gs = gridspec.GridSpec(T_past + T_future, 4, figure=fig, 
                           width_ratios=[1, 1, 1, 0.8],
                           wspace=0.3, hspace=0.4)
    
    for i in range(T_past):
        # Past frames
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(past_tokens[i].numpy(), cmap='tab20', vmin=0, vmax=1023)
        ax.set_title(f'Past Frame {i}', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        for j in range(1, 4):
            ax = fig.add_subplot(gs[i, j])
            ax.axis('off')
    
    for i in range(T_future):
        row = T_past + i
        
        # Ground truth
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(gt_tokens[i].numpy(), cmap='tab20', vmin=0, vmax=1023)
        ax.set_title(f'GT Future {i}', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Prediction
        ax = fig.add_subplot(gs[row, 1])
        ax.imshow(pred_tokens[i].numpy(), cmap='tab20', vmin=0, vmax=1023)
        ax.set_title(f'Predicted {i}', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Accuracy map
        ax = fig.add_subplot(gs[row, 2])
        match = (pred_tokens[i] == gt_tokens[i]).float().numpy()
        im = ax.imshow(match, cmap='RdYlGn', vmin=0, vmax=1)
        acc = match.mean() * 100
        ax.set_title(f'Accuracy: {acc:.1f}%', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Confidence map
        ax = fig.add_subplot(gs[row, 3])
        conf = confidences[i].reshape(pred_tokens.shape[1], pred_tokens.shape[2]).numpy()
        im = ax.imshow(conf, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'Confidence\n(mean: {conf.mean():.2f})', fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    fig.suptitle(f'World Model Prediction Comparison\nOverall Token Accuracy: {metrics["overall_accuracy"]:.1f}%', 
                 fontsize=14, fontweight='bold')
    plt.savefig(output_dir / 'comparison_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Accuracy over time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Per-frame accuracy
    frames = list(range(T_future))
    ax1.plot(frames, metrics['per_frame_accuracy'], 'o-', linewidth=2, markersize=8)
    ax1.axhline(metrics['overall_accuracy'], color='r', linestyle='--', 
                label=f'Overall: {metrics["overall_accuracy"]:.1f}%')
    ax1.set_xlabel('Future Frame')
    ax1.set_ylabel('Token Accuracy (%)')
    ax1.set_title('Accuracy Degradation Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 100)
    
    # Spatial accuracy heatmap (average across future frames)
    spatial_acc = metrics['spatial_accuracy_map']
    im = ax2.imshow(spatial_acc * 100, cmap='RdYlGn', vmin=0, vmax=100)
    ax2.set_title('Spatial Accuracy Map\n(Averaged over future frames)')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, label='Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Error distribution
    fig, axes = plt.subplots(1, T_future, figsize=(6*T_future, 5))
    if T_future == 1:
        axes = [axes]
    
    for i in range(T_future):
        ax = axes[i]
        errors = (pred_tokens[i] != gt_tokens[i]).float().numpy().flatten()
        
        # Create colored bars manually
        correct_count = (errors == 0).sum()
        wrong_count = (errors == 1).sum()
        
        bars = ax.bar(['Correct', 'Wrong'], [correct_count, wrong_count], 
                      color=['green', 'red'], edgecolor='black', alpha=0.7)
        ax.set_ylabel('Count')
        ax.set_title(f'Frame {i}: {metrics["per_frame_accuracy"][i]:.1f}% correct')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Confidence vs Accuracy scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    
    all_confs = []
    all_correct = []
    
    for i in range(T_future):
        conf_flat = confidences[i].reshape(-1).numpy()
        correct_flat = (pred_tokens[i] == gt_tokens[i]).float().reshape(-1).numpy()
        all_confs.extend(conf_flat)
        all_correct.extend(correct_flat)
    
    # Bin by confidence
    conf_bins = np.linspace(0, 1, 11)
    bin_accs = []
    bin_counts = []
    
    for i in range(len(conf_bins)-1):
        mask = (np.array(all_confs) >= conf_bins[i]) & (np.array(all_confs) < conf_bins[i+1])
        if mask.sum() > 0:
            bin_accs.append(np.array(all_correct)[mask].mean() * 100)
            bin_counts.append(mask.sum())
        else:
            bin_accs.append(0)
            bin_counts.append(0)
    
    bin_centers = (conf_bins[:-1] + conf_bins[1:]) / 2
    ax.bar(bin_centers, bin_accs, width=0.08, edgecolor='black')
    ax.set_xlabel('Prediction Confidence')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy vs Confidence Calibration')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_calibration.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--token_dir", default="data/kitti/tokens")
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--start_frame", type=int, default=500)
    parser.add_argument("--num_past_frames", type=int, default=2)
    parser.add_argument("--num_future_frames", type=int, default=3)
    parser.add_argument("--output_dir", default="outputs/world_model_eval")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load model
    world_model, wm_cfg = load_world_model(args.checkpoint, device)
    
    # Load data
    total_frames = args.num_past_frames + args.num_future_frames
    tokens, actions, poses = load_sequence_data(
        args.token_dir, args.sequence, args.start_frame, total_frames
    )
    
    past_tokens = tokens[:args.num_past_frames]
    past_actions = actions[:args.num_past_frames]
    gt_future_tokens = tokens[args.num_past_frames:]
    
    print(f"\nSetup:")
    print(f"  Past frames: {past_tokens.shape}")
    print(f"  Future frames (GT): {gt_future_tokens.shape}")
    
    # Predict
    print(f"\nPredicting {args.num_future_frames} future frames...")
    pred_tokens, confidences = predict_future(
        world_model, past_tokens, past_actions,
        args.num_future_frames, wm_cfg, device
    )
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_token_metrics(pred_tokens, gt_future_tokens)
    
    print(f"\nResults:")
    print(f"  Overall Token Accuracy: {metrics['overall_accuracy']:.2f}%")
    for i, acc in enumerate(metrics['per_frame_accuracy']):
        print(f"  Frame {i}: {acc:.2f}%")
    
    # Visualize
    print("\nCreating visualizations...")
    visualize_results(past_tokens, pred_tokens, gt_future_tokens, 
                     confidences, metrics, args.output_dir)
    
    # Save metrics
    with open(Path(args.output_dir) / 'metrics.txt', 'w') as f:
        f.write("World Model Evaluation\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Sequence: {args.sequence}\n")
        f.write(f"Start frame: {args.start_frame}\n")
        f.write(f"Past frames: {args.num_past_frames}\n")
        f.write(f"Future frames: {args.num_future_frames}\n\n")
        
        f.write(f"Overall Token Accuracy: {metrics['overall_accuracy']:.2f}%\n\n")
        
        f.write("Per-Frame Accuracy:\n")
        for i, acc in enumerate(metrics['per_frame_accuracy']):
            f.write(f"  Frame {i}: {acc:.2f}%\n")
    
    print(f"\n✅ Evaluation complete!")
    print(f"   Results saved to: {args.output_dir}")
    print(f"   - comparison_grid.png: Main visualization")
    print(f"   - accuracy_analysis.png: Accuracy over time and spatial map")
    print(f"   - error_distribution.png: Error histograms")
    print(f"   - confidence_calibration.png: Confidence vs accuracy")
    print(f"   - metrics.txt: Detailed metrics")


if __name__ == "__main__":
    main()
