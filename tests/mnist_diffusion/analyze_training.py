#!/usr/bin/env python
"""
Analyze training progress: plot loss curves and generate visualizations.
"""
import re
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generate_validation_results import (
    tokens_to_video, predict_future, generate_video,
    save_comparison_grid, MovingMNISTPrecomputed
)
from train_mnist_full import SimpleVideoTransformer


def parse_training_log(log_path):
    """Parse training log to extract loss values."""
    train_losses = []
    val_losses = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Training iteration loss: loss=X.XXXX
            train_match = re.search(r'loss=([0-9.]+)', line)
            if train_match:
                train_losses.append(float(train_match.group(1)))
            
            # Validation loss: Val Loss: X.XXXX
            val_match = re.search(r'Val Loss:\s+([0-9.]+)', line)
            if val_match:
                val_losses.append(float(val_match.group(1)))
    
    return np.array(train_losses), np.array(val_losses)


def plot_loss_curves(train_losses, val_losses, output_path):
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training loss over iterations
    ax1 = axes[0]
    iterations = np.arange(len(train_losses))
    ax1.plot(iterations, train_losses, alpha=0.3, color='blue', linewidth=0.5)
    
    # Smoothed curve (moving average)
    window = 100
    if len(train_losses) > window:
        smoothed = np.convolve(train_losses, np.ones(window)/window, mode='valid')
        ax1.plot(iterations[window-1:], smoothed, color='blue', linewidth=2, label=f'MA({window})')
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Validation loss over epochs
    ax2 = axes[1]
    if len(val_losses) > 0:
        epochs = np.arange(1, len(val_losses) + 1)
        ax2.plot(epochs, val_losses, 'o-', color='red', linewidth=2, markersize=8)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Validation Loss per Epoch')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Loss curves saved to: {output_path}")
    plt.close()
    
    return fig


def load_model_from_checkpoint(checkpoint_path, device, args):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    epoch = checkpoint.get('epoch', 'unknown')
    
    # Create model
    model = SimpleVideoTransformer(
        vocab_size=args.num_token_levels,
        mask_token_id=args.num_token_levels,
        num_frames=20,
        height=args.frame_size,
        width=args.frame_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=0.0,
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, checkpoint, epoch


def generate_reconstruction_comparison(checkpoint_path, data_path, output_dir, args, num_samples=5):
    """Generate reconstruction samples showing input, predicted, and ground truth."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, _, epoch = load_model_from_checkpoint(checkpoint_path, device, args)
    print(f"Loaded model from epoch {epoch}")
    
    # Load data
    val_dataset = MovingMNISTPrecomputed(
        data_path=data_path,
        seq_len=20,
        frame_size=args.frame_size,
        num_token_levels=args.num_token_levels,
        num_sequences=2000,
        start_idx=8000,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(num_samples, 6, figsize=(18, 3*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if idx >= num_samples:
                break
            
            # Handle dict or tuple return
            if isinstance(batch, dict):
                frames = batch['frames']
            else:
                frames = batch[0]
            
            # Split into past and future
            past_frames = frames[:, :args.num_past_frames]
            gt_future = frames[:, args.num_past_frames:]
            
            past_frames = past_frames.to(device)
            gt_future = gt_future.to(device)
            
            B, T_past, H, W = past_frames.shape
            T_future = gt_future.shape[1]
            
            # Tokenize past frames: quantize float [0,1] to int tokens [0, num_levels-1]
            past_tokens = (past_frames * (args.num_token_levels - 1)).long()
            
            # Create dummy future actions
            future_actions = torch.zeros(1, T_future, 4, device=device)
            
            # Predict future
            pred_tokens = predict_future(
                model, past_tokens, future_actions,
                num_steps=args.num_sampling_steps,
                temperature=1.0
            )
            
            # Convert tokens back to frames
            pred_future = pred_tokens.float() / (args.num_token_levels - 1)
            
            # Show: past[0], past[-1], gt_future[0], gt_future[-1], pred_future[0], pred_future[-1]
            images = [
                past_frames[0, 0].cpu().numpy(),
                past_frames[0, -1].cpu().numpy(),
                gt_future[0, 0].cpu().numpy(),
                gt_future[0, -1].cpu().numpy(),
                pred_future[0, 0].cpu().numpy(),
                pred_future[0, -1].cpu().numpy(),
            ]
            
            titles = [
                'Past First', 'Past Last',
                'GT Future First', 'GT Future Last',
                'Pred Future First', 'Pred Future Last'
            ]
            
            for j, (img, title) in enumerate(zip(images, titles)):
                ax = axes[idx, j]
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                if idx == 0:
                    ax.set_title(title, fontsize=10)
                ax.axis('off')
    
    plt.suptitle(f'Reconstruction Comparison (Epoch {epoch})', fontsize=14)
    plt.tight_layout()
    output_path = output_dir / 'reconstruction_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Reconstruction comparison saved to: {output_path}")
    plt.close()


def generate_denoising_steps_visualization(checkpoint_path, data_path, output_dir, args):
    """Visualize the denoising process over sampling steps."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, _, epoch = load_model_from_checkpoint(checkpoint_path, device, args)
    
    # Load single sample
    val_dataset = MovingMNISTPrecomputed(
        data_path=data_path,
        seq_len=20,
        frame_size=args.frame_size,
        num_token_levels=args.num_token_levels,
        num_sequences=2000,
        start_idx=8000,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    batch = next(iter(val_loader))
    if isinstance(batch, dict):
        frames = batch['frames']
    else:
        frames = batch[0]
    
    past_frames = frames[:, :args.num_past_frames]
    gt_future = frames[:, args.num_past_frames:]
    
    past_frames = past_frames.to(device)
    gt_future = gt_future.to(device)
    
    T_future = gt_future.shape[1]
    mask_token_id = args.num_token_levels
    
    # Tokenize: quantize float [0,1] to int tokens [0, num_levels-1]
    past_tokens = (past_frames * (args.num_token_levels - 1)).long()
    
    B, T_past, H, W = past_tokens.shape
    N = H * W
    
    # Initialize with all masked
    future_tokens = torch.full((B, T_future, H, W), mask_token_id, 
                               dtype=torch.long, device=device)
    
    # Combine
    tokens = torch.cat([past_tokens, future_tokens], dim=1)
    T_total = T_past + T_future
    
    # Track denoising at different steps
    steps_to_track = [0, 4, 9, 14, 19]
    snapshots = []
    
    num_steps = 20
    temperature = 1.0
    
    with torch.no_grad():
        for step in range(num_steps):
            t = 1.0 - (step / num_steps)
            
            # Forward pass
            tokens_flat = tokens.reshape(B, T_total * N)
            logits = model(tokens_flat)  # [B, T_total*N, vocab_size]
            logits = logits.reshape(B, T_total, N, args.num_token_levels)
            
            # Sample
            probs = torch.softmax(logits / temperature, dim=-1)
            sampled = torch.multinomial(probs.reshape(-1, args.num_token_levels), 1).reshape(B, T_total, N)
            
            # Create confidence (max prob)
            confidence = probs.max(dim=-1).values  # [B, T_total, N]
            
            # Determine which tokens to unmask
            mask_ratio = 1.0 - ((step + 1) / num_steps)
            is_masked = (tokens == mask_token_id)
            
            if mask_ratio > 0:
                # Among masked positions, keep some
                masked_conf = confidence * is_masked.float()
                num_masked = is_masked.sum(dim=-1, keepdim=True)
                num_keep = (mask_ratio * num_masked).long()
                
                for b in range(B):
                    for t_idx in range(T_total):
                        masked_positions = torch.where(is_masked[b, t_idx])[0]
                        if len(masked_positions) > 0:
                            k = min(num_keep[b, t_idx].item(), len(masked_positions))
                            if k > 0:
                                conf_masked = masked_conf[b, t_idx, masked_positions]
                                _, keep_indices = torch.topk(conf_masked, k)
                                keep_positions = masked_positions[keep_indices]
                                is_masked[b, t_idx, keep_positions] = False
            
            # Update: unmasked positions get sampled values
            tokens_flat = tokens.reshape(B, T_total * N)
            sampled_flat = sampled.reshape(B, T_total * N)
            is_masked_flat = is_masked.reshape(B, T_total * N)
            
            tokens_flat = torch.where(~is_masked_flat, sampled_flat, tokens_flat)
            tokens = tokens_flat.reshape(B, T_total, H, W)
            
            # Save snapshot
            if step in steps_to_track:
                snapshots.append(tokens[0, T_past:].cpu().clone())
    
    # Visualize
    fig, axes = plt.subplots(2, len(steps_to_track), figsize=(3*len(steps_to_track), 6))
    
    for i, snapshot in enumerate(snapshots):
        # First and last frame of predicted future
        for row, frame_idx in enumerate([0, -1]):
            tokens_frame = snapshot[frame_idx]
            # Convert tokens back to image
            img = tokens_frame.float() / (args.num_token_levels - 1)
            img = img.numpy()
            
            ax = axes[row, i]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            if row == 0:
                ax.set_title(f'Step {steps_to_track[i]+1}')
            ax.axis('off')
    
    axes[0, 0].set_ylabel('First Frame', rotation=0, ha='right', fontsize=10)
    axes[1, 0].set_ylabel('Last Frame', rotation=0, ha='right', fontsize=10)
    
    plt.suptitle(f'Denoising Progress - Future Prediction (Epoch {epoch})', fontsize=14)
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'denoising_progress.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Denoising progress saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', default='outputs/mnist_diffusion_full/training_nohup.log')
    parser.add_argument('--checkpoint', default='outputs/mnist_diffusion_full/best_model.pt')
    parser.add_argument('--output_dir', default='outputs/mnist_diffusion_full/analysis')
    parser.add_argument('--data_path', default='mnist_test_seq.1.npy')
    
    # Model config
    parser.add_argument('--num_token_levels', type=int, default=16)
    parser.add_argument('--frame_size', type=int, default=32)
    parser.add_argument('--num_past_frames', type=int, default=10)
    parser.add_argument('--embed_dim', type=int, default=384)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--num_sampling_steps', type=int, default=20)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Analyzing Training Progress")
    print("="*60)
    
    # 1. Parse and plot loss curves
    print("\n1. Parsing training log...")
    train_losses, val_losses = parse_training_log(args.log_path)
    print(f"   Found {len(train_losses)} training iterations")
    print(f"   Found {len(val_losses)} validation epochs")
    
    if len(train_losses) > 0:
        print("\n2. Plotting loss curves...")
        plot_loss_curves(train_losses, val_losses, output_dir / 'loss_curves.png')
    
    # 2. Generate reconstruction samples
    print("\n3. Generating reconstruction samples...")
    generate_reconstruction_comparison(
        args.checkpoint, args.data_path, output_dir, args
    )
    
    # 3. Generate denoising visualization
    print("\n4. Generating denoising steps visualization...")
    try:
        generate_denoising_steps_visualization(
            args.checkpoint, args.data_path, output_dir, args
        )
    except Exception as e:
        print(f"   Error in denoising viz: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"Analysis complete! Results in: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
