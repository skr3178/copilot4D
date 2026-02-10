"""Generate validation samples and metrics from saved checkpoint.

Loads a checkpoint and generates:
1. Future prediction samples with ground truth comparison
2. Pure generation samples
3. Quantitative metrics (MSE, PSNR, SSIM)
4. Comparison visualizations
"""

import os
import sys
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.mnist_diffusion.simple_model import SimpleVideoTransformer
from tests.mnist_diffusion.moving_mnist_precomputed import MovingMNISTPrecomputed


def compute_mse(pred, target):
    """Compute MSE between predicted and target frames."""
    return ((pred - target) ** 2).mean()


def compute_psnr(pred, target):
    """Compute PSNR between predicted and target frames."""
    mse = compute_mse(pred, target)
    if mse == 0:
        return float('inf')
    max_val = 1.0
    return 20 * np.log10(max_val / np.sqrt(mse))


def compute_ssim(pred, target):
    """Compute SSIM between predicted and target frames (simplified)."""
    mu1 = pred.mean()
    mu2 = target.mean()
    sigma1 = pred.std()
    sigma2 = target.std()
    sigma12 = ((pred - mu1) * (target - mu2)).mean()
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
    return ssim


def tokens_to_video(tokens, levels=16):
    """Convert token indices back to video frames."""
    frames = tokens.astype(np.float32) / (levels - 1)
    return frames


def cosine_schedule(t: float) -> float:
    """Cosine mask schedule: gamma(t) = cos(t * pi/2)."""
    return math.cos(t * math.pi / 2)


@torch.no_grad()
def predict_future(
    model,
    past_frames: torch.Tensor,
    future_actions: torch.Tensor,
    num_steps: int = 20,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Predict future frames given past context."""
    model.eval()
    
    B, T_past, H, W = past_frames.shape
    T_future = future_actions.shape[1]
    device = past_frames.device
    mask_token_id = model.mask_token_id
    N = H * W
    
    future_tokens = torch.full((B, T_future, H, W), mask_token_id, 
                               dtype=torch.long, device=device)
    
    past_actions = torch.zeros(B, T_past, future_actions.shape[-1], device=device)
    
    all_tokens = torch.cat([past_frames, future_tokens], dim=1)
    all_actions = torch.cat([past_actions, future_actions], dim=1)
    T_total = T_past + T_future
    
    is_masked = torch.zeros((B, T_total, H, W), dtype=torch.bool, device=device)
    is_masked[:, T_past:] = True
    
    K = num_steps
    causal_mask = torch.full((T_total, T_total), float("-inf"), device=device)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    
    for step_idx in range(K):
        paper_k = K - 1 - step_idx
        num_keep = math.ceil(cosine_schedule(paper_k / K) * N)
        
        logits = model(all_tokens, all_actions, causal_mask)
        
        future_logits = logits[:, T_past:]
        
        B_curr, T_f, N_tok, V = future_logits.shape
        future_logits_flat = future_logits.reshape(B_curr * T_f * N_tok, V)
        future_logits_flat[:, mask_token_id] = float('-inf')
        future_logits_flat = future_logits_flat / temperature
        
        probs = F.softmax(future_logits_flat, dim=-1)
        sampled_flat = torch.multinomial(probs, num_samples=1).squeeze(-1)
        sampled = sampled_flat.reshape(B_curr, T_f, N_tok)
        
        log_probs = F.log_softmax(future_logits, dim=-1)
        selected_log_probs = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        
        gumbel_scale = paper_k / K if K > 0 else 0
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(selected_log_probs) + 1e-10) + 1e-10)
        confidence = selected_log_probs + gumbel_noise * gumbel_scale
        
        confidence[~is_masked[:, T_past:].reshape(B_curr, T_f, N_tok)] = float('inf')
        
        # Update future tokens (flatten for consistent indexing)
        future_flat = all_tokens[:, T_past:].reshape(B_curr, T_f * N_tok)
        sampled_flat = sampled.reshape(B_curr, T_f * N_tok)
        future_flat = torch.where(is_masked[:, T_past:].reshape(B_curr, T_f * N_tok), sampled_flat, future_flat)
        
        if paper_k > 0:
            conf_flat = confidence.reshape(B_curr, T_f * N_tok)
            _, top_indices = torch.topk(conf_flat, k=num_keep, dim=-1)
            
            # Create update mask: positions to unmask (top-M AND currently masked)
            update_mask = torch.zeros((B_curr, T_f * N_tok), dtype=torch.bool, device=device)
            update_mask.scatter_(1, top_indices, True)
            current_mask_flat = is_masked[:, T_past:].reshape(B_curr, T_f * N_tok)
            update_mask &= current_mask_flat  # Only update masked positions
            
            # Update: keep previously unmasked tokens, only change selected masked ones
            future_flat = torch.where(update_mask, sampled_flat, future_flat)
            
            # Update mask tracking: unmask the positions we just filled
            new_mask_flat = current_mask_flat.clone()
            new_mask_flat &= ~update_mask
            is_masked[:, T_past:] = new_mask_flat.reshape(B_curr, T_f, H, W)
        else:
            # Last step: use all sampled tokens
            future_flat = sampled_flat
            is_masked[:, T_past:] = False
        
        all_tokens[:, T_past:] = future_flat.reshape(B_curr, T_f, H, W)
    
    return all_tokens[:, T_past:]


@torch.no_grad()
def generate_video(
    model,
    num_frames: int,
    height: int,
    width: int,
    num_steps: int = 20,
    device: str = 'cuda',
    temperature: float = 1.0,
    action_dim: int = 2,
) -> torch.Tensor:
    """Generate a video from scratch using Algorithm 2."""
    model.eval()
    
    B = 1
    mask_token_id = model.mask_token_id
    N = height * width
    
    tokens = torch.full((B, num_frames, height, width), mask_token_id, 
                        dtype=torch.long, device=device)
    
    actions = torch.zeros(B, num_frames, action_dim, device=device)
    actions[:, :, 1] = 1.0
    
    is_masked = torch.ones((B, num_frames, height, width), dtype=torch.bool, device=device)
    
    K = num_steps
    causal_mask = torch.full((num_frames, num_frames), float("-inf"), device=device)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    
    for step_idx in range(K):
        paper_k = K - 1 - step_idx
        num_keep = math.ceil(cosine_schedule(paper_k / K) * N)
        
        logits = model(tokens, actions, causal_mask)
        
        B_curr, T, N_total, V = logits.shape
        logits_flat = logits.reshape(B_curr * T * N_total, V)
        logits_flat[:, mask_token_id] = float('-inf')
        logits_flat = logits_flat / temperature
        
        probs = F.softmax(logits_flat, dim=-1)
        sampled_flat = torch.multinomial(probs, num_samples=1).squeeze(-1)
        sampled = sampled_flat.reshape(B_curr, T, N_total)
        
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        
        gumbel_scale = paper_k / K if K > 0 else 0
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(selected_log_probs) + 1e-10) + 1e-10)
        confidence = selected_log_probs + gumbel_noise * gumbel_scale
        
        confidence[~is_masked.reshape(B_curr, T, N_total)] = float('inf')
        
        tokens_flat = tokens.reshape(B_curr, T * N_total)
        sampled_flat = sampled.reshape(B_curr, T * N_total)
        tokens_flat = torch.where(is_masked.reshape(B_curr, T * N_total), sampled_flat, tokens_flat)
        
        if paper_k > 0:
            confidence_flat = confidence.reshape(B_curr, T * N_total)
            _, top_indices = torch.topk(confidence_flat, k=num_keep, dim=-1)
            
            # Create update mask: positions to unmask (top-M AND currently masked)
            update_mask = torch.zeros((B_curr, T * N_total), dtype=torch.bool, device=device)
            update_mask.scatter_(1, top_indices, True)
            current_mask_flat = is_masked.reshape(B_curr, T * N_total)
            update_mask &= current_mask_flat  # Only update masked positions
            
            # Update: keep previously unmasked tokens, only change selected masked ones
            tokens_flat = torch.where(update_mask, sampled_flat, tokens_flat)
            
            # Update mask tracking: unmask the positions we just filled
            new_mask_flat = current_mask_flat.clone()
            new_mask_flat &= ~update_mask
            is_masked = new_mask_flat.reshape(B_curr, T, height, width)
        else:
            # Last step: use all sampled tokens
            tokens_flat = sampled_flat
            is_masked = torch.zeros((B_curr, T, height, width), dtype=torch.bool, device=device)
        
        tokens = tokens_flat.reshape(B_curr, T, height, width)
    
    return tokens


def save_comparison_grid(
    gt_video: np.ndarray,
    pred_video: np.ndarray,
    save_path: str,
    title: str = "",
    num_frames: int = 10,
):
    """Save a comparison grid of ground truth vs predicted frames."""
    T = len(gt_video) if gt_video is not None else len(pred_video)
    indices = np.linspace(0, T-1, min(num_frames, T), dtype=int)
    
    cols = len(indices)
    rows = 2 if gt_video is not None else 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.2, rows*2.5))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, frame_idx in enumerate(indices):
        if gt_video is not None:
            axes[0, idx].imshow(gt_video[frame_idx], cmap='gray', vmin=0, vmax=1)
            axes[0, idx].set_title(f"GT Frame {frame_idx}")
            axes[0, idx].axis('off')
            
            axes[1, idx].imshow(pred_video[frame_idx], cmap='gray', vmin=0, vmax=1)
            axes[1, idx].set_title(f"Pred Frame {frame_idx}")
            axes[1, idx].axis('off')
        else:
            axes[0, idx].imshow(pred_video[frame_idx], cmap='gray', vmin=0, vmax=1)
            axes[0, idx].set_title(f"Frame {frame_idx}")
            axes[0, idx].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def save_metrics_analysis(
    all_metrics: List[Dict],
    per_frame_metrics: List[Dict],
    output_dir: Path,
    mode: str = "future_prediction"
):
    """Save metrics analysis plots and text file."""
    # Extract metrics
    all_mse = [m['mse'] for m in all_metrics]
    all_psnr = [m['psnr'] for m in all_metrics]
    all_ssim = [m['ssim'] for m in all_metrics]
    
    # Per-frame metrics
    num_frames = len(per_frame_metrics[0]['mse'])
    per_frame_mse = np.array([[m['mse'][t] for t in range(num_frames)] for m in per_frame_metrics])
    per_frame_psnr = np.array([[m['psnr'][t] for t in range(num_frames)] for m in per_frame_metrics])
    
    mean_mse_t = per_frame_mse.mean(axis=0)
    std_mse_t = per_frame_mse.std(axis=0)
    mean_psnr_t = per_frame_psnr.mean(axis=0)
    std_psnr_t = per_frame_psnr.std(axis=0)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(all_mse, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('MSE')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'MSE Distribution (μ={np.mean(all_mse):.4f})')
    axes[0, 0].axvline(np.mean(all_mse), color='r', linestyle='--', label='Mean')
    axes[0, 0].legend()
    
    axes[0, 1].hist(all_psnr, bins=20, alpha=0.7, edgecolor='black', color='green')
    axes[0, 1].set_xlabel('PSNR (dB)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title(f'PSNR Distribution (μ={np.mean(all_psnr):.2f} dB)')
    axes[0, 1].axvline(np.mean(all_psnr), color='r', linestyle='--', label='Mean')
    axes[0, 1].legend()
    
    t_range = np.arange(num_frames)
    axes[1, 0].plot(t_range, mean_mse_t, 'b-', linewidth=2)
    axes[1, 0].fill_between(t_range, mean_mse_t - std_mse_t, mean_mse_t + std_mse_t, alpha=0.3)
    axes[1, 0].set_xlabel('Frame Index')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('Per-Frame MSE Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(t_range, mean_psnr_t, 'g-', linewidth=2)
    axes[1, 1].fill_between(t_range, mean_psnr_t - std_psnr_t, mean_psnr_t + std_psnr_t, alpha=0.3, color='green')
    axes[1, 1].set_xlabel('Frame Index')
    axes[1, 1].set_ylabel('PSNR (dB)')
    axes[1, 1].set_title('Per-Frame PSNR Over Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'metrics_analysis.png'}")
    
    # Save metrics to file
    with open(output_dir / "metrics.txt", 'w') as f:
        f.write(f"Metrics Summary ({mode})\n")
        f.write("="*60 + "\n")
        f.write(f"Number of samples: {len(all_mse)}\n\n")
        f.write(f"MSE:  {np.mean(all_mse):.6f} ± {np.std(all_mse):.6f}\n")
        f.write(f"PSNR: {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB\n")
        f.write(f"SSIM: {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}\n\n")
        f.write("Per-frame metrics:\n")
        f.write("Frame | MSE | PSNR\n")
        for t in range(num_frames):
            f.write(f"{t:5d} | {mean_mse_t[t]:.6f} | {mean_psnr_t[t]:.2f}\n")
    
    print(f"  Saved: {output_dir / 'metrics.txt'}")


def evaluate_future_prediction(
    model,
    val_dataset,
    num_samples: int = 50,
    num_past_frames: int = 10,
    num_sampling_steps: int = 20,
    device: str = 'cuda',
    output_dir: Path = None,
    levels: int = 16,
):
    """Evaluate future prediction with comprehensive metrics."""
    print("\n" + "="*60)
    print(f"Evaluating Future Prediction ({num_samples} samples)")
    print(f"  Sampling steps: {num_sampling_steps}")
    print("="*60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    per_frame_metrics = []
    
    for i in tqdm(range(num_samples), desc="Predicting"):
        sample = val_dataset[i]
        gt_tokens = sample["tokens"]
        actions = sample["actions"]
        
        # Handle both tensor and numpy array returns
        if isinstance(gt_tokens, torch.Tensor):
            past_frames = gt_tokens[:num_past_frames].unsqueeze(0).to(device)
            future_actions = actions[num_past_frames:].unsqueeze(0).to(device)
            gt_future = gt_tokens[num_past_frames:].cpu().numpy()
        else:
            past_frames = torch.from_numpy(gt_tokens[:num_past_frames]).unsqueeze(0).to(device)
            future_actions = torch.from_numpy(actions[num_past_frames:]).unsqueeze(0).to(device)
            gt_future = gt_tokens[num_past_frames:]
        
        pred_future = predict_future(
            model,
            past_frames,
            future_actions,
            num_steps=num_sampling_steps,
            temperature=1.0,
        )
        
        gt_past = gt_tokens[:num_past_frames].cpu().numpy() if isinstance(gt_tokens, torch.Tensor) else gt_tokens[:num_past_frames]
        gt_full = np.concatenate([gt_past, gt_future], axis=0)
        pred_full = np.concatenate([
            gt_past,
            pred_future[0].cpu().numpy()
        ], axis=0)
        
        gt_video = tokens_to_video(gt_full, levels=levels)
        pred_video = tokens_to_video(pred_full, levels=levels)
        
        # Compute per-frame metrics
        future_metrics = {'mse': [], 'psnr': [], 'ssim': []}
        for t in range(len(gt_future)):
            mse = compute_mse(pred_video[num_past_frames + t], gt_video[num_past_frames + t])
            psnr = compute_psnr(pred_video[num_past_frames + t], gt_video[num_past_frames + t])
            ssim = compute_ssim(pred_video[num_past_frames + t], gt_video[num_past_frames + t])
            future_metrics['mse'].append(mse)
            future_metrics['psnr'].append(psnr)
            future_metrics['ssim'].append(ssim)
        
        avg_metrics = {
            'mse': np.mean(future_metrics['mse']),
            'psnr': np.mean(future_metrics['psnr']),
            'ssim': np.mean(future_metrics['ssim']),
        }
        all_metrics.append(avg_metrics)
        per_frame_metrics.append(future_metrics)
        
        # Save comparison for first 10 samples
        if i < 10:
            save_comparison_grid(
                gt_video, pred_video,
                str(output_dir / f"sample_{i:03d}.png"),
                title=f"Sample {i}: Ground Truth (top) vs Predicted (bottom)"
            )
            
            # Save numpy arrays
            np.save(output_dir / f"sample_{i:03d}_real.npy", gt_video)
            np.save(output_dir / f"sample_{i:03d}_gen.npy", pred_video)
    
    # Compute and save summary metrics
    avg_mse = np.mean([m['mse'] for m in all_metrics])
    avg_psnr = np.mean([m['psnr'] for m in all_metrics])
    avg_ssim = np.mean([m['ssim'] for m in all_metrics])
    
    print("\n" + "="*60)
    print("Future Prediction Results (Future Frames Only)")
    print("="*60)
    print(f"  MSE:  {avg_mse:.6f}")
    print(f"  PSNR: {avg_psnr:.2f} dB")
    print(f"  SSIM: {avg_ssim:.4f}")
    print("="*60)
    
    # Save analysis
    save_metrics_analysis(all_metrics, per_frame_metrics, output_dir, mode="future_prediction")
    
    return avg_mse, avg_psnr, avg_ssim


def evaluate_generation(
    model,
    num_samples: int = 10,
    num_frames: int = 20,
    height: int = 32,
    width: int = 32,
    num_sampling_steps: int = 20,
    device: str = 'cuda',
    output_dir: Path = None,
    levels: int = 16,
):
    """Evaluate pure generation."""
    print("\n" + "="*60)
    print(f"Evaluating Pure Generation ({num_samples} samples)")
    print(f"  Sampling steps: {num_sampling_steps}")
    print("="*60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in tqdm(range(num_samples), desc="Generating"):
        gen_tokens = generate_video(
            model,
            num_frames=num_frames,
            height=height,
            width=width,
            num_steps=num_sampling_steps,
            device=device,
            temperature=1.0,
        )
        
        gen_video = tokens_to_video(gen_tokens[0].cpu().numpy(), levels=levels)
        
        save_comparison_grid(
            None, gen_video,
            str(output_dir / f"sample_{i:03d}.png"),
            title=f"Generated Sample {i}",
            num_frames=10
        )
        
        np.save(output_dir / f"sample_{i:03d}_gen.npy", gen_video)
    
    print(f"\nGeneration samples saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate validation results from checkpoint")
    
    parser.add_argument("--checkpoint", type=str, 
                       default="outputs/mnist_diffusion_full/checkpoint_epoch1.pt",
                       help="Path to checkpoint")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA model if available")
    
    # Data config (must match training)
    parser.add_argument("--data_path", type=str, default="mnist_test_seq.1.npy")
    parser.add_argument("--num_token_levels", type=int, default=16)
    parser.add_argument("--frame_size", type=int, default=32)
    parser.add_argument("--num_past_frames", type=int, default=10)
    
    # Model config (must match training)
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=6)
    
    # Evaluation settings
    parser.add_argument("--num_sampling_steps", type=int, default=20)
    parser.add_argument("--num_pred_samples", type=int, default=50)
    parser.add_argument("--num_gen_samples", type=int, default=10)
    
    # What to evaluate
    parser.add_argument("--skip_prediction", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/mnist_diffusion_full/validation")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint info
    print("\n" + "="*60)
    print("Loading checkpoint...")
    print(f"  Path: {args.checkpoint}")
    print("="*60)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"  Epoch: {epoch}")
    
    # Create model
    model = SimpleVideoTransformer(
        vocab_size=args.num_token_levels + 1,
        mask_token_id=args.num_token_levels,
        action_dim=2,
        num_frames=20,
        height=args.frame_size,
        width=args.frame_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=0.0,
    ).to(device)
    
    # Load weights
    if args.use_ema and "ema" in checkpoint:
        print("  Using EMA model weights")
        model.load_state_dict(checkpoint["ema"]["ema_model"])
    else:
        print("  Using regular model weights")
        model.load_state_dict(checkpoint["model_state_dict"])
    
    print("  Model loaded successfully")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load validation dataset
    print("\n" + "="*60)
    print("Loading validation dataset...")
    print("="*60)
    
    val_dataset = MovingMNISTPrecomputed(
        data_path=args.data_path,
        seq_len=20,
        frame_size=args.frame_size,
        num_token_levels=args.num_token_levels,
        num_sequences=2000,
        start_idx=8000,
        use_ego_centric=True,
        ego_digit_id=0,
    )
    print(f"  Val samples: {len(val_dataset)}")
    
    # Run evaluation
    if not args.skip_prediction:
        evaluate_future_prediction(
            model,
            val_dataset,
            num_samples=args.num_pred_samples,
            num_past_frames=args.num_past_frames,
            num_sampling_steps=args.num_sampling_steps,
            device=device,
            output_dir=output_dir / "future_pred",
            levels=args.num_token_levels,
        )
    
    if not args.skip_generation:
        evaluate_generation(
            model,
            num_samples=args.num_gen_samples,
            num_frames=20,
            height=args.frame_size,
            width=args.frame_size,
            num_sampling_steps=args.num_sampling_steps,
            device=device,
            output_dir=output_dir / "generation",
            levels=args.num_token_levels,
        )
    
    print("\n" + "="*60)
    print("Validation complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
