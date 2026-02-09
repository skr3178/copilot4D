"""Evaluate trained MNIST diffusion model.

Loads a checkpoint and runs comprehensive evaluation:
1. Future prediction accuracy (quantitative metrics)
2. Pure generation quality (visual inspection)
3. Sampling process visualization
"""

import os
import sys
import math
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

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
from tests.mnist_diffusion.analyze_results import compute_mse, compute_psnr, compute_ssim

def tokens_to_video(tokens, levels=16):
    """Convert token indices back to video frames."""
    # tokens: (T, H, W) with values in [0, levels-1]
    frames = tokens.astype(np.float32) / (levels - 1)
    return frames

def compute_metrics(pred, target):
    """Compute all metrics."""
    return {
        'mse': compute_mse(pred, target),
        'psnr': compute_psnr(pred, target),
        'ssim': compute_ssim(pred, target),
    }


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
    """Predict future frames given past context using Algorithm 2."""
    model.eval()
    
    B, T_past, H, W = past_frames.shape
    T_future = future_actions.shape[1]
    device = past_frames.device
    mask_token_id = model.mask_token_id
    N = H * W
    
    # Initialize future with mask tokens
    future_tokens = torch.full((B, T_future, H, W), mask_token_id, 
                               dtype=torch.long, device=device)
    
    # Create dummy past actions
    past_actions = torch.zeros(B, T_past, future_actions.shape[-1], device=device)
    
    # Combine
    all_tokens = torch.cat([past_frames, future_tokens], dim=1)
    all_actions = torch.cat([past_actions, future_actions], dim=1)
    T_total = T_past + T_future
    
    # Track masking
    is_masked = torch.zeros((B, T_total, H, W), dtype=torch.bool, device=device)
    is_masked[:, T_past:] = True
    
    K = num_steps
    causal_mask = torch.full((T_total, T_total), float("-inf"), device=device)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    
    # Iterative decoding - Algorithm 2
    for step_idx in range(K):
        paper_k = K - 1 - step_idx
        num_keep = math.ceil(cosine_schedule(paper_k / K) * N)
        
        logits = model(all_tokens, all_actions, causal_mask)
        
        # Only consider future positions
        future_logits = logits[:, T_past:]
        
        # Sample for future
        B_curr, T_f, N_tok, V = future_logits.shape
        future_logits_flat = future_logits.reshape(B_curr * T_f * N_tok, V)
        future_logits_flat[:, mask_token_id] = float('-inf')
        future_logits_flat = future_logits_flat / temperature
        
        probs = F.softmax(future_logits_flat, dim=-1)
        sampled_flat = torch.multinomial(probs, num_samples=1).squeeze(-1)
        sampled = sampled_flat.reshape(B_curr, T_f, N_tok)
        
        # Confidence
        log_probs = F.log_softmax(future_logits, dim=-1)
        selected_log_probs = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        
        gumbel_scale = paper_k / K if K > 0 else 0
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(selected_log_probs) + 1e-10) + 1e-10)
        confidence = selected_log_probs + gumbel_noise * gumbel_scale
        
        confidence[~is_masked[:, T_past:].reshape(B_curr, T_f, N_tok)] = float('inf')
        
        # Update
        future_flat = all_tokens[:, T_past:].reshape(B_curr, T_f, N_tok)
        future_flat = torch.where(is_masked[:, T_past:].reshape(B_curr, T_f, N_tok), sampled, future_flat)
        
        if paper_k > 0:
            conf_flat = confidence.reshape(B_curr, T_f * N_tok)
            _, top_indices = torch.topk(conf_flat, k=num_keep, dim=-1)
            
            new_mask = torch.ones((B_curr, T_f * N_tok), dtype=torch.bool, device=device)
            new_mask.scatter_(1, top_indices, False)
            
            future_flat[new_mask] = mask_token_id
            is_masked[:, T_past:] = new_mask.reshape(B_curr, T_f, N_tok)
        else:
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
    action_dim: int = 4,
) -> torch.Tensor:
    """Generate a video from scratch using Algorithm 2."""
    model.eval()
    
    B = 1
    mask_token_id = model.mask_token_id
    N = height * width
    
    # Initialize with all mask tokens
    tokens = torch.full((B, num_frames, height, width), mask_token_id, 
                        dtype=torch.long, device=device)
    
    # Create actions (straight line down motion)
    actions = torch.zeros(B, num_frames, action_dim, device=device)
    actions[:, :, 1] = 1.0  # down motion
    
    # Track which positions are still masked
    is_masked = torch.ones((B, num_frames, height, width), dtype=torch.bool, device=device)
    
    K = num_steps
    causal_mask = torch.full((num_frames, num_frames), float("-inf"), device=device)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    
    # Iterative decoding - Algorithm 2
    for step_idx in range(K):
        paper_k = K - 1 - step_idx
        num_keep = math.ceil(cosine_schedule(paper_k / K) * N)
        
        logits = model(tokens, actions, causal_mask)
        
        # Sample from predictions
        B_curr, T, N_total, V = logits.shape
        logits_flat = logits.reshape(B_curr * T * N_total, V)
        
        # Exclude mask token from sampling
        logits_flat[:, mask_token_id] = float('-inf')
        
        # Apply temperature
        logits_flat = logits_flat / temperature
        
        # Sample
        probs = F.softmax(logits_flat, dim=-1)
        sampled_flat = torch.multinomial(probs, num_samples=1).squeeze(-1)
        sampled = sampled_flat.reshape(B_curr, T, N_total)
        
        # Compute confidence
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        
        gumbel_scale = paper_k / K if K > 0 else 0
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(selected_log_probs) + 1e-10) + 1e-10)
        confidence = selected_log_probs + gumbel_noise * gumbel_scale
        
        # On non-mask indices: confidence = +inf (keep unmasked)
        confidence[~is_masked.reshape(B_curr, T, N_total)] = float('inf')
        
        # Update tokens with samples
        tokens_flat = tokens.reshape(B_curr, T, N_total)
        tokens_flat = torch.where(is_masked.reshape(B_curr, T, N_total), sampled, tokens_flat)
        
        if paper_k > 0:
            # Keep top-M confident predictions unmasked
            confidence_flat = confidence.reshape(B_curr, T * N_total)
            _, top_indices = torch.topk(confidence_flat, k=num_keep, dim=-1)
            
            new_mask = torch.ones((B_curr, T * N_total), dtype=torch.bool, device=device)
            new_mask.scatter_(1, top_indices, False)
            
            # Re-mask low confidence positions
            tokens_flat[new_mask] = mask_token_id
            is_masked = new_mask.reshape(B_curr, T, N_total)
        else:
            # Last step: keep everything unmasked
            is_masked = torch.zeros((B_curr, T, N_total), dtype=torch.bool, device=device)
        
        tokens = tokens_flat.reshape(B_curr, T, height, width)
    
    return tokens


def save_video_grid(
    frames: np.ndarray,
    save_path: str,
    num_frames: int = 10,
    title: str = "",
):
    """Save a grid of video frames."""
    T = len(frames)
    indices = np.linspace(0, T-1, min(num_frames, T), dtype=int)
    
    cols = min(5, len(indices))
    rows = (len(indices) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, frame_idx in enumerate(indices):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]
        ax.imshow(frames[frame_idx], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Frame {frame_idx}")
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(len(indices), rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def evaluate_future_prediction(
    model,
    val_dataset,
    num_samples: int = 50,
    num_past_frames: int = 10,
    num_sampling_steps: int = 20,
    device: str = 'cuda',
    output_dir: str = "outputs/eval",
    levels: int = 16,
):
    """Evaluate future prediction accuracy."""
    print("\n" + "="*60)
    print(f"Evaluating Future Prediction ({num_samples} samples)")
    print(f"  Sampling steps: {num_sampling_steps}")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    
    for i in tqdm(range(num_samples), desc="Predicting"):
        # Get a sample
        sample = val_dataset[i]
        gt_tokens = sample["tokens"]
        actions = sample["actions"]
        
        # Split into past and future
        past_frames = torch.from_numpy(gt_tokens[:num_past_frames]).unsqueeze(0).to(device)
        future_actions = torch.from_numpy(actions[num_past_frames:]).unsqueeze(0).to(device)
        gt_future = gt_tokens[num_past_frames:]
        
        # Predict
        pred_future = predict_future(
            model,
            past_frames,
            future_actions,
            num_steps=num_sampling_steps,
            temperature=1.0,
        )
        
        # Convert to videos
        gt_full = np.concatenate([gt_tokens[:num_past_frames], gt_future], axis=0)
        pred_full = np.concatenate([
            gt_tokens[:num_past_frames],
            pred_future[0].cpu().numpy()
        ], axis=0)
        
        gt_video = tokens_to_video(gt_full, levels=levels)
        pred_video = tokens_to_video(pred_full, levels=levels)
        
        # Compute metrics on future frames only
        future_metrics = []
        for t in range(len(gt_future)):
            m = compute_metrics(pred_video[num_past_frames + t], gt_video[num_past_frames + t])
            future_metrics.append(m)
        
        avg_metrics = {
            'mse': np.mean([m['mse'] for m in future_metrics]),
            'psnr': np.mean([m['psnr'] for m in future_metrics]),
            'ssim': np.mean([m['ssim'] for m in future_metrics]),
        }
        all_metrics.append(avg_metrics)
        
        # Save visualization for first 10 samples
        if i < 10:
            from tests.mnist_diffusion.analyze_samples import save_comparison_grid
            save_comparison_grid(
                gt_video, pred_video,
                str(output_dir / f"pred_sample_{i:03d}.png"),
                title=f"Sample {i}: GT (top) vs Pred (bottom)"
            )
    
    # Compute average metrics
    avg_mse = np.mean([m['mse'] for m in all_metrics])
    avg_psnr = np.mean([m['psnr'] for m in all_metrics])
    avg_ssim = np.mean([m['ssim'] for m in all_metrics])
    
    print("\n" + "="*60)
    print("Future Prediction Results (Future Frames Only)")
    print("="*60)
    print(f"  MSE:  {avg_mse:.4f}")
    print(f"  PSNR: {avg_psnr:.2f} dB")
    print(f"  SSIM: {avg_ssim:.4f}")
    print("="*60)
    
    # Save metrics to file
    with open(output_dir / "metrics_future_pred.txt", "w") as f:
        f.write("Future Prediction Metrics\n")
        f.write("="*60 + "\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write(f"Past frames: {num_past_frames}\n")
        f.write(f"Sampling steps: {num_sampling_steps}\n\n")
        f.write(f"MSE:  {avg_mse:.4f}\n")
        f.write(f"PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"SSIM: {avg_ssim:.4f}\n")
    
    return avg_mse, avg_psnr, avg_ssim


def evaluate_generation(
    model,
    num_samples: int = 10,
    num_frames: int = 20,
    height: int = 32,
    width: int = 32,
    num_sampling_steps: int = 20,
    device: str = 'cuda',
    output_dir: str = "outputs/eval",
    levels: int = 16,
):
    """Evaluate pure generation quality."""
    print("\n" + "="*60)
    print(f"Evaluating Pure Generation ({num_samples} samples)")
    print(f"  Sampling steps: {num_sampling_steps}")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in tqdm(range(num_samples), desc="Generating"):
        # Generate
        gen_tokens = generate_video(
            model,
            num_frames=num_frames,
            height=height,
            width=width,
            num_steps=num_sampling_steps,
            device=device,
            temperature=1.0,
        )
        
        # Convert to video
        gen_video = tokens_to_video(gen_tokens[0].cpu().numpy(), levels=levels)
        
        # Save visualization
        save_video_grid(
            gen_video,
            str(output_dir / f"gen_sample_{i:03d}.png"),
            num_frames=10,
            title=f"Generated Sample {i}"
        )
    
    print(f"\nGeneration samples saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained MNIST diffusion model")
    
    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA model if available")
    
    # Data
    parser.add_argument("--data_path", type=str, default="mnist_test_seq.1.npy")
    parser.add_argument("--num_token_levels", type=int, default=16)
    parser.add_argument("--frame_size", type=int, default=32)
    
    # Model architecture
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=6)
    
    # Evaluation settings
    parser.add_argument("--num_sampling_steps", type=int, default=20, help="Number of diffusion steps")
    parser.add_argument("--num_past_frames", type=int, default=10)
    parser.add_argument("--num_pred_samples", type=int, default=50, help="Number of samples for prediction eval")
    parser.add_argument("--num_gen_samples", type=int, default=10, help="Number of samples for generation eval")
    
    # What to evaluate
    parser.add_argument("--eval_prediction", action="store_true", default=True)
    parser.add_argument("--eval_generation", action="store_true", default=True)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/mnist_eval")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("\n" + "="*60)
    print("Loading model...")
    print(f"  Checkpoint: {args.checkpoint}")
    print("="*60)
    
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
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if args.use_ema and "ema" in checkpoint:
        print("  Using EMA model weights")
        model.load_state_dict(checkpoint["ema"]["ema_model"])
    else:
        print("  Using regular model weights")
        model.load_state_dict(checkpoint["model_state_dict"])
    
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"  Loaded from epoch {epoch}")
    
    # Load validation dataset
    if args.eval_prediction:
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
        )
        print(f"  Val samples: {len(val_dataset)}")
    
    # Run evaluation
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.eval_prediction:
        evaluate_future_prediction(
            model,
            val_dataset,
            num_samples=args.num_pred_samples,
            num_past_frames=args.num_past_frames,
            num_sampling_steps=args.num_sampling_steps,
            device=device,
            output_dir=str(output_dir / "future_pred"),
            levels=args.num_token_levels,
        )
    
    if args.eval_generation:
        evaluate_generation(
            model,
            num_samples=args.num_gen_samples,
            num_frames=20,
            height=args.frame_size,
            width=args.frame_size,
            num_sampling_steps=args.num_sampling_steps,
            device=device,
            output_dir=str(output_dir / "generation"),
            levels=args.num_token_levels,
        )
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
