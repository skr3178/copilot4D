"""Analyze and visualize discrete diffusion results.

Computes metrics and creates comprehensive visualizations.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch


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
    # Simple SSIM approximation
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


def analyze_samples(sample_dir, mode="generation"):
    """Analyze generated samples and compute metrics.
    
    Args:
        sample_dir: Directory containing sample_*.npy files
        mode: "generation" or "future_prediction"
    """
    sample_dir = Path(sample_dir)
    
    # Find all sample files
    real_files = sorted(sample_dir.glob("*_real.npy"))
    gen_files = sorted(sample_dir.glob("*_gen.npy"))
    
    if not real_files or not gen_files:
        print(f"No samples found in {sample_dir}")
        return
    
    print(f"Found {len(real_files)} samples in {sample_dir}")
    
    # Compute metrics
    all_mse = []
    all_psnr = []
    all_ssim = []
    
    per_frame_mse = []
    per_frame_psnr = []
    
    for real_file, gen_file in zip(real_files, gen_files):
        real = np.load(real_file)  # (T, H, W)
        gen = np.load(gen_file)    # (T, H, W)
        
        # Overall metrics
        mse = compute_mse(gen, real)
        psnr = compute_psnr(gen, real)
        ssim_val = compute_ssim(gen, real)
        
        all_mse.append(mse)
        all_psnr.append(psnr)
        all_ssim.append(ssim_val)
        
        # Per-frame metrics
        sample_frame_mse = []
        sample_frame_psnr = []
        for t in range(len(real)):
            mse_t = compute_mse(gen[t], real[t])
            psnr_t = compute_psnr(gen[t], real[t])
            sample_frame_mse.append(mse_t)
            sample_frame_psnr.append(psnr_t)
        per_frame_mse.append(sample_frame_mse)
        per_frame_psnr.append(sample_frame_psnr)
    
    # Summary statistics
    print("\n" + "="*50)
    print(f"METRICS SUMMARY ({mode})")
    print("="*50)
    print(f"MSE:  {np.mean(all_mse):.6f} ± {np.std(all_mse):.6f}")
    print(f"PSNR: {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB")
    print(f"SSIM: {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
    print("="*50)
    
    # Per-frame analysis
    per_frame_mse = np.array(per_frame_mse)
    per_frame_psnr = np.array(per_frame_psnr)
    
    mean_mse_t = per_frame_mse.mean(axis=0)
    std_mse_t = per_frame_mse.std(axis=0)
    mean_psnr_t = per_frame_psnr.mean(axis=0)
    std_psnr_t = per_frame_psnr.std(axis=0)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Overall metrics histogram
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
    
    # 2. Per-frame MSE
    t_range = np.arange(len(mean_mse_t))
    axes[1, 0].plot(t_range, mean_mse_t, 'b-', linewidth=2)
    axes[1, 0].fill_between(t_range, mean_mse_t - std_mse_t, mean_mse_t + std_mse_t, alpha=0.3)
    axes[1, 0].set_xlabel('Frame Index')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('Per-Frame MSE Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 3. Per-frame PSNR
    axes[1, 1].plot(t_range, mean_psnr_t, 'g-', linewidth=2)
    axes[1, 1].fill_between(t_range, mean_psnr_t - std_psnr_t, mean_psnr_t + std_psnr_t, alpha=0.3, color='green')
    axes[1, 1].set_xlabel('Frame Index')
    axes[1, 1].set_ylabel('PSNR (dB)')
    axes[1, 1].set_title('Per-Frame PSNR Over Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = sample_dir / "metrics_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved analysis plot to {save_path}")
    plt.close()
    
    # Save metrics to file
    metrics_path = sample_dir / "metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"Metrics Summary ({mode})\n")
        f.write("="*50 + "\n")
        f.write(f"Number of samples: {len(all_mse)}\n\n")
        f.write(f"MSE:  {np.mean(all_mse):.6f} ± {np.std(all_mse):.6f}\n")
        f.write(f"PSNR: {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB\n")
        f.write(f"SSIM: {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}\n")
        f.write("\nPer-frame metrics:\n")
        f.write("Frame | MSE | PSNR\n")
        for t in range(len(mean_mse_t)):
            f.write(f"{t:5d} | {mean_mse_t[t]:.6f} | {mean_psnr_t[t]:.2f}\n")
    
    print(f"Saved metrics to {metrics_path}")
    
    return {
        'mse': np.mean(all_mse),
        'psnr': np.mean(all_psnr),
        'ssim': np.mean(all_ssim),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze discrete diffusion results")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory from training")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Analyze generation samples
    gen_dir = output_dir / "samples_generation"
    if gen_dir.exists():
        print("\n" + "="*70)
        print("ANALYZING: Pure Generation Samples")
        print("="*70)
        gen_metrics = analyze_samples(gen_dir, mode="generation")
    
    # Analyze future prediction samples
    future_dir = output_dir / "samples_future_pred"
    if future_dir.exists():
        print("\n" + "="*70)
        print("ANALYZING: Future Prediction Samples")
        print("="*70)
        future_metrics = analyze_samples(future_dir, mode="future_prediction")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
