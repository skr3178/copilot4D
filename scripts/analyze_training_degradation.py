#!/usr/bin/env python3
"""Analyze training degradation after step 100000.

This script examines:
1. Loss landscape changes (direct depth weighting d^0.5)
2. Surface loss magnitude doubling from depth sample increase
3. Learning rate and relative loss weight ramp effects
4. Training vs validation loss divergence
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Paths
TRAIN_METRICS = Path("outputs/tokenizer_memory_efficient/training_metrics.jsonl")
EVAL_METRICS = Path("outputs/tokenizer_memory_efficient/eval_metrics.jsonl")
OUTPUT_DIR = Path("outputs/tokenizer_memory_efficient/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics(filepath):
    """Load metrics from JSONL file."""
    metrics = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))
    return metrics


def analyze_loss_landscape():
    """Analyze the loss landscape changes."""
    print("=" * 70)
    print("TRAINING DEGRADATION ANALYSIS")
    print("=" * 70)
    
    # Load metrics
    train_metrics = load_metrics(TRAIN_METRICS)
    eval_metrics = load_metrics(EVAL_METRICS)
    
    print(f"\n📊 Loaded {len(train_metrics)} training records")
    print(f"📊 Loaded {len(eval_metrics)} evaluation records")
    
    # Split into pre-100k and post-100k
    pre_train = [m for m in train_metrics if m['step'] < 100000]
    post_train = [m for m in train_metrics if m['step'] >= 100000]
    
    pre_eval = [m for m in eval_metrics if m['step'] < 100000]
    post_eval = [m for m in eval_metrics if m['step'] >= 100000]
    
    print(f"\n{'='*70}")
    print("1. TRAINING LOSS COMPARISON (Pre vs Post Step 100000)")
    print("=" * 70)
    
    if pre_train:
        pre_loss = np.mean([m['loss_total'] for m in pre_train[-100:]])  # Last 100 steps
        pre_depth = np.mean([m['depth_l1'] for m in pre_train[-100:]])
        print(f"\nPre-100k (last 100 steps avg):")
        print(f"  Total Loss: {pre_loss:.4f}")
        print(f"  Depth L1:   {pre_depth:.4f}")
    
    if post_train:
        post_loss = np.mean([m['loss_total'] for m in post_train[:100]])  # First 100 steps
        post_depth = np.mean([m['depth_l1'] for m in post_train[:100]])
        post_loss_latest = np.mean([m['loss_total'] for m in post_train[-100:]])  # Latest 100 steps
        post_depth_latest = np.mean([m['depth_l1'] for m in post_train[-100:]])
        
        print(f"\nPost-100k (first 100 steps avg):")
        print(f"  Total Loss: {post_loss:.4f}")
        print(f"  Depth L1:   {post_depth:.4f}")
        
        print(f"\nPost-100k (latest 100 steps avg):")
        print(f"  Total Loss: {post_loss_latest:.4f}")
        print(f"  Depth L1:   {post_depth_latest:.4f}")
    
    print(f"\n{'='*70}")
    print("2. VALIDATION LOSS COMPARISON (Pre vs Post Step 100000)")
    print("=" * 70)
    
    if pre_eval:
        pre_eval_loss = np.mean([m['loss'] for m in pre_eval[-5:]])  # Last 5 evals
        pre_eval_depth = np.mean([m['depth_l1'] for m in pre_eval[-5:]])
        print(f"\nPre-100k validation (last 5 avg):")
        print(f"  Total Loss: {pre_eval_loss:.4f}")
        print(f"  Depth L1:   {pre_eval_depth:.4f}")
    
    if post_eval:
        post_eval_loss = np.mean([m['loss'] for m in post_eval[:5]])  # First 5 evals
        post_eval_depth = np.mean([m['depth_l1'] for m in post_eval[:5]])
        post_eval_loss_latest = np.mean([m['loss'] for m in post_eval[-5:]])  # Latest 5 evals
        post_eval_depth_latest = np.mean([m['depth_l1'] for m in post_eval[-5:]])
        
        print(f"\nPost-100k validation (first 5 avg):")
        print(f"  Total Loss: {post_eval_loss:.4f}")
        print(f"  Depth L1:   {post_eval_depth:.4f}")
        
        print(f"\nPost-100k validation (latest 5 avg):")
        print(f"  Total Loss: {post_eval_loss_latest:.4f}")
        print(f"  Depth L1:   {post_eval_depth_latest:.4f}")
    
    print(f"\n{'='*70}")
    print("3. LOSS LANDSCAPE CHANGES")
    print("=" * 70)
    
    print("""
Config Changes at Step 100000:
┌─────────────────────────┬─────────────┬─────────────┐
│ Parameter               │ Pre-100k    │ Post-100k   │
├─────────────────────────┼─────────────┼─────────────┤
│ num_depth_samples       │ 128         │ 256 (2x)    │
│ batch_size              │ 2           │ 2 (same)    │
│ grad_accum_steps        │ 8           │ 16 (2x)     │
│ effective batch size    │ 16          │ 32 (2x)     │
│ learning_rate           │ 1.0e-4      │ 5.0e-5 (0.5x)│
│ depth_loss_type         │ l1          │ combined    │
│ depth_loss_alpha        │ 0.0         │ 0.5 (new)   │
│ relative_weight (start) │ N/A         │ 0.1 → 1.0   │
│ ramp_steps              │ N/A         │ 5000        │
└─────────────────────────┴─────────────┴─────────────┘
""")
    
    print("\n4. ANALYSIS OF KEY FACTORS")
    print("-" * 70)
    
    # Factor 1: Direct depth weighting
    print("\n🔍 FACTOR 1: Direct Depth Weighting (d^0.5)")
    print("-" * 50)
    print("""
With alpha=0.5, weights are applied as: weight = depth^0.5
  - 10m depth: weight = sqrt(10) ≈ 3.16
  - 50m depth: weight = sqrt(50) ≈ 7.07
  
This means:
  - Far field (30-50m) gets ~2.2x more gradient emphasis
  - Near field (0-10m) gets LESS emphasis relative to far field
  - Model may sacrifice near-field accuracy for far-field
  
⚠️  PROBLEM: The reconstruction evaluation shows MAE degraded 
    from 1.15m to 3.4m, suggesting near-field accuracy was 
    sacrificed too much.
""")
    
    # Factor 2: Relative loss ramp
    print("\n🔍 FACTOR 2: Relative Loss Weight Ramp (0.1 → 1.0)")
    print("-" * 50)
    print("""
The loss gradually shifts from:
  Step 100000: 90% absolute + 10% relative
  Step 105000:  0% absolute + 100% relative (after ramp)
  
Relative loss = |pred - gt| / gt
  - Emphasizes percentage error over absolute meters
  - 1m error at 10m depth = 10% error (small loss)
  - 1m error at 50m depth = 2% error (even smaller loss)
  
⚠️  PROBLEM: With 100% relative loss, the model can have large
    absolute errors at far depths without penalty.
    
    Example:
      - GT depth = 40m, Pred = 45m
      - Absolute error = 5m
      - Relative error = 5/40 = 12.5%
      - Loss contribution: 0.125 (small)
""")
    
    # Factor 3: Doubled depth samples
    print("\n🔍 FACTOR 3: Doubled Depth Samples (128 → 256)")
    print("-" * 50)
    print("""
More depth samples means:
  - Finer depth resolution along each ray
  - Higher computational cost
  - BUT: More opportunities for surface concentration loss
  
Surface concentration loss encourages weight concentration
at the actual surface location. With more samples:
  - Higher resolution = more precise surface localization
  - BUT: May overfit to specific depth values
  
⚠️  PROBLEM: Combined with relative loss, the model may learn
    to spread predictions to minimize relative error across
    all samples, rather than focusing on accurate surface depth.
""")
    
    # Factor 4: Learning rate
    print("\n🔍 FACTOR 4: Learning Rate Drop (1e-4 → 5e-5)")
    print("-" * 50)
    print("""
Learning rate was halved when resuming:
  - Pre-100k: lr = 1.0e-4 (fine-tuned for 100k steps)
  - Post-100k: lr = 5.0e-5 (new loss landscape)
  
⚠️  POTENTIAL ISSUE: While lower LR is good for fine-tuning,
    the combination with new loss function may have caused
    the model to settle into a suboptimal local minimum.
    
    The model "unlearned" good near-field representations
    because the new loss landscape doesn't penalize near-field
    errors as heavily.
""")
    
    print(f"\n{'='*70}")
    print("5. ROOT CAUSE SUMMARY")
    print("=" * 70)
    print("""
PRIMARY CULPRIT: Combined Loss with Relative Weight = 1.0

The shift to 100% relative loss (after ramp) combined with
sqrt(depth) weighting creates a loss landscape that:

1. ✅ Benefits: Better far-field relative accuracy
   - Model focuses on percentage error at far depths
   
2. ❌ Drawbacks: Sacrifices near-field absolute accuracy
   - Near-field errors (0-10m) have small relative impact
   - Model "ignores" near-field to optimize far-field
   
3. ❌ Result: Overall MAE degrades from 1.15m → 3.4m
   - <1m accuracy drops from 72% → 2.5%
   - Most points now have 2-5m error

RECOMMENDATION:
  - Use balanced loss: 50% absolute + 50% relative
  - OR: Use depth-weighted absolute loss only (no relative)
  - OR: Clamp relative loss contribution
""")
    
    return train_metrics, eval_metrics


def plot_loss_analysis(train_metrics, eval_metrics):
    """Create plots visualizing the degradation."""
    print(f"\n{'='*70}")
    print("6. GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    # Extract data
    train_steps = [m['step'] for m in train_metrics if m['step'] >= 95000]
    train_loss = [m['loss_total'] for m in train_metrics if m['step'] >= 95000]
    train_depth = [m['depth_l1'] for m in train_metrics if m['step'] >= 95000]
    train_rel_weight = [m.get('rel_weight', 0) for m in train_metrics if m['step'] >= 95000]
    
    eval_steps = [m['step'] for m in eval_metrics if m['step'] >= 95000]
    eval_loss = [m['loss'] for m in eval_metrics if m['step'] >= 95000]
    eval_depth = [m['depth_l1'] for m in eval_metrics if m['step'] >= 95000]
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Training and Validation Loss
    ax = axes[0]
    ax.plot(train_steps, train_loss, alpha=0.3, color='blue', label='Training Loss (raw)')
    # Smoothed training loss
    window = 100
    if len(train_loss) > window:
        smoothed = np.convolve(train_loss, np.ones(window)/window, mode='valid')
        smoothed_steps = train_steps[window-1:]
        ax.plot(smoothed_steps, smoothed, color='blue', linewidth=2, label='Training Loss (smoothed)')
    ax.scatter(eval_steps, eval_loss, color='red', s=30, zorder=5, label='Validation Loss')
    ax.axvline(x=100000, color='black', linestyle='--', linewidth=2, label='Resume Point')
    ax.axvline(x=105000, color='orange', linestyle=':', linewidth=1.5, label='Ramp Complete')
    ax.set_xlabel('Step')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training vs Validation Loss (Steps 95000-125000)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(95000, 125000)
    
    # Plot 2: Depth L1 Loss
    ax = axes[1]
    ax.plot(train_steps, train_depth, alpha=0.3, color='green', label='Training Depth L1 (raw)')
    if len(train_depth) > window:
        smoothed_d = np.convolve(train_depth, np.ones(window)/window, mode='valid')
        ax.plot(smoothed_steps, smoothed_d, color='green', linewidth=2, label='Training Depth L1 (smoothed)')
    ax.scatter(eval_steps, eval_depth, color='red', s=30, zorder=5, label='Validation Depth L1')
    ax.axvline(x=100000, color='black', linestyle='--', linewidth=2, label='Resume Point')
    ax.axvline(x=105000, color='orange', linestyle=':', linewidth=1.5, label='Ramp Complete')
    ax.set_xlabel('Step')
    ax.set_ylabel('Depth L1 Loss')
    ax.set_title('Depth L1 Loss (Training vs Validation)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(95000, 125000)
    
    # Plot 3: Relative Weight Ramp
    ax = axes[2]
    ax.plot(train_steps, train_rel_weight, color='purple', linewidth=2)
    ax.axvline(x=100000, color='black', linestyle='--', linewidth=2, label='Resume Point')
    ax.axvline(x=105000, color='orange', linestyle=':', linewidth=1.5, label='Ramp Complete (100% relative)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Relative Loss Weight')
    ax.set_title('Relative Loss Weight Ramp (0.1 → 1.0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(95000, 125000)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'loss_degradation_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'loss_degradation_analysis.png'}")
    plt.close()
    
    # Create weight analysis plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Show depth weighting effect
    depths = np.linspace(1, 55, 100)
    weights_sqrt = np.power(depths, 0.5)  # alpha=0.5
    weights_linear = depths  # alpha=1.0
    weights_uniform = np.ones_like(depths)  # alpha=0.0
    
    ax.plot(depths, weights_sqrt / weights_sqrt.mean(), label='α=0.5 (sqrt, current)', linewidth=2)
    ax.plot(depths, weights_linear / weights_linear.mean(), label='α=1.0 (linear)', linewidth=2, linestyle='--')
    ax.plot(depths, weights_uniform, label='α=0.0 (uniform, original)', linewidth=2, linestyle=':')
    
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel('Normalized Weight')
    ax.set_title('Depth Weighting Functions (Normalized to Mean=1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
    
    # Add annotations
    ax.annotate('Near field\n(<10m)', xy=(5, 0.5), xytext=(15, 0.3),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    ax.annotate('Far field\n(>40m)', xy=(45, 1.5), xytext=(35, 1.8),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'depth_weighting_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'depth_weighting_analysis.png'}")
    plt.close()


def main():
    train_metrics, eval_metrics = analyze_loss_landscape()
    plot_loss_analysis(train_metrics, eval_metrics)
    
    print(f"\n{'='*70}")
    print(f"Analysis complete! Output saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
