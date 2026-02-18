#!/usr/bin/env python3
"""Evaluate and compare all CoPilot4D tokenizer checkpoints on the same sequence.

This ensures fair comparison by using the same test data across all checkpoints.

Usage:
    python scripts/evaluate_all_checkpoints.py
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Configuration ──
CHECKPOINT_DIR = Path("outputs/tokenizer_memory_efficient")
CONFIG = Path("configs/tokenizer_memory_efficient.yaml")

# Checkpoints to evaluate
CHECKPOINTS = [
    ("100000", CHECKPOINT_DIR / "checkpoint_step_100000.pt"),
    ("110000", CHECKPOINT_DIR / "checkpoint_step_110000.pt"),
    ("120000", CHECKPOINT_DIR / "checkpoint_step_120000.pt"),
    ("130000", CHECKPOINT_DIR / "checkpoint_step_130000.pt"),
]

# Evaluation settings
SEQUENCE = "00"  # Use sequence 00 for all (has poses)
SAMPLE_IDX = 0
SPLIT = "val"


def run_reconstruction(checkpoint_path, output_dir):
    """Run reconstruction for a checkpoint"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "python", "scripts/reconstruct_lidar.py",
        "--checkpoint", str(checkpoint_path),
        "--config", str(CONFIG),
        "--sample_idx", str(SAMPLE_IDX),
        "--split", SPLIT,
        "--sequence", SEQUENCE,
        "--output_dir", str(output_dir),
        "--dense",
        "--chunk_size", "512",
    ]
    
    print(f"\nRunning reconstruction for {checkpoint_path.name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    return True


def parse_metrics(metrics_file):
    """Parse metrics.txt file and return dict"""
    if not metrics_file.exists():
        return None
    
    metrics = {}
    with open(metrics_file, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line and not line.startswith('─'):
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                try:
                    metrics[key] = float(value)
                except ValueError:
                    metrics[key] = value
    return metrics


def create_comparison_plots(all_metrics, comparison_dir):
    """Create comparison plots across checkpoints"""
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    steps = [m[0] for m in all_metrics]
    
    # Metrics to plot
    metric_configs = [
        ("MAE (m)", "Mean Absolute Error (m)", "MAE", "m"),
        ("RMSE (m)", "Root Mean Square Error (m)", "RMSE", "m"),
        ("Median AE (m)", "Median Absolute Error (m)", "Median_AE", "m"),
        ("MeanRelErr", "Mean Relative Error", "Mean_Rel_Err", ""),
        ("< 1m (%)", "Points within 1m (%)", "Accuracy_1m", "%"),
        ("< 2m (%)", "Points within 2m (%)", "Accuracy_2m", "%"),
        ("< 5m (%)", "Points within 5m (%)", "Accuracy_5m", "%"),
    ]
    
    for key, title, filename, unit in metric_configs:
        values = [m[1].get(key, 0) for m in all_metrics]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(steps))
        bars = ax.bar(x, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(steps)], alpha=0.8)
        
        # Highlight the latest checkpoint
        bars[-1].set_color('#d62728')
        bars[-1].set_alpha(1.0)
        
        ax.set_xticks(x)
        ax.set_xticklabels([f"Step {s}" for s in steps])
        ax.set_ylabel(f"{title} {unit}" if unit else title)
        ax.set_title(f"{title} Comparison")
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}{unit}',
                   ha='center', va='bottom', fontsize=9,
                   fontweight='bold' if i == len(steps)-1 else 'normal')
        
        plt.tight_layout()
        plt.savefig(comparison_dir / f"{filename}_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {comparison_dir / f'{filename}_comparison.png'}")
    
    # Create combined metrics plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # MAE, RMSE, Median AE
    ax = axes[0, 0]
    mae_vals = [m[1].get("MAE (m)", 0) for m in all_metrics]
    rmse_vals = [m[1].get("RMSE (m)", 0) for m in all_metrics]
    median_vals = [m[1].get("Median AE (m)", 0) for m in all_metrics]
    
    x = np.arange(len(steps))
    width = 0.25
    ax.bar(x - width, mae_vals, width, label='MAE', alpha=0.8)
    ax.bar(x, rmse_vals, width, label='RMSE', alpha=0.8)
    ax.bar(x + width, median_vals, width, label='Median AE', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {s}" for s in steps])
    ax.set_ylabel("Error (m)")
    ax.set_title("Depth Error Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Accuracy percentages
    ax = axes[0, 1]
    acc1m = [m[1].get("< 1m (%)", 0) for m in all_metrics]
    acc2m = [m[1].get("< 2m (%)", 0) for m in all_metrics]
    acc5m = [m[1].get("< 5m (%)", 0) for m in all_metrics]
    
    ax.bar(x - width, acc1m, width, label='< 1m', alpha=0.8)
    ax.bar(x, acc2m, width, label='< 2m', alpha=0.8)
    ax.bar(x + width, acc5m, width, label='< 5m', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {s}" for s in steps])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Depth Accuracy (% within threshold)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Mean Relative Error
    ax = axes[1, 0]
    rel_err = [m[1].get("MeanRelErr", 0) for m in all_metrics]
    bars = ax.bar(x, rel_err, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(steps)], alpha=0.8)
    bars[-1].set_color('#d62728')
    bars[-1].set_alpha(1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {s}" for s in steps])
    ax.set_ylabel("Mean Relative Error")
    ax.set_title("Mean Relative Error")
    ax.grid(True, alpha=0.3, axis='y')
    
    # VQ Metrics
    ax = axes[1, 1]
    vq_active = [m[1].get("VQ active", 0) for m in all_metrics]
    ax2 = ax.twinx()
    vq_perpl = [m[1].get("VQ perpl", 0) for m in all_metrics]
    
    bars1 = ax.bar(x - width/2, vq_active, width, label='Active Codes', color='steelblue', alpha=0.8)
    bars2 = ax2.bar(x + width/2, vq_perpl, width, label='Perplexity', color='coral', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {s}" for s in steps])
    ax.set_ylabel("Active Codes", color='steelblue')
    ax2.set_ylabel("Perplexity", color='coral')
    ax.set_title("VQ Metrics")
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.suptitle(f"Tokenizer Checkpoint Comparison (Sequence {SEQUENCE}, Frame {SAMPLE_IDX})", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(comparison_dir / "all_metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {comparison_dir / 'all_metrics_comparison.png'}")


def generate_markdown_report(all_metrics, comparison_dir):
    """Generate comprehensive markdown report"""
    report_path = CHECKPOINT_DIR / f"checkpoint_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_path, 'w') as f:
        f.write("# CoPilot4D Tokenizer Checkpoint Comparison Report\n\n")
        f.write(f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Sequence:** {SEQUENCE}\n")
        f.write(f"**Frame:** {SAMPLE_IDX}\n")
        f.write(f"**Config:** `{CONFIG}`\n\n")
        
        f.write("## Summary Table\n\n")
        f.write("| Checkpoint | MAE (m) | RMSE (m) | Median AE (m) | < 1m (%) | < 2m (%) | < 5m (%) | VQ Active | VQ Perplexity |\n")
        f.write("|------------|---------|----------|---------------|----------|----------|----------|-----------|---------------|\n")
        
        for step, metrics in all_metrics:
            f.write(f"| Step {step} | "
                    f"{metrics.get('MAE (m)', 0):.4f} | "
                    f"{metrics.get('RMSE (m)', 0):.4f} | "
                    f"{metrics.get('Median AE (m)', 0):.4f} | "
                    f"{metrics.get('< 1m (%)', 0):.2f} | "
                    f"{metrics.get('< 2m (%)', 0):.2f} | "
                    f"{metrics.get('< 5m (%)', 0):.2f} | "
                    f"{int(metrics.get('VQ active', 0))} | "
                    f"{metrics.get('VQ perpl', 0):.2f} |\n")
        
        f.write("\n## Detailed Metrics\n\n")
        
        for step, metrics in all_metrics:
            f.write(f"### Step {step}\n\n")
            f.write(f"- **Raw points:** {int(metrics.get('Raw points', 0)):,}\n")
            f.write(f"- **ROI points:** {int(metrics.get('ROI points', 0)):,}\n")
            f.write(f"- **Valid rays:** {int(metrics.get('Valid rays', 0)):,}\n")
            f.write(f"- **MAE:** {metrics.get('MAE (m)', 0):.4f} m\n")
            f.write(f"- **RMSE:** {metrics.get('RMSE (m)', 0):.4f} m\n")
            f.write(f"- **Median AE:** {metrics.get('Median AE (m)', 0):.4f} m\n")
            f.write(f"- **Mean Rel Err:** {metrics.get('MeanRelErr', 0):.4f}\n")
            f.write(f"- **< 1m:** {metrics.get('< 1m (%)', 0):.2f}%\n")
            f.write(f"- **< 2m:** {metrics.get('< 2m (%)', 0):.2f}%\n")
            f.write(f"- **< 5m:** {metrics.get('< 5m (%)', 0):.2f}%\n")
            f.write(f"- **VQ Active:** {int(metrics.get('VQ active', 0))}\n")
            f.write(f"- **VQ Perplexity:** {metrics.get('VQ perpl', 0):.2f}\n\n")
        
        f.write("## Visualizations\n\n")
        f.write(f"All comparison plots saved in: `{comparison_dir}`\n\n")
        f.write("### All Metrics Comparison\n")
        f.write(f"![All Metrics]({comparison_dir.name}/all_metrics_comparison.png)\n\n")
        
        f.write("### Individual Metric Comparisons\n\n")
        metric_files = [
            ("MAE", "Mean Absolute Error"),
            ("RMSE", "Root Mean Square Error"),
            ("Median_AE", "Median Absolute Error"),
            ("Mean_Rel_Err", "Mean Relative Error"),
            ("Accuracy_1m", "Accuracy within 1m"),
            ("Accuracy_2m", "Accuracy within 2m"),
            ("Accuracy_5m", "Accuracy within 5m"),
        ]
        
        for filename, title in metric_files:
            f.write(f"#### {title}\n")
            f.write(f"![{title}]({comparison_dir.name}/{filename}_comparison.png)\n\n")
        
        f.write("\n## Analysis\n\n")
        
        # Find best checkpoint for each metric
        f.write("### Best Performing Checkpoints\n\n")
        
        best_mae = min(all_metrics, key=lambda x: x[1].get('MAE (m)', float('inf')))
        f.write(f"- **Lowest MAE:** Step {best_mae[0]} ({best_mae[1].get('MAE (m)', 0):.4f} m)\n")
        
        best_rmse = min(all_metrics, key=lambda x: x[1].get('RMSE (m)', float('inf')))
        f.write(f"- **Lowest RMSE:** Step {best_rmse[0]} ({best_rmse[1].get('RMSE (m)', 0):.4f} m)\n")
        
        best_acc = max(all_metrics, key=lambda x: x[1].get('< 1m (%)', 0))
        f.write(f"- **Best < 1m Accuracy:** Step {best_acc[0]} ({best_acc[1].get('< 1m (%)', 0):.2f}%)\n")
        
        best_perpl = max(all_metrics, key=lambda x: x[1].get('VQ perpl', 0))
        f.write(f"- **Highest VQ Perplexity:** Step {best_perpl[0]} ({best_perpl[1].get('VQ perpl', 0):.2f})\n")
        
        f.write("\n---\n")
        f.write("*Generated by evaluate_all_checkpoints.py*\n")
    
    return report_path


def main():
    print("=" * 70)
    print("CoPilot4D Tokenizer - Fair Checkpoint Comparison")
    print("=" * 70)
    print(f"\nSequence: {SEQUENCE}")
    print(f"Frame: {SAMPLE_IDX}")
    print(f"Config: {CONFIG}")
    
    all_metrics = []
    
    # Evaluate each checkpoint
    for step, checkpoint_path in CHECKPOINTS:
        if not checkpoint_path.exists():
            print(f"\n⚠️ Checkpoint not found: {checkpoint_path}")
            continue
        
        output_dir = CHECKPOINT_DIR / f"reconstruction_step_{step}_seq{SEQUENCE}"
        
        # Run reconstruction
        if not run_reconstruction(checkpoint_path, output_dir):
            print(f"❌ Failed to evaluate step {step}")
            continue
        
        # Parse metrics
        metrics = parse_metrics(output_dir / "metrics.txt")
        if metrics:
            all_metrics.append((step, metrics))
            print(f"✅ Step {step}: MAE={metrics.get('MAE (m)', 0):.4f}m, <1m={metrics.get('< 1m (%)', 0):.2f}%")
    
    if not all_metrics:
        print("\n❌ No metrics collected. Exiting.")
        sys.exit(1)
    
    # Create comparison directory
    comparison_dir = CHECKPOINT_DIR / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparison plots
    print("\n" + "=" * 70)
    print("Creating Comparison Plots")
    print("=" * 70)
    create_comparison_plots(all_metrics, comparison_dir)
    
    # Generate report
    print("\n" + "=" * 70)
    print("Generating Report")
    print("=" * 70)
    report_path = generate_markdown_report(all_metrics, comparison_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n📊 Evaluated {len(all_metrics)} checkpoints on Sequence {SEQUENCE}, Frame {SAMPLE_IDX}")
    print(f"\n📁 Results saved to:")
    print(f"   • Comparison plots: {comparison_dir}")
    print(f"   • Report: {report_path}")
    
    print("\n📈 Key Metrics Comparison:\n")
    print("-" * 80)
    print(f"{'Checkpoint':<12} {'MAE (m)':<10} {'RMSE (m)':<10} {'< 1m (%)':<10} {'< 2m (%)':<10} {'VQ Perpl':<10}")
    print("-" * 80)
    for step, metrics in all_metrics:
        print(f"Step {step:<6} {metrics.get('MAE (m)', 0):<10.4f} {metrics.get('RMSE (m)', 0):<10.4f} "
              f"{metrics.get('< 1m (%)', 0):<10.2f} {metrics.get('< 2m (%)', 0):<10.2f} "
              f"{metrics.get('VQ perpl', 0):<10.2f}")
    print("-" * 80)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
