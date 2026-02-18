#!/usr/bin/env python3
"""Evaluate CoPilot4D tokenizer checkpoint_step_120000.pt

This script:
1. Runs reconstruction on the checkpoint
2. Compares metrics with earlier checkpoints (110000, 100000)
3. Generates visualizations and a summary report

Usage:
    python scripts/evaluate_checkpoint_120000.py
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

# ── Configuration ──
CHECKPOINT_DIR = Path("outputs/tokenizer_memory_efficient")
CHECKPOINT = CHECKPOINT_DIR / "checkpoint_step_120000.pt"
CONFIG = Path("configs/tokenizer_memory_efficient.yaml")
OUTPUT_DIR = CHECKPOINT_DIR / "reconstruction_step_120000"
COMPARISON_DIR = CHECKPOINT_DIR / "comparison_step120000"

# Earlier checkpoints for comparison
COMPARISON_CHECKPOINTS = [
    ("110000", CHECKPOINT_DIR / "reconstruction_step_110000"),
    ("100000", CHECKPOINT_DIR / "reconstruction_step_100000"),
]


def run_reconstruction():
    """Run reconstruction for checkpoint_step_120000.pt"""
    print("=" * 70)
    print("STEP 1: Running Reconstruction for checkpoint_step_120000.pt")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "python", "scripts/reconstruct_lidar.py",
        "--checkpoint", str(CHECKPOINT),
        "--config", str(CONFIG),
        "--sample_idx", "0",
        "--split", "val",
        "--sequence", "00",  # Use sequence 00 which has poses
        "--output_dir", str(OUTPUT_DIR),
        "--dense",  # Also render dense spherical scan
        "--chunk_size", "512",
    ]
    
    print(f"Running command:\n  {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"Error running reconstruction: {result.stderr}")
        return False
    
    print(f"\n✅ Reconstruction completed. Output saved to: {OUTPUT_DIR}")
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
                    # Try to convert to float if possible
                    metrics[key] = float(value)
                except ValueError:
                    metrics[key] = value
    return metrics


def compare_metrics():
    """Compare metrics across checkpoints"""
    print("\n" + "=" * 70)
    print("STEP 2: Comparing Metrics Across Checkpoints")
    print("=" * 70)
    
    # Load current checkpoint metrics
    current_metrics = parse_metrics(OUTPUT_DIR / "metrics.txt")
    if not current_metrics:
        print(f"❌ Could not find metrics for current checkpoint")
        return
    
    # Load comparison checkpoint metrics
    comparison_metrics = []
    for step, path in COMPARISON_CHECKPOINTS:
        metrics = parse_metrics(path / "metrics.txt")
        if metrics:
            comparison_metrics.append((step, metrics))
    
    # Print comparison table
    print("\n📊 Depth Metrics Comparison:\n")
    print("-" * 80)
    print(f"{'Metric':<25} {'Step 120000':>15} {'Step 110000':>15} {'Step 100000':>15} {'Δ (120k-110k)':>15}")
    print("-" * 80)
    
    metric_keys = [
        "MAE (m)",
        "RMSE (m)",
        "Median AE (m)",
        "MeanRelErr",
        "< 1m (%)",
        "< 2m (%)",
        "< 5m (%)",
    ]
    
    for key in metric_keys:
        current_val = current_metrics.get(key, "N/A")
        step_110k = comparison_metrics[0][1].get(key, "N/A") if len(comparison_metrics) > 0 else "N/A"
        step_100k = comparison_metrics[1][1].get(key, "N/A") if len(comparison_metrics) > 1 else "N/A"
        
        if isinstance(current_val, (int, float)) and isinstance(step_110k, (int, float)):
            delta = current_val - step_110k
            delta_str = f"{delta:+.4f}"
        else:
            delta_str = "N/A"
        
        current_str = f"{current_val:.4f}" if isinstance(current_val, (int, float)) else str(current_val)
        step_110k_str = f"{step_110k:.4f}" if isinstance(step_110k, (int, float)) else str(step_110k)
        step_100k_str = f"{step_100k:.4f}" if isinstance(step_100k, (int, float)) else str(step_100k)
        
        print(f"{key:<25} {current_str:>15} {step_110k_str:>15} {step_100k_str:>15} {delta_str:>15}")
    
    print("-" * 80)
    
    # VQ metrics
    print("\n📊 VQ Metrics:\n")
    print("-" * 60)
    print(f"{'Metric':<25} {'Step 120000':>15} {'Step 110000':>15}")
    print("-" * 60)
    
    vq_keys = ["VQ active", "VQ perpl"]
    for key in vq_keys:
        current_val = current_metrics.get(key, "N/A")
        step_110k = comparison_metrics[0][1].get(key, "N/A") if len(comparison_metrics) > 0 else "N/A"
        
        current_str = f"{current_val:.2f}" if isinstance(current_val, (int, float)) else str(current_val)
        step_110k_str = f"{step_110k:.2f}" if isinstance(step_110k, (int, float)) else str(step_110k)
        
        print(f"{key:<25} {current_str:>15} {step_110k_str:>15}")
    
    print("-" * 60)
    
    return current_metrics, comparison_metrics


def generate_summary_report(current_metrics, comparison_metrics):
    """Generate a markdown summary report"""
    print("\n" + "=" * 70)
    print("STEP 3: Generating Summary Report")
    print("=" * 70)
    
    report_path = CHECKPOINT_DIR / "evaluation_step_120000.md"
    
    with open(report_path, 'w') as f:
        f.write("# CoPilot4D Tokenizer Evaluation Report\n\n")
        f.write(f"**Checkpoint:** `checkpoint_step_120000.pt`\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Config:** `{CONFIG}`\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"- **Raw points:** {int(current_metrics.get('Raw points', 0)):,}\n")
        f.write(f"- **ROI points:** {int(current_metrics.get('ROI points', 0)):,}\n")
        f.write(f"- **Valid rays:** {int(current_metrics.get('Valid rays', 0)):,}\n")
        f.write(f"- **Sequence:** {current_metrics.get('Sequence', 'N/A')}\n")
        f.write(f"- **Frame:** {int(current_metrics.get('frame', 0))}\n\n")
        
        f.write("## Depth Reconstruction Metrics\n\n")
        f.write("| Metric | Step 120000 | Step 110000 | Improvement |\n")
        f.write("|--------|-------------|-------------|-------------|\n")
        
        metric_keys = [
            ("MAE (m)", "m", True),
            ("RMSE (m)", "m", True),
            ("Median AE (m)", "m", True),
            ("MeanRelErr", "", True),
            ("< 1m (%)", "%", False),
            ("< 2m (%)", "%", False),
            ("< 5m (%)", "%", False),
        ]
        
        for key, unit, lower_is_better in metric_keys:
            current_val = current_metrics.get(key, 0)
            step_110k = comparison_metrics[0][1].get(key, 0) if len(comparison_metrics) > 0 else 0
            
            if isinstance(current_val, (int, float)) and isinstance(step_110k, (int, float)):
                delta = current_val - step_110k
                if lower_is_better:
                    improvement = "✅ Better" if delta < 0 else "⚠️ Worse" if delta > 0 else "➡️ Same"
                else:
                    improvement = "✅ Better" if delta > 0 else "⚠️ Worse" if delta < 0 else "➡️ Same"
                delta_str = f"{delta:+.4f}{unit} ({improvement})"
            else:
                delta_str = "N/A"
            
            unit_str = f" {unit}" if unit else ""
            f.write(f"| {key} | {current_val:.4f}{unit_str} | {step_110k:.4f}{unit_str} | {delta_str} |\n")
        
        f.write("\n## VQ Metrics\n\n")
        f.write(f"- **Active codes:** {current_metrics.get('VQ active', 'N/A')}\n")
        f.write(f"- **Perplexity:** {current_metrics.get('VQ perpl', 'N/A'):.2f}\n\n")
        
        f.write("## Generated Files\n\n")
        f.write(f"All reconstruction outputs are saved in: `{OUTPUT_DIR}`\n\n")
        f.write("### Files:\n")
        for file in sorted(OUTPUT_DIR.glob("*")):
            f.write(f"- `{file.name}`\n")
        
        f.write("\n## Visualizations\n\n")
        f.write("### BEV Comparison\n")
        f.write("![BEV Comparison](reconstruction_step_120000/bev_comparison.png)\n\n")
        f.write("### Side View Comparison\n")
        f.write("![Side Comparison](reconstruction_step_120000/side_comparison.png)\n\n")
        f.write("### BEV Overlay\n")
        f.write("![BEV Overlay](reconstruction_step_120000/bev_overlay.png)\n\n")
        f.write("### Depth Scatter Plot\n")
        f.write("![Depth Scatter](reconstruction_step_120000/depth_scatter.png)\n\n")
        f.write("### Error Histogram\n")
        f.write("![Error Histogram](reconstruction_step_120000/error_histogram.png)\n\n")
        f.write("### Dense Reconstruction\n")
        f.write("![Dense BEV](reconstruction_step_120000/dense_bev.png)\n\n")
        
        f.write("\n## Comparison with Earlier Checkpoints\n\n")
        f.write("| Checkpoint | MAE (m) | RMSE (m) | < 1m (%) | VQ Perplexity |\n")
        f.write("|------------|---------|----------|----------|---------------|\n")
        
        # Current
        f.write(f"| Step 120000 | {current_metrics.get('MAE (m)', 0):.4f} | "
                f"{current_metrics.get('RMSE (m)', 0):.4f} | "
                f"{current_metrics.get('< 1m (%)', 0):.2f} | "
                f"{current_metrics.get('VQ perpl', 0):.2f} |\n")
        
        # Comparisons
        for step, metrics in comparison_metrics:
            f.write(f"| Step {step} | {metrics.get('MAE (m)', 0):.4f} | "
                    f"{metrics.get('RMSE (m)', 0):.4f} | "
                    f"{metrics.get('< 1m (%)', 0):.2f} | "
                    f"{metrics.get('VQ perpl', 0):.2f} |\n")
        
        f.write("\n---\n")
        f.write(f"*Generated by evaluate_checkpoint_120000.py*\n")
    
    print(f"✅ Summary report saved to: {report_path}")
    return report_path


def print_final_summary(current_metrics):
    """Print final summary to console"""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY - checkpoint_step_120000.pt")
    print("=" * 70)
    
    print(f"\n📁 Output Directory: {OUTPUT_DIR}")
    print(f"📊 Report: {CHECKPOINT_DIR / 'evaluation_step_120000.md'}")
    
    print("\n📈 Key Metrics:")
    print(f"  • MAE:           {current_metrics.get('MAE (m)', 0):.4f} m")
    print(f"  • RMSE:          {current_metrics.get('RMSE (m)', 0):.4f} m")
    print(f"  • Median AE:     {current_metrics.get('Median AE (m)', 0):.4f} m")
    print(f"  • Mean Rel Err:  {current_metrics.get('MeanRelErr', 0):.4f}")
    print(f"  • < 1m accuracy: {current_metrics.get('< 1m (%)', 0):.2f}%")
    print(f"  • < 2m accuracy: {current_metrics.get('< 2m (%)', 0):.2f}%")
    print(f"  • < 5m accuracy: {current_metrics.get('< 5m (%)', 0):.2f}%")
    
    print("\n🔢 VQ Metrics:")
    print(f"  • Active codes:  {current_metrics.get('VQ active', 'N/A')}")
    print(f"  • Perplexity:    {current_metrics.get('VQ perpl', 0):.2f}")
    
    print("\n📂 Generated Files:")
    for file in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  • {file.name}")
    
    print("\n" + "=" * 70)
    print("✅ Evaluation Complete!")
    print("=" * 70)


def main():
    print("\n" + "=" * 70)
    print("CoPilot4D Tokenizer Evaluation - Step 120000")
    print("=" * 70)
    print(f"\nCheckpoint: {CHECKPOINT}")
    print(f"Config: {CONFIG}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Step 1: Run reconstruction
    if not run_reconstruction():
        print("\n❌ Reconstruction failed. Exiting.")
        sys.exit(1)
    
    # Step 2: Compare metrics
    result = compare_metrics()
    if not result:
        print("\n❌ Could not compare metrics.")
        sys.exit(1)
    
    current_metrics, comparison_metrics = result
    
    # Step 3: Generate summary report
    report_path = generate_summary_report(current_metrics, comparison_metrics)
    
    # Print final summary
    print_final_summary(current_metrics)
    
    print(f"\n📝 View the full report at: {report_path}")


if __name__ == "__main__":
    main()
