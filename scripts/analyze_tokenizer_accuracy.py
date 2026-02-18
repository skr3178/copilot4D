#!/usr/bin/env python3
"""Analyze tokenizer accuracy in specific scenarios:
- Close pedestrians/objects (near field)
- Turns/curves
- Buildings/walls
- Far field vs near field

Usage:
    python scripts/analyze_tokenizer_accuracy.py \
        --checkpoint outputs/tokenizer_memory_efficient/checkpoint_step_100000.pt \
        --config configs/tokenizer_memory_efficient.yaml \
        --sequence 00 \
        --frames 1000,1020,1040,1060,1080 \
        --output_dir outputs/accuracy_analysis
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).parent.parent))

from copilot4d.tokenizer.tokenizer_model import CoPilot4DTokenizer
from copilot4d.utils.config import TokenizerConfig
from copilot4d.data.point_cloud_utils import filter_roi, generate_rays
from copilot4d.data.kitti_dataset import KITTITokenizerDataset, tokenizer_collate_fn
from pykitti import odometry


def load_config(config_path: str) -> TokenizerConfig:
    import yaml
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return TokenizerConfig(**config_dict)


def load_model(checkpoint_path: str, cfg: TokenizerConfig, device: str = "cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CoPilot4DTokenizer(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_original_lidar(kitti_root: str, sequence: str, frame_idx: int, cfg: TokenizerConfig):
    try:
        dataset = odometry(kitti_root, sequence)
        points = dataset.get_velo(frame_idx)
        points_filtered = filter_roi(points, cfg)
        return points_filtered[:, :3]
    except Exception as e:
        print(f"Warning: Could not load LiDAR frame {frame_idx}: {e}")
        return None


@torch.no_grad()
def reconstruct_frame(model, cfg, device, sequence: str, frame_idx: int):
    dataset = KITTITokenizerDataset(cfg, sequences=[sequence])
    
    sample_idx = None
    for idx, (seq, frm) in enumerate(dataset.samples):
        if seq == sequence and frm == frame_idx:
            sample_idx = idx
            break
    
    if sample_idx is None:
        return None
    
    sample = dataset[sample_idx]
    batch = tokenizer_collate_fn([sample])
    
    features = batch["features"].to(device)
    num_points = batch["num_points"].to(device)
    coords = batch["coords"].to(device)
    batch_size = batch["batch_size"]
    
    bev = model.encode_voxels(features, num_points, coords, batch_size)
    encoder_out = model.encoder(bev)
    quantized, indices, _, vq_metrics = model.vq(encoder_out)
    
    decoder_output = model.decoder(quantized)
    nfg = model.nfg.build_nfg(decoder_output)
    
    raw_points = dataset.datasets[sequence].get_velo(frame_idx)
    points_roi = filter_roi(raw_points, cfg)
    
    if len(points_roi) == 0:
        return np.zeros((0, 3))
    
    ray_data = generate_rays(points_roi, cfg)
    ray_origins = ray_data["ray_origins"]
    ray_directions = ray_data["ray_directions"]
    gt_depths = ray_data["ray_depths"]
    
    valid = (gt_depths >= cfg.ray_depth_min) & (gt_depths <= cfg.ray_depth_max)
    ray_origins = ray_origins[valid]
    ray_directions = ray_directions[valid]
    
    if len(ray_directions) == 0:
        return np.zeros((0, 3))
    
    chunk_size = 1024
    all_depths = []
    
    origins_t = torch.from_numpy(ray_origins).unsqueeze(0).to(device)
    dirs_t = torch.from_numpy(ray_directions).unsqueeze(0).to(device)
    
    for start in range(0, len(ray_directions), chunk_size):
        end = min(start + chunk_size, len(ray_directions))
        ro = origins_t[:, start:end]
        rd = dirs_t[:, start:end]
        
        depths, _ = model.nfg.query_rays(nfg, ro, rd, cfg.ray_depth_min, cfg.ray_depth_max)
        all_depths.append(depths.squeeze(0).cpu())
    
    pred_depths = torch.cat(all_depths, dim=0)
    pred_points = ray_origins + ray_directions * pred_depths.numpy()[:, None]
    
    return pred_points


def compute_error_metrics(gt_points, pred_points):
    """Compute per-point depth errors."""
    # Compute depths from origin
    gt_depths = np.linalg.norm(gt_points, axis=1)
    pred_depths = np.linalg.norm(pred_points, axis=1)
    
    errors = np.abs(pred_depths - gt_depths)
    rel_errors = errors / np.maximum(gt_depths, 1.0)
    
    return gt_depths, errors, rel_errors


def analyze_scenario(gt_points, pred_points, scenario_type, cfg):
    """Analyze accuracy in specific scenarios."""
    if len(gt_points) == 0 or len(pred_points) == 0:
        return None
    
    gt_depths, errors, rel_errors = compute_error_metrics(gt_points, pred_points)
    
    # Define scenario regions
    if scenario_type == "near_field":
        # Within 10m of vehicle
        mask = gt_depths < 10.0
        label = "Near Field (<10m)"
    elif scenario_type == "mid_field":
        # 10-25m
        mask = (gt_depths >= 10.0) & (gt_depths < 25.0)
        label = "Mid Field (10-25m)"
    elif scenario_type == "far_field":
        # Beyond 25m
        mask = gt_depths >= 25.0
        label = "Far Field (>25m)"
    elif scenario_type == "pedestrian_zone":
        # Close to vehicle, front-left/right areas
        front_mask = gt_points[:, 0] > 0  # In front of vehicle
        side_mask = np.abs(gt_points[:, 1]) < 10  # Within 10m left/right
        close_mask = gt_depths < 15  # Within 15m
        mask = front_mask & side_mask & close_mask
        label = "Pedestrian Zone (Front, <15m)"
    elif scenario_type == "turn_zone":
        # Side areas for turns
        side_mask = np.abs(gt_points[:, 1]) > 5  # More than 5m to side
        mask = side_mask
        label = "Turn Zone (Sides)"
    else:
        mask = np.ones(len(gt_points), dtype=bool)
        label = "All Points"
    
    if mask.sum() == 0:
        return None
    
    region_errors = errors[mask]
    region_depths = gt_depths[mask]
    region_rel_errors = rel_errors[mask]
    
    return {
        "label": label,
        "count": len(region_errors),
        "mean_depth": np.mean(region_depths),
        "mae": np.mean(region_errors),
        "rmse": np.sqrt(np.mean(region_errors**2)),
        "median_error": np.median(region_errors),
        "mean_rel_error": np.mean(region_rel_errors),
        "accuracy_1m": (region_errors < 1.0).mean() * 100,
        "accuracy_2m": (region_errors < 2.0).mean() * 100,
    }


def create_detailed_analysis(gt_points, pred_points, cfg, frame_idx, checkpoint_step, output_dir):
    """Create detailed accuracy analysis visualization."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main BEV comparison
    ax_main = fig.add_subplot(gs[0:2, 0:2])
    
    # Plot GT in blue with lower alpha
    ax_main.scatter(gt_points[:, 0], gt_points[:, 1], c='dodgerblue', s=0.3, alpha=0.4, label='Real LiDAR')
    
    # Plot prediction in red
    ax_main.scatter(pred_points[:, 0], pred_points[:, 1], c='red', s=0.3, alpha=0.4, label='Tokenizer')
    
    # Draw region of interest boxes
    # Near field circle
    near_circle = Circle((0, 0), 10, fill=False, edgecolor='green', linewidth=2, linestyle='--', label='Near field (<10m)')
    ax_main.add_patch(near_circle)
    
    # Mid field
    mid_circle = Circle((0, 0), 25, fill=False, edgecolor='orange', linewidth=2, linestyle='--', label='Mid field (10-25m)')
    ax_main.add_patch(mid_circle)
    
    ax_main.set_xlim(cfg.x_min, cfg.x_max)
    ax_main.set_ylim(cfg.y_min, cfg.y_max)
    ax_main.set_aspect('equal')
    ax_main.set_xlabel('X (m)')
    ax_main.set_ylabel('Y (m)')
    ax_main.set_title(f'Frame {frame_idx} - Checkpoint Step {checkpoint_step}')
    ax_main.legend(loc='upper right', markerscale=10)
    ax_main.grid(True, alpha=0.3)
    
    # Error by distance plot
    ax_error = fig.add_subplot(gs[0, 2])
    gt_depths, errors, rel_errors = compute_error_metrics(gt_points, pred_points)
    
    # Scatter plot: depth vs error
    ax_error.scatter(gt_depths, errors, s=0.5, alpha=0.3, c='steelblue')
    ax_error.axhline(y=1.0, color='green', linestyle='--', label='1m threshold')
    ax_error.axhline(y=2.0, color='orange', linestyle='--', label='2m threshold')
    ax_error.set_xlabel('GT Depth (m)')
    ax_error.set_ylabel('Absolute Error (m)')
    ax_error.set_title('Error vs Depth')
    ax_error.legend()
    ax_error.grid(True, alpha=0.3)
    
    # Error histogram
    ax_hist = fig.add_subplot(gs[1, 2])
    ax_hist.hist(errors, bins=50, range=(0, 10), color='steelblue', edgecolor='black', alpha=0.7)
    ax_hist.axvline(x=np.median(errors), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.2f}m')
    ax_hist.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}m')
    ax_hist.set_xlabel('Absolute Error (m)')
    ax_hist.set_ylabel('Count')
    ax_hist.set_title('Error Distribution')
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)
    
    # Scenario analysis table
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')
    
    scenarios = ["near_field", "mid_field", "far_field", "pedestrian_zone", "turn_zone", "all"]
    scenario_results = []
    
    for scenario in scenarios:
        result = analyze_scenario(gt_points, pred_points, scenario, cfg)
        if result:
            scenario_results.append(result)
    
    # Create table
    table_data = []
    headers = ["Region", "Points", "Mean Depth", "MAE (m)", "RMSE (m)", "Med Error", "<1m (%)", "<2m (%)"]
    table_data.append(headers)
    
    for r in scenario_results:
        row = [
            r["label"],
            f"{r['count']:,}",
            f"{r['mean_depth']:.1f}m",
            f"{r['mae']:.2f}",
            f"{r['rmse']:.2f}",
            f"{r['median_error']:.2f}",
            f"{r['accuracy_1m']:.1f}%",
            f"{r['accuracy_2m']:.1f}%",
        ]
        table_data.append(row)
    
    table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                          bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color best performing rows
    for i in range(1, len(table_data)):
        accuracy_1m = float(table_data[i][6].replace('%', ''))
        if accuracy_1m > 70:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#C6EFCE')  # Green
        elif accuracy_1m > 40:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#FFEB9C')  # Yellow
        else:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#FFC7CE')  # Red
    
    ax_table.set_title('Accuracy by Region', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle(f'Tokenizer Accuracy Analysis - Step {checkpoint_step} - Frame {frame_idx}', 
                 fontsize=16, fontweight='bold')
    
    output_path = output_dir / f"accuracy_analysis_step{checkpoint_step}_frame{frame_idx:04d}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    return scenario_results


def create_comparison_across_frames(results_100k, results_110k, frame_idx, output_dir):
    """Create comparison between checkpoints."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Prepare data for plotting
    regions = [r["label"] for r in results_100k]
    mae_100k = [r["mae"] for r in results_100k]
    mae_110k = [r["mae"] for r in results_110k]
    acc1m_100k = [r["accuracy_1m"] for r in results_100k]
    acc1m_110k = [r["accuracy_1m"] for r in results_110k]
    
    x = np.arange(len(regions))
    width = 0.35
    
    # MAE comparison
    ax = axes[0, 0]
    bars1 = ax.bar(x - width/2, mae_100k, width, label='Step 100000', color='steelblue')
    bars2 = ax.bar(x + width/2, mae_110k, width, label='Step 110000', color='coral')
    ax.set_ylabel('MAE (m)')
    ax.set_title('Mean Absolute Error by Region')
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # <1m accuracy comparison
    ax = axes[0, 1]
    bars1 = ax.bar(x - width/2, acc1m_100k, width, label='Step 100000', color='steelblue')
    bars2 = ax.bar(x + width/2, acc1m_110k, width, label='Step 110000', color='coral')
    ax.set_ylabel('< 1m Accuracy (%)')
    ax.set_title('Points within 1m by Region')
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='50% threshold')
    
    # Detailed table
    ax = axes[1, 0]
    ax.axis('off')
    
    table_data = [["Region", "Step 100k MAE", "Step 110k MAE", "Degradation"]]
    for i, region in enumerate(regions):
        degradation = mae_110k[i] - mae_100k[i]
        table_data.append([
            region,
            f"{mae_100k[i]:.2f}m",
            f"{mae_110k[i]:.2f}m",
            f"+{degradation:.2f}m" if degradation > 0 else f"{degradation:.2f}m"
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('MAE Comparison', fontsize=12, fontweight='bold', pad=10)
    
    # Summary text
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
SUMMARY - Frame {frame_idx}

BEST PERFORMING REGIONS (Step 100k):
"""
    # Sort by accuracy
    sorted_results = sorted(results_100k, key=lambda x: x["accuracy_1m"], reverse=True)
    for i, r in enumerate(sorted_results[:3], 1):
        summary_text += f"{i}. {r['label']}: {r['accuracy_1m']:.1f}% <1m\n"
    
    summary_text += f"""
WORST PERFORMING REGIONS (Step 100k):
"""
    for i, r in enumerate(sorted_results[-3:], 1):
        summary_text += f"{i}. {r['label']}: {r['accuracy_1m']:.1f}% <1m\n"
    
    summary_text += f"""
DEGRADATION (Step 110k vs 100k):
"""
    for i, region in enumerate(regions):
        degradation = acc1m_110k[i] - acc1m_100k[i]
        summary_text += f"• {region}: {degradation:+.1f}%\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Checkpoint Comparison - Frame {frame_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f"comparison_frame{frame_idx:04d}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint100", type=str, required=True, help="Step 100000 checkpoint")
    parser.add_argument("--checkpoint110", type=str, required=True, help="Step 110000 checkpoint")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sequence", type=str, default="00")
    parser.add_argument("--frames", type=str, default="1000,1020,1040",
                       help="Comma-separated frame indices to analyze")
    parser.add_argument("--output_dir", type=str, default="outputs/accuracy_analysis")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frames = [int(f.strip()) for f in args.frames.split(",")]
    
    # Load config
    cfg = load_config(args.config)
    kitti_root = getattr(cfg, 'kitti_root', 'data/kitti/dataset')
    if 'pykitti' in kitti_root:
        kitti_root = 'data/kitti/dataset'
    
    print("="*70)
    print("TOKENIZER ACCURACY ANALYSIS")
    print("="*70)
    print(f"\nConfig: {args.config}")
    print(f"Sequence: {args.sequence}")
    print(f"Frames: {frames}")
    
    # Load both models
    print("\nLoading models...")
    model_100k = load_model(args.checkpoint100, cfg, device)
    model_110k = load_model(args.checkpoint110, cfg, device)
    
    for frame_idx in frames:
        print(f"\n{'='*70}")
        print(f"Analyzing Frame {frame_idx}")
        print("="*70)
        
        # Load GT
        gt_points = load_original_lidar(kitti_root, args.sequence, frame_idx, cfg)
        if gt_points is None:
            continue
        
        # Reconstruct with both models
        print("  Reconstructing with step 100000...")
        pred_100k = reconstruct_frame(model_100k, cfg, device, args.sequence, frame_idx)
        
        print("  Reconstructing with step 110000...")
        pred_110k = reconstruct_frame(model_110k, cfg, device, args.sequence, frame_idx)
        
        # Analyze
        results_100k = create_detailed_analysis(gt_points, pred_100k, cfg, frame_idx, 100000, output_dir)
        results_110k = create_detailed_analysis(gt_points, pred_110k, cfg, frame_idx, 110000, output_dir)
        
        # Compare
        create_comparison_across_frames(results_100k, results_110k, frame_idx, output_dir)
    
    print(f"\n{'='*70}")
    print(f"Analysis complete! Output: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
