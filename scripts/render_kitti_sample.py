#!/usr/bin/env python3
"""Render a sample from KITTI dataset using the CoPilot4D tokenizer.

This script demonstrates the full pipeline:
1. Load KITTI point cloud
2. Voxelize and prepare rays
3. Run through tokenizer (encode -> VQ -> decode -> render)
4. Visualize results

Note: Without a trained model, outputs will be random. This demonstrates the pipeline.
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from copilot4d.utils.config import TokenizerConfig
from copilot4d.tokenizer.tokenizer_model import CoPilot4DTokenizer
from copilot4d.data.point_cloud_utils import (
    filter_roi,
    voxelize_points,
    generate_rays,
    sample_training_rays,
)


def load_kitti_bin(bin_path: str) -> np.ndarray:
    """Load KITTI point cloud from .bin file.
    
    Returns:
        points: (N, 4) array [x, y, z, reflectance]
    """
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points


def compute_gt_occupancy(points: np.ndarray, cfg: TokenizerConfig) -> np.ndarray:
    """Compute binary occupancy grid from point cloud."""
    H, W, Z = cfg.voxel_grid_xy, cfg.voxel_grid_xy, cfg.voxel_grid_z
    
    ix = ((points[:, 0] - cfg.x_min) / (cfg.x_max - cfg.x_min) * H).astype(np.int32)
    iy = ((points[:, 1] - cfg.y_min) / (cfg.y_max - cfg.y_min) * W).astype(np.int32)
    iz = ((points[:, 2] - cfg.z_min) / (cfg.z_max - cfg.z_min) * Z).astype(np.int32)
    
    ix = np.clip(ix, 0, H - 1)
    iy = np.clip(iy, 0, W - 1)
    iz = np.clip(iz, 0, Z - 1)
    
    occupancy = np.zeros((H, W, Z), dtype=np.float32)
    occupancy[ix, iy, iz] = 1.0
    return occupancy


def visualize_point_cloud_bev(points: np.ndarray, title: str = "BEV"):
    """Visualize point cloud in Bird-Eye View."""
    plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=0.1, cmap='viridis')
    plt.colorbar(label='Height (z)')
    plt.xlabel('X (forward)')
    plt.ylabel('Y (left)')
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    return plt.gcf()


def visualize_depth_comparison(gt_depths: np.ndarray, pred_depths: np.ndarray, 
                               ray_dirs: np.ndarray, title: str = "Depth Comparison"):
    """Visualize ground truth vs predicted depths projected to 2D.
    
    Args:
        gt_depths: (R,) ground truth depths
        pred_depths: (R,) predicted depths
        ray_dirs: (R, 3) ray directions
    """
    # Project rays to 2D (x, y) using direction * depth
    gt_xy = ray_dirs[:, :2] * gt_depths[:, None]
    pred_xy = ray_dirs[:, :2] * pred_depths[:, None]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Ground truth
    axes[0].scatter(gt_xy[:, 0], gt_xy[:, 1], c=gt_depths, s=1, cmap='plasma', vmin=0, vmax=80)
    axes[0].set_title("Ground Truth Depths")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].axis('equal')
    axes[0].grid(True)
    
    # Predicted
    sc = axes[1].scatter(pred_xy[:, 0], pred_xy[:, 1], c=pred_depths, s=1, 
                          cmap='plasma', vmin=0, vmax=80)
    axes[1].set_title("Predicted Depths (Tokenizer Render)")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Y (m)")
    axes[1].axis('equal')
    axes[1].grid(True)
    
    # Error
    error = np.abs(pred_depths - gt_depths)
    axes[2].scatter(gt_xy[:, 0], gt_xy[:, 1], c=error, s=1, cmap='hot', vmin=0, vmax=20)
    axes[2].set_title(f"Absolute Error (L1={error.mean():.2f}m)")
    axes[2].set_xlabel("X (m)")
    axes[2].set_ylabel("Y (m)")
    axes[2].axis('equal')
    axes[2].grid(True)
    
    plt.colorbar(sc, ax=axes, label='Depth (m)', shrink=0.6)
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def main():
    print("=" * 70)
    print("CoPilot4D Tokenizer - KITTI Sample Render Demo")
    print("=" * 70)
    
    # Configuration
    cfg = TokenizerConfig()
    print(f"\nConfig:")
    print(f"  ROI: x=[{cfg.x_min}, {cfg.x_max}], y=[{cfg.y_min}, {cfg.y_max}], z=[{cfg.z_min}, {cfg.z_max}]")
    print(f"  Voxel grid: {cfg.voxel_grid_xy}x{cfg.voxel_grid_xy}x{cfg.voxel_grid_z}")
    print(f"  Token grid: {cfg.token_grid_size}x{cfg.token_grid_size}")
    print(f"  Codebook size: {cfg.vq_codebook_size}")
    
    # Load KITTI point cloud
    kitti_path = "/media/skr/storage/self_driving/CoPilot4D/data/kitti/dataset/sequences/00/velodyne/000000.bin"
    print(f"\nLoading KITTI point cloud from:\n  {kitti_path}")
    
    points = load_kitti_bin(kitti_path)
    print(f"  Original points: {len(points)}")
    print(f"  Intensity range: [{points[:, 3].min():.2f}, {points[:, 3].max():.2f}]")
    
    # Filter to ROI
    points = filter_roi(points, cfg)
    print(f"  Points after ROI filter: {len(points)}")
    
    # Voxelize
    print("\nVoxelizing point cloud...")
    voxel_data = voxelize_points(points, cfg)
    print(f"  Number of occupied voxels: {len(voxel_data['coords'])}")
    print(f"  Max points per voxel: {voxel_data['features'].shape[1]}")
    
    # Generate rays
    print("\nGenerating rays...")
    ray_data = generate_rays(points, cfg)
    print(f"  Total rays: {len(ray_data['ray_depths'])}")
    print(f"  Depth range: [{ray_data['ray_depths'].min():.2f}, {ray_data['ray_depths'].max():.2f}]m")
    
    # Sample rays for rendering
    ray_data = sample_training_rays(
        ray_data,
        num_rays=cfg.rays_per_frame,
        depth_min=cfg.ray_depth_min,
        depth_max=cfg.ray_depth_max,
    )
    print(f"  Sampled rays for rendering: {len(ray_data['ray_depths'])}")
    
    # Compute GT occupancy
    gt_occupancy = compute_gt_occupancy(points, cfg)
    
    # Prepare batch data (batch_size=1)
    print("\nPreparing tensors...")
    coords = torch.from_numpy(voxel_data["coords"]).long()
    coords[:, 0] = 0  # batch index
    features = torch.from_numpy(voxel_data["features"]).float()
    num_points = torch.from_numpy(voxel_data["num_points"]).long()
    
    ray_origins = torch.from_numpy(ray_data["ray_origins"]).float().unsqueeze(0)
    ray_directions = torch.from_numpy(ray_data["ray_directions"]).float().unsqueeze(0)
    ray_depths = torch.from_numpy(ray_data["ray_depths"]).float().unsqueeze(0)
    gt_occupancy = torch.from_numpy(gt_occupancy).float().unsqueeze(0)
    
    print(f"  coords: {coords.shape}")
    print(f"  features: {features.shape}")
    print(f"  ray_origins: {ray_origins.shape}")
    print(f"  ray_depths: {ray_depths.shape}")
    
    # Initialize tokenizer (untrained - random weights)
    print("\nInitializing tokenizer (untrained - random weights)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    tokenizer = CoPilot4DTokenizer(cfg).to(device)
    tokenizer.eval()  # eval mode
    
    # Count parameters
    num_params = sum(p.numel() for p in tokenizer.parameters())
    print(f"  Model parameters: {num_params:,}")
    
    # Move data to device
    coords = coords.to(device)
    features = features.to(device)
    num_points = num_points.to(device)
    ray_origins = ray_origins.to(device)
    ray_directions = ray_directions.to(device)
    ray_depths = ray_depths.to(device)
    gt_occupancy = gt_occupancy.to(device)
    
    # Forward pass
    print("\n" + "=" * 70)
    print("Running tokenizer forward pass...")
    print("=" * 70)
    
    with torch.no_grad():
        output = tokenizer(
            features=features,
            num_points=num_points,
            coords=coords,
            batch_size=1,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            gt_depths=ray_depths,
            gt_occupancy=gt_occupancy,
        )
    
    print("\nOutput keys:")
    for key, val in output.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")
    
    # Extract results
    pred_depths = output["pred_depths"][0].cpu().numpy()
    gt_depths_np = ray_depths[0].cpu().numpy()
    ray_dirs_np = ray_directions[0].cpu().numpy()
    token_indices = output["indices"][0].cpu().numpy()
    vq_loss = output["vq_loss"].item()
    
    print(f"\nToken indices shape: {token_indices.shape}")
    print(f"Unique tokens used: {len(np.unique(token_indices))}")
    print(f"VQ commitment loss: {vq_loss:.4f}")
    
    # Depth error
    depth_error = np.abs(pred_depths - gt_depths_np)
    print(f"\nDepth prediction error (untrained model):")
    print(f"  Mean L1 error: {depth_error.mean():.2f}m")
    print(f"  Median L1 error: {np.median(depth_error):.2f}m")
    print(f"  Max error: {depth_error.max():.2f}m")
    
    # Visualize
    print("\nGenerating visualizations...")
    
    # Create output directory
    output_dir = Path("outputs/render_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. BEV of original point cloud
    fig1 = visualize_point_cloud_bev(points, "KITTI Sample - Bird Eye View")
    fig1.savefig(output_dir / "01_original_bev.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_dir / '01_original_bev.png'}")
    
    # 2. Token grid visualization
    fig2, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(token_indices, cmap="tab20", interpolation="nearest")
    ax.set_title(f"Discrete Token Grid ({cfg.token_grid_size}x{cfg.token_grid_size})\n{len(np.unique(token_indices))} unique tokens")
    plt.colorbar(im, ax=ax, label="Token ID")
    plt.tight_layout()
    fig2.savefig(output_dir / "02_token_grid.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_dir / '02_token_grid.png'}")
    
    # 3. Depth comparison
    fig3 = visualize_depth_comparison(gt_depths_np, pred_depths, ray_dirs_np, 
                                       "Depth Render (Untrained Model)")
    fig3.savefig(output_dir / "03_depth_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_dir / '03_depth_comparison.png'}")
    
    # 4. Error histogram
    fig4, ax = plt.subplots(figsize=(10, 6))
    ax.hist(depth_error, bins=50, color='red', alpha=0.7, edgecolor='black')
    ax.axvline(depth_error.mean(), color='blue', linestyle='--', 
               label=f'Mean: {depth_error.mean():.2f}m')
    ax.set_xlabel("Absolute Error (m)")
    ax.set_ylabel("Count")
    ax.set_title("Depth Prediction Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig4.savefig(output_dir / "04_error_histogram.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_dir / '04_error_histogram.png'}")
    
    # 5. Depth scatter plot
    fig5, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(gt_depths_np, pred_depths, alpha=0.3, s=1)
    ax.plot([0, 80], [0, 80], 'r--', label='Perfect prediction')
    ax.set_xlabel("Ground Truth Depth (m)")
    ax.set_ylabel("Predicted Depth (m)")
    ax.set_title(f"Depth Correlation (Untrained)\nL1 Error: {depth_error.mean():.2f}m")
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 80)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    fig5.savefig(output_dir / "05_depth_correlation.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_dir / '05_depth_correlation.png'}")
    
    print("\n" + "=" * 70)
    print("Render complete!")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 70)
    print("\nNote: This demo uses an UNTRAINED model (random weights).")
    print("The high error is expected. A trained model would show:")
    print("  - Mean L1 error < 1-2 meters")
    print("  - Sharp depth predictions matching ground truth")
    print("  - Meaningful token usage patterns")
    
    # Optionally show plots
    plt.show()


if __name__ == "__main__":
    main()
