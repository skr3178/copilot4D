#!/usr/bin/env python3
"""Test that voxelization matches CoPilot4D paper specifications.

Paper (Appendix A.2.1):
- Voxel size: 15.625cm x 15.625cm x 14.0625cm
- ROI: [-80m, 80m] x [-80m, 80m] x [-4.5m, 4.5m]
- After PointNet: 3D feature volume of 1024 x 1024 x 64 x 64
- Encodes distance to voxel center
- Sum pooling + LayerNorm (not max pooling)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from copilot4d.utils.config import TokenizerConfig


def load_kitti_bin(bin_path: str) -> np.ndarray:
    """Load KITTI point cloud from .bin file."""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points


def filter_roi(points: np.ndarray, cfg: TokenizerConfig) -> np.ndarray:
    """Filter to ROI [-80m, 80m] x [-80m, 80m] x [-4.5m, 4.5m]."""
    mask = (
        (points[:, 0] >= cfg.x_min) & (points[:, 0] < cfg.x_max) &
        (points[:, 1] >= cfg.y_min) & (points[:, 1] < cfg.y_max) &
        (points[:, 2] >= cfg.z_min) & (points[:, 2] < cfg.z_max)
    )
    return points[mask]


def voxelize_points_3d_paper(points: np.ndarray, cfg: TokenizerConfig):
    """3D voxelization matching paper specifications.
    
    Returns:
        - Voxel coordinates (ix, iy, iz)
        - Features: [dx, dy, dz, reflectance] (distance to voxel center)
        - Number of occupied voxels
    """
    H, W, Z = cfg.voxel_grid_xy, cfg.voxel_grid_xy, cfg.voxel_grid_z
    voxel_size_x = cfg.voxel_size_x  # 15.625cm
    voxel_size_y = cfg.voxel_size_y  # 15.625cm
    voxel_size_z = cfg.voxel_size_z  # 14.0625cm
    max_pts = cfg.max_points_per_voxel
    
    print(f"\nVoxelization parameters:")
    print(f"  Grid: {H} x {W} x {Z}")
    print(f"  Voxel size: {voxel_size_x*100:.4f}cm x {voxel_size_y*100:.4f}cm x {voxel_size_z*100:.4f}cm")
    print(f"  ROI: [{cfg.x_min}, {cfg.x_max}] x [{cfg.y_min}, {cfg.y_max}] x [{cfg.z_min}, {cfg.z_max}]")
    
    # Compute voxel indices
    ix = ((points[:, 0] - cfg.x_min) / voxel_size_x).astype(np.int32)
    iy = ((points[:, 1] - cfg.y_min) / voxel_size_y).astype(np.int32)
    iz = ((points[:, 2] - cfg.z_min) / voxel_size_z).astype(np.int32)
    
    # Clamp
    ix = np.clip(ix, 0, H - 1)
    iy = np.clip(iy, 0, W - 1)
    iz = np.clip(iz, 0, Z - 1)
    
    # Compute voxel centers
    center_x = cfg.x_min + (ix + 0.5) * voxel_size_x
    center_y = cfg.y_min + (iy + 0.5) * voxel_size_y
    center_z = cfg.z_min + (iz + 0.5) * voxel_size_z
    
    # Distance to voxel center (paper: "encodes the distance of each point to its voxel center")
    dx = points[:, 0] - center_x
    dy = points[:, 1] - center_y
    dz = points[:, 2] - center_z
    reflectance = points[:, 3]
    
    print(f"\nDistance to voxel center stats:")
    print(f"  dx: [{dx.min():.4f}, {dx.max():.4f}]m")
    print(f"  dy: [{dy.min():.4f}, {dy.max():.4f}]m")
    print(f"  dz: [{dz.min():.4f}, {dz.max():.4f}]m")
    print(f"  Expected range: ±{voxel_size_x/2:.4f}m, ±{voxel_size_y/2:.4f}m, ±{voxel_size_z/2:.4f}m")
    
    # Group by 3D voxel
    flat_idx = ix * (W * Z) + iy * Z + iz
    unique_voxels, inverse, counts = np.unique(flat_idx, return_inverse=True, return_counts=True)
    V = len(unique_voxels)
    
    print(f"\nOccupied voxels: {V:,}")
    print(f"  Expected range: ~50k-150k for typical scene")
    print(f"  Points per voxel: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
    
    # Expected 3D feature volume size
    expected_volume_size = H * W * Z * cfg.voxel_feat_dim
    print(f"\n3D Feature Volume:")
    print(f"  Shape: {H} x {W} x {Z} x {cfg.voxel_feat_dim}")
    print(f"  Total elements if dense: {expected_volume_size:,}")
    print(f"  Non-zero elements: {V * cfg.voxel_feat_dim:,}")
    print(f"  Sparsity: {V / (H * W * Z) * 100:.4f}%")
    
    return {
        "voxel_coords": np.stack([ix, iy, iz], axis=1),
        "voxel_indices_flat": flat_idx,
        "unique_voxels": unique_voxels,
        "point_to_voxel": inverse,
        "distances": np.stack([dx, dy, dz, reflectance], axis=1),
        "counts": counts,
        "num_voxels": V,
    }


def test_voxel_size_computation():
    """Verify voxel size matches paper specification."""
    print("\n" + "=" * 70)
    print("TEST: Voxel Size Computation")
    print("=" * 70)
    
    cfg = TokenizerConfig()
    
    # Paper: 15.625cm x 15.625cm x 14.0625cm
    expected_xy = 0.15625  # 15.625cm in meters
    expected_z = 0.140625  # 14.0625cm in meters
    
    print(f"\nExpected (from paper):")
    print(f"  XY voxel size: 15.625cm = {expected_xy}m")
    print(f"  Z voxel size: 14.0625cm = {expected_z}m")
    
    print(f"\nComputed from config:")
    print(f"  XY: {cfg.voxel_size_xy:.5f}m = {cfg.voxel_size_xy * 100:.4f}cm")
    print(f"  Z: {cfg.voxel_size_z:.5f}m = {cfg.voxel_size_z * 100:.4f}cm")
    
    # Verify
    assert abs(cfg.voxel_size_x - expected_xy) < 1e-6, "X voxel size mismatch"
    assert abs(cfg.voxel_size_y - expected_xy) < 1e-6, "Y voxel size mismatch"
    assert abs(cfg.voxel_size_z - expected_z) < 1e-6, "Z voxel size mismatch"
    
    print("\n✅ Voxel sizes match paper specification")
    
    # Grid dimensions
    print(f"\nGrid dimensions:")
    print(f"  Paper: 1024 x 1024 x 64")
    print(f"  Config: {cfg.voxel_grid_xy} x {cfg.voxel_grid_xy} x {cfg.voxel_grid_z}")
    
    assert cfg.voxel_grid_xy == 1024, "XY grid size mismatch"
    assert cfg.voxel_grid_z == 64, "Z grid size mismatch"
    
    print("✅ Grid dimensions match paper")


def test_roi_bounds():
    """Verify ROI matches paper specification."""
    print("\n" + "=" * 70)
    print("TEST: ROI Bounds")
    print("=" * 70)
    
    cfg = TokenizerConfig()
    
    print(f"\nPaper: [-80m, 80m] x [-80m, 80m] x [-4.5m, 4.5m]")
    print(f"Config: [{cfg.x_min}, {cfg.x_max}] x [{cfg.y_min}, {cfg.y_max}] x [{cfg.z_min}, {cfg.z_max}]")
    
    assert cfg.x_min == -80.0 and cfg.x_max == 80.0
    assert cfg.y_min == -80.0 and cfg.y_max == 80.0
    assert cfg.z_min == -4.5 and cfg.z_max == 4.5
    
    print("✅ ROI bounds match paper")


def test_feature_dimension():
    """Verify PointNet output dimension."""
    print("\n" + "=" * 70)
    print("TEST: Feature Dimension")
    print("=" * 70)
    
    cfg = TokenizerConfig()
    
    print(f"\nPaper: feature dimension 64")
    print(f"Config: voxel_feat_dim = {cfg.voxel_feat_dim}")
    
    assert cfg.voxel_feat_dim == 64, "Feature dimension mismatch"
    
    print("✅ Feature dimension matches paper")


def test_voxelization_on_kitti():
    """Test voxelization on actual KITTI data."""
    print("\n" + "=" * 70)
    print("TEST: Voxelization on KITTI Sample")
    print("=" * 70)
    
    cfg = TokenizerConfig()
    
    # Load KITTI
    kitti_path = "/media/skr/storage/self_driving/CoPilot4D/data/kitti/dataset/sequences/00/velodyne/000000.bin"
    points = load_kitti_bin(kitti_path)
    print(f"\nLoaded {len(points):,} points from KITTI")
    
    # Filter to ROI
    filtered = filter_roi(points, cfg)
    print(f"After ROI filter: {len(filtered):,} points")
    
    # Voxelize
    result = voxelize_points_3d_paper(filtered, cfg)
    
    # Verify distance ranges
    distances = result["distances"]
    dx, dy, dz = distances[:, 0], distances[:, 1], distances[:, 2]
    
    # Distance should be within ± half voxel size
    assert np.abs(dx).max() <= cfg.voxel_size_x / 2 + 1e-5, "dx out of range"
    assert np.abs(dy).max() <= cfg.voxel_size_y / 2 + 1e-5, "dy out of range"
    assert np.abs(dz).max() <= cfg.voxel_size_z / 2 + 1e-5, "dz out of range"
    
    print("\n✅ All distances within expected voxel bounds")
    
    return result, filtered


def visualize_voxelization(result, points):
    """Create visualizations of voxelization."""
    print("\n" + "=" * 70)
    print("Creating Visualizations")
    print("=" * 70)
    
    output_dir = Path("outputs/voxelization_paper")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    voxel_coords = result["voxel_coords"]
    
    # 1. Voxel grid occupancy (BEV projection)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Point cloud BEV
    axes[0].scatter(points[:, 0], points[:, 1], c=points[:, 2], s=0.1, cmap='viridis')
    axes[0].set_title(f"Point Cloud (BEV)\n{len(points):,} points")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].axis('equal')
    axes[0].set_xlim(-80, 80)
    axes[0].set_ylim(-80, 80)
    axes[0].grid(True, alpha=0.3)
    
    # Voxel occupancy BEV (project all z to 2D)
    unique_xy = np.unique(voxel_coords[:, :2], axis=0)
    axes[1].scatter(unique_xy[:, 1], unique_xy[:, 0], c='blue', s=0.5, alpha=0.5)
    axes[1].set_title(f"3D Voxel Occupancy (BEV Projection)\n{len(unique_xy):,} unique (x,y) pillars")
    axes[1].set_xlabel("Y voxel index")
    axes[1].set_ylabel("X voxel index")
    axes[1].set_xlim(0, 1024)
    axes[1].set_ylim(0, 1024)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / "01_voxel_occupancy.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_dir / '01_voxel_occupancy.png'}")
    plt.close()
    
    # 2. Height distribution in voxel grid
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Z voxel index histogram
    axes[0].hist(voxel_coords[:, 2], bins=64, color='steelblue', edgecolor='black')
    axes[0].set_xlabel("Z voxel index")
    axes[0].set_ylabel("Number of occupied voxels")
    axes[0].set_title("Height Distribution in Voxel Grid")
    axes[0].grid(True, alpha=0.3)
    
    # Side view showing voxelization
    iz = voxel_coords[:, 2]
    ix = voxel_coords[:, 0]
    axes[1].scatter(ix, iz, c='blue', s=0.5, alpha=0.3)
    axes[1].set_xlabel("X voxel index")
    axes[1].set_ylabel("Z voxel index")
    axes[1].set_title("Side View (Voxel Grid)")
    axes[1].set_ylim(0, 64)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / "02_height_distribution.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_dir / '02_height_distribution.png'}")
    plt.close()
    
    # 3. Distance to voxel center distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    distances = result["distances"]
    dx, dy, dz = distances[:, 0], distances[:, 1], distances[:, 2]
    
    axes[0].hist(dx, bins=50, color='red', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel("dx (m)")
    axes[0].set_title("Distance to Voxel Center (X)")
    axes[0].axvline(0, color='black', linestyle='--')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(dy, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel("dy (m)")
    axes[1].set_title("Distance to Voxel Center (Y)")
    axes[1].axvline(0, color='black', linestyle='--')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist(dz, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[2].set_xlabel("dz (m)")
    axes[2].set_title("Distance to Voxel Center (Z)")
    axes[2].axvline(0, color='black', linestyle='--')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / "03_distance_to_center.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_dir / '03_distance_to_center.png'}")
    plt.close()
    
    print(f"\nAll visualizations saved to: {output_dir.absolute()}")


def main():
    print("=" * 70)
    print("CoPilot4D Paper-Aligned Voxelization Test")
    print("=" * 70)
    
    # Run tests
    test_voxel_size_computation()
    test_roi_bounds()
    test_feature_dimension()
    result, filtered = test_voxelization_on_kitti()
    
    # Visualizations
    visualize_voxelization(result, filtered)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
All tests passed! Implementation matches paper specifications:

| Specification | Paper | Implementation | Status |
|--------------|-------|----------------|--------|
| Voxel size (XY) | 15.625cm | 15.625cm | ✅ |
| Voxel size (Z) | 14.0625cm | 14.0625cm | ✅ |
| Grid (XY) | 1024 | 1024 | ✅ |
| Grid (Z) | 64 | 64 | ✅ |
| ROI X | [-80m, 80m] | [-80m, 80m] | ✅ |
| ROI Y | [-80m, 80m] | [-80m, 80m] | ✅ |
| ROI Z | [-4.5m, 4.5m] | [-4.5m, 4.5m] | ✅ |
| Feature dim | 64 | 64 | ✅ |
| Distance to center | Yes | Yes | ✅ |

The voxelization now properly creates:
- 3D voxel grid: 1024 x 1024 x 64
- Features: [dx, dy, dz, reflectance] (distance to voxel center)
- Ready for sum pooling + LayerNorm (not max pooling)
""")


if __name__ == "__main__":
    main()
