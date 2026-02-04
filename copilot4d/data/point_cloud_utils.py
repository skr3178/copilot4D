"""Point cloud utilities: ROI filtering, voxelization, ray generation.

Following CoPilot4D paper:
- Voxelization creates 3D feature volume: 1024 x 1024 x 64 x 64 (H x W x Z x C)
- Each point is represented as distance to its voxel center
- After PointNet, pool 3D volume to 2D BEV representation
"""

import torch
import numpy as np
from typing import Tuple, Dict

from copilot4d.utils.config import TokenizerConfig


def filter_roi(points: np.ndarray, cfg: TokenizerConfig) -> np.ndarray:
    """Filter point cloud to region of interest.
    
    Paper: model 3D world in [-80m, 80m] x [-80m, 80m] x [-4.5m, 4.5m]

    Args:
        points: (N, 4) array [x, y, z, reflectance]
        cfg: TokenizerConfig

    Returns:
        Filtered points (M, 4) within ROI bounds.
    """
    mask = (
        (points[:, 0] >= cfg.x_min) & (points[:, 0] < cfg.x_max) &
        (points[:, 1] >= cfg.y_min) & (points[:, 1] < cfg.y_max) &
        (points[:, 2] >= cfg.z_min) & (points[:, 2] < cfg.z_max)
    )
    return points[mask]


def voxelize_points_3d(
    points: np.ndarray,
    cfg: TokenizerConfig,
) -> Dict[str, np.ndarray]:
    """Voxelize a filtered point cloud into 3D voxel representation.
    
    Paper specification:
    - Voxel size: 15.625cm x 15.625cm x 14.0625cm
    - Grid: 1024 x 1024 x 64
    - Each point is represented as distance to its voxel center [dx, dy, dz, reflectance]
    - After PointNet: 3D feature volume of 1024 x 1024 x 64 x 64
    
    Args:
        points: (M, 4) filtered points [x, y, z, reflectance]
        cfg: TokenizerConfig

    Returns:
        Dictionary with:
            coords: (V, 4) int32 [batch_idx=0, ix, iy, iz] voxel coordinates
            features: (V, max_pts, 4) float32 padded point features [dx, dy, dz, reflectance]
            num_points: (V,) int32 number of actual points per voxel
            point_voxel_indices: (M,) int32 mapping from point to voxel
    """
    H, W, Z = cfg.voxel_grid_xy, cfg.voxel_grid_xy, cfg.voxel_grid_z
    voxel_size_x = cfg.voxel_size_x  # 15.625cm
    voxel_size_y = cfg.voxel_size_y  # 15.625cm
    voxel_size_z = cfg.voxel_size_z  # 14.0625cm
    max_pts = cfg.max_points_per_voxel

    # Compute voxel indices for each point
    ix = ((points[:, 0] - cfg.x_min) / voxel_size_x).astype(np.int32)
    iy = ((points[:, 1] - cfg.y_min) / voxel_size_y).astype(np.int32)
    iz = ((points[:, 2] - cfg.z_min) / voxel_size_z).astype(np.int32)

    # Clamp to valid range
    ix = np.clip(ix, 0, H - 1)
    iy = np.clip(iy, 0, W - 1)
    iz = np.clip(iz, 0, Z - 1)

    # Compute voxel centers in world coordinates
    center_x = cfg.x_min + (ix + 0.5) * voxel_size_x
    center_y = cfg.y_min + (iy + 0.5) * voxel_size_y
    center_z = cfg.z_min + (iz + 0.5) * voxel_size_z

    # Compute distance to voxel center (paper: encodes distance to voxel center)
    dx = points[:, 0] - center_x
    dy = points[:, 1] - center_y
    dz = points[:, 2] - center_z
    reflectance = points[:, 3]

    # Group points by 3D voxel using flat index
    flat_idx = ix * (W * Z) + iy * Z + iz

    # Find unique voxels
    unique_voxels, inverse = np.unique(flat_idx, return_inverse=True)
    V = len(unique_voxels)

    # Allocate output arrays
    features = np.zeros((V, max_pts, 4), dtype=np.float32)  # [dx, dy, dz, reflectance]
    num_points = np.zeros(V, dtype=np.int32)
    coords = np.zeros((V, 4), dtype=np.int32)  # [batch_idx, ix, iy, iz]

    for v_idx in range(V):
        voxel_mask = inverse == v_idx
        voxel_points_dx = dx[voxel_mask]
        voxel_points_dy = dy[voxel_mask]
        voxel_points_dz = dz[voxel_mask]
        voxel_points_refl = reflectance[voxel_mask]
        n = min(len(voxel_points_dx), max_pts)

        if len(voxel_points_dx) > max_pts:
            # Random sample if too many points
            choice = np.random.choice(len(voxel_points_dx), max_pts, replace=False)
            voxel_points_dx = voxel_points_dx[choice]
            voxel_points_dy = voxel_points_dy[choice]
            voxel_points_dz = voxel_points_dz[choice]
            voxel_points_refl = voxel_points_refl[choice]
            n = max_pts

        # Store [dx, dy, dz, reflectance]
        features[v_idx, :n, 0] = voxel_points_dx[:n]
        features[v_idx, :n, 1] = voxel_points_dy[:n]
        features[v_idx, :n, 2] = voxel_points_dz[:n]
        features[v_idx, :n, 3] = voxel_points_refl[:n]
        num_points[v_idx] = n

        # Store voxel coordinates
        flat = unique_voxels[v_idx]
        coords[v_idx, 0] = 0  # batch index
        coords[v_idx, 1] = flat // (W * Z)  # ix
        coords[v_idx, 2] = (flat // Z) % W  # iy
        coords[v_idx, 3] = flat % Z         # iz

    return {
        "coords": coords,
        "features": features,
        "num_points": num_points,
    }


def pool_3d_to_bev(
    voxel_features: np.ndarray,
    voxel_coords: np.ndarray,
    cfg: TokenizerConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pool 3D voxel features to 2D BEV representation.
    
    Paper: "pool the 3D feature volume into a 2D Bird-Eye View (BEV) representation,
    using our aggregation function (sum operation + LayerNorm) on the z-axis,
    after going through another Linear layer and adding a learnable embedding 
    based on the z-axis of a voxel."
    
    Args:
        voxel_features: (V, feat_dim) features from VoxelPointNet
        voxel_coords: (V, 4) [batch_idx, ix, iy, iz]
        cfg: TokenizerConfig
        
    Returns:
        bev_features: (H, W, feat_dim) aggregated BEV features
        bev_coords: (V_bev, 3) [batch_idx, ix, iy] for BEV pillars
    """
    H, W = cfg.voxel_grid_xy, cfg.voxel_grid_xy
    
    # Group by (ix, iy) - collapse z-axis
    ix = voxel_coords[:, 1]
    iy = voxel_coords[:, 2]
    iz = voxel_coords[:, 3]
    
    # Create flat index for BEV (x, y only)
    bev_flat_idx = ix * W + iy
    unique_bev, inverse = np.unique(bev_flat_idx, return_inverse=True)
    
    V_bev = len(unique_bev)
    feat_dim = voxel_features.shape[1]
    
    # Aggregate by sum along z-axis (paper: sum operation + LayerNorm)
    bev_features = np.zeros((V_bev, feat_dim), dtype=np.float32)
    bev_coords = np.zeros((V_bev, 3), dtype=np.int32)  # [batch_idx, ix, iy]
    
    for b_idx in range(V_bev):
        mask = inverse == b_idx
        # Sum aggregation (paper modification from max pooling)
        bev_features[b_idx] = voxel_features[mask].sum(axis=0)
        
        flat = unique_bev[b_idx]
        bev_coords[b_idx, 0] = 0  # batch index
        bev_coords[b_idx, 1] = flat // W  # ix
        bev_coords[b_idx, 2] = flat % W   # iy
    
    return bev_features, bev_coords


def generate_rays(
    points: np.ndarray,
    cfg: TokenizerConfig,
) -> Dict[str, np.ndarray]:
    """Generate rays from sensor origin to each point.

    Sensor origin is at (0,0,0) in Velodyne frame.

    Args:
        points: (M, 4) filtered points [x, y, z, reflectance]
        cfg: TokenizerConfig

    Returns:
        Dictionary with:
            ray_origins: (M, 3) all zeros (sensor at origin)
            ray_directions: (M, 3) unit direction vectors
            ray_depths: (M,) ground truth depth (distance to each point)
    """
    xyz = points[:, :3]
    depths = np.linalg.norm(xyz, axis=1)

    # Avoid division by zero
    safe_depths = np.maximum(depths, 1e-6)
    directions = xyz / safe_depths[:, None]

    origins = np.zeros_like(xyz)

    return {
        "ray_origins": origins.astype(np.float32),
        "ray_directions": directions.astype(np.float32),
        "ray_depths": depths.astype(np.float32),
    }


def sample_training_rays(
    ray_data: Dict[str, np.ndarray],
    num_rays: int,
    depth_min: float = 1.0,
    depth_max: float = 80.0,
) -> Dict[str, np.ndarray]:
    """Sample a subset of rays for training, filtered by depth range.

    Args:
        ray_data: output of generate_rays()
        num_rays: number of rays to sample
        depth_min: minimum depth threshold
        depth_max: maximum depth threshold

    Returns:
        Same keys as ray_data, subsampled to num_rays.
    """
    depths = ray_data["ray_depths"]
    valid_mask = (depths >= depth_min) & (depths <= depth_max)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        # Fallback: use all rays
        valid_indices = np.arange(len(depths))

    if len(valid_indices) >= num_rays:
        chosen = np.random.choice(valid_indices, num_rays, replace=False)
    else:
        chosen = np.random.choice(valid_indices, num_rays, replace=True)

    return {
        "ray_origins": ray_data["ray_origins"][chosen],
        "ray_directions": ray_data["ray_directions"][chosen],
        "ray_depths": ray_data["ray_depths"][chosen],
    }


# Backwards compatibility alias
voxelize_points = voxelize_points_3d
