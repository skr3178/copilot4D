"""Point cloud utilities: ROI filtering, voxelization, ray generation."""

import torch
import numpy as np
from typing import Tuple, Dict

from copilot4d.utils.config import TokenizerConfig


def filter_roi(points: np.ndarray, cfg: TokenizerConfig) -> np.ndarray:
    """Filter point cloud to region of interest.

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


def voxelize_points(
    points: np.ndarray,
    cfg: TokenizerConfig,
) -> Dict[str, np.ndarray]:
    """Voxelize a filtered point cloud into sparse pillar representation.

    Args:
        points: (M, 4) filtered points [x, y, z, reflectance]
        cfg: TokenizerConfig

    Returns:
        Dictionary with:
            coords: (V, 3) int32 [batch_idx=0, ix, iy] voxel coordinates
            features: (V, max_pts, 4) float32 padded point features
            num_points: (V,) int32 number of actual points per voxel
    """
    H, W = cfg.voxel_grid_xy, cfg.voxel_grid_xy
    voxel_size_x = (cfg.x_max - cfg.x_min) / H
    voxel_size_y = (cfg.y_max - cfg.y_min) / W
    max_pts = cfg.max_points_per_voxel

    # Compute voxel indices (pillar = x,y only for BEV)
    ix = ((points[:, 0] - cfg.x_min) / voxel_size_x).astype(np.int32)
    iy = ((points[:, 1] - cfg.y_min) / voxel_size_y).astype(np.int32)

    # Clamp to valid range
    ix = np.clip(ix, 0, H - 1)
    iy = np.clip(iy, 0, W - 1)

    # Group points by pillar using a flat index
    flat_idx = ix * W + iy

    # Find unique pillars
    unique_pillars, inverse = np.unique(flat_idx, return_inverse=True)
    V = len(unique_pillars)

    features = np.zeros((V, max_pts, 4), dtype=np.float32)
    num_points = np.zeros(V, dtype=np.int32)
    coords = np.zeros((V, 3), dtype=np.int32)  # [batch_idx, ix, iy]

    for v_idx in range(V):
        pillar_mask = inverse == v_idx
        pillar_points = points[pillar_mask]
        n = min(len(pillar_points), max_pts)

        if len(pillar_points) > max_pts:
            choice = np.random.choice(len(pillar_points), max_pts, replace=False)
            pillar_points = pillar_points[choice]
            n = max_pts

        features[v_idx, :n] = pillar_points[:n]
        num_points[v_idx] = n

        flat = unique_pillars[v_idx]
        coords[v_idx, 0] = 0  # batch index, set later in collate
        coords[v_idx, 1] = flat // W
        coords[v_idx, 2] = flat % W

    return {
        "coords": coords,
        "features": features,
        "num_points": num_points,
    }


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
