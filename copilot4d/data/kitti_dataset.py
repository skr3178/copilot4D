"""KITTI Odometry dataset for tokenizer training."""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional

from copilot4d.utils.config import TokenizerConfig
from copilot4d.data.point_cloud_utils import (
    filter_roi,
    voxelize_points,
    generate_rays,
    sample_training_rays,
)

# Add pykitti to path
_PYKITTI_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "kitti", "pykitti")
if os.path.exists(_PYKITTI_PATH):
    sys.path.insert(0, _PYKITTI_PATH)


class KITTITokenizerDataset(Dataset):
    """KITTI Odometry dataset that returns voxelized point clouds and rays."""

    def __init__(
        self,
        cfg: TokenizerConfig,
        sequences: Optional[List[str]] = None,
        split: str = "train",
    ):
        super().__init__()
        self.cfg = cfg

        if sequences is None:
            if split == "train":
                sequences = cfg.train_sequences
            elif split == "val":
                sequences = cfg.val_sequences
            else:
                sequences = cfg.test_sequences

        self.samples = []  # list of (sequence, frame_idx)

        try:
            from pykitti import odometry
        except ImportError:
            raise ImportError(
                "pykitti not found. Make sure pykitti is installed or "
                "available at data/kitti/pykitti/"
            )

        self.datasets = {}
        for seq in sequences:
            dataset = odometry(cfg.kitti_root, seq)
            self.datasets[seq] = dataset
            num_frames = len(dataset)
            for idx in range(num_frames):
                self.samples.append((seq, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        seq, frame_idx = self.samples[index]
        dataset = self.datasets[seq]

        # Load point cloud (N, 4): [x, y, z, reflectance]
        points = dataset.get_velo(frame_idx)

        # Filter to ROI
        points = filter_roi(points, self.cfg)

        # Voxelize
        voxel_data = voxelize_points(points, self.cfg)

        # Generate and sample rays
        ray_data = generate_rays(points, self.cfg)
        ray_data = sample_training_rays(
            ray_data,
            num_rays=self.cfg.rays_per_frame,
            depth_min=self.cfg.ray_depth_min,
            depth_max=self.cfg.ray_depth_max,
        )

        # Generate ground truth occupancy grid for spatial skip loss
        gt_occupancy = self._compute_gt_occupancy(points)

        return {
            # Voxel data
            "coords": voxel_data["coords"],            # (V, 3) int32
            "features": voxel_data["features"],          # (V, max_pts, 4) float32
            "num_points": voxel_data["num_points"],      # (V,) int32
            # Ray data
            "ray_origins": ray_data["ray_origins"],      # (R, 3) float32
            "ray_directions": ray_data["ray_directions"],# (R, 3) float32
            "ray_depths": ray_data["ray_depths"],        # (R,) float32
            # Spatial skip GT
            "gt_occupancy": gt_occupancy,                # (H, W, Z) float32
        }

    def _compute_gt_occupancy(self, points: np.ndarray) -> np.ndarray:
        """Compute binary occupancy grid from point cloud.

        Returns:
            gt_occupancy: (H, W, Z) float32 binary grid
        """
        cfg = self.cfg
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


def tokenizer_collate_fn(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """Custom collate function that handles variable-length voxel data.

    Assigns batch indices to voxel coords and concatenates along the voxel dimension.
    Stacks ray data and occupancy grids normally.
    """
    all_coords = []
    all_features = []
    all_num_points = []
    all_ray_origins = []
    all_ray_directions = []
    all_ray_depths = []
    all_gt_occupancy = []

    for b_idx, sample in enumerate(batch):
        coords = sample["coords"].copy()
        coords[:, 0] = b_idx  # set batch index
        all_coords.append(coords)
        all_features.append(sample["features"])
        all_num_points.append(sample["num_points"])
        all_ray_origins.append(sample["ray_origins"])
        all_ray_directions.append(sample["ray_directions"])
        all_ray_depths.append(sample["ray_depths"])
        all_gt_occupancy.append(sample["gt_occupancy"])

    return {
        "coords": torch.from_numpy(np.concatenate(all_coords, axis=0)),       # (V_total, 3)
        "features": torch.from_numpy(np.concatenate(all_features, axis=0)),    # (V_total, max_pts, 4)
        "num_points": torch.from_numpy(np.concatenate(all_num_points, axis=0)),# (V_total,)
        "ray_origins": torch.from_numpy(np.stack(all_ray_origins, axis=0)),    # (B, R, 3)
        "ray_directions": torch.from_numpy(np.stack(all_ray_directions, axis=0)),# (B, R, 3)
        "ray_depths": torch.from_numpy(np.stack(all_ray_depths, axis=0)),      # (B, R)
        "gt_occupancy": torch.from_numpy(np.stack(all_gt_occupancy, axis=0)),  # (B, H, W, Z)
        "batch_size": len(batch),
    }
