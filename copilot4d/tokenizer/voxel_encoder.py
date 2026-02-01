"""VoxelPointNet: per-voxel point cloud feature extraction via MLP + sum pooling."""

import torch
import torch.nn as nn


class VoxelPointNet(nn.Module):
    """PointNet-style encoder for individual voxels.

    MLP(4 -> 16 -> 16) applied per-point, masked sum pooling, LayerNorm.

    Input:
        features: (V, max_pts, 4)  -- padded point features
        num_points: (V,)           -- actual point count per voxel

    Output:
        voxel_features: (V, out_dim)
    """

    def __init__(self, in_dim: int = 4, hidden_dim: int = 16, out_dim: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, features: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (V, P, 4) padded point features
            num_points: (V,) number of real points per voxel

        Returns:
            (V, out_dim) voxel features
        """
        V, P, _ = features.shape

        # Build mask: (V, P)
        arange = torch.arange(P, device=features.device).unsqueeze(0)  # (1, P)
        mask = arange < num_points.unsqueeze(1)  # (V, P)

        # MLP per point
        x = self.mlp(features)  # (V, P, out_dim)

        # Masked sum pooling
        x = x * mask.unsqueeze(-1).float()  # zero out padded points
        x = x.sum(dim=1)  # (V, out_dim)

        # LayerNorm
        x = self.norm(x)

        return x
