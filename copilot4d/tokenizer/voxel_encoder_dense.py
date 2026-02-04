"""Dense VoxelPointNet: produces full 3D feature volume (1024 x 1024 x 64 x 16).

Following the CoPilot4D spec:
- Output: Dense 3D Voxel Feature Volume (1024, 1024, 64, 16)
"""

import torch
import torch.nn as nn


class DenseVoxelPointNet(nn.Module):
    """PointNet-style encoder that produces dense 3D voxel features.
    
    Input: Sparse voxel features (V, P, 4) with coordinates (V, 4)
    Output: Dense 3D feature volume (B, 1024, 1024, 64, 16)
    """

    def __init__(self, in_dim: int = 4, hidden_dim: int = 16, out_dim: int = 16):
        super().__init__()
        self.out_dim = out_dim
        
        # MLP with LayerNorm inside
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),  # 4 -> 16
            nn.LayerNorm(hidden_dim),        # LN(16)
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),  # 16 -> 16
        )
        # Final LayerNorm after sum aggregation
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        features: torch.Tensor,
        num_points: torch.Tensor,
        coords: torch.Tensor,
        batch_size: int,
        grid_h: int,
        grid_w: int,
        grid_z: int,
    ) -> torch.Tensor:
        """
        Args:
            features: (V, P, 4) padded point features [dx, dy, dz, reflectance]
            num_points: (V,) number of real points per voxel
            coords: (V, 4) [batch_idx, ix, iy, iz] voxel coordinates
            batch_size: B
            grid_h, grid_w, grid_z: grid dimensions (1024, 1024, 64)

        Returns:
            dense_volume: (B, grid_h, grid_w, grid_z, 16) dense 3D features
        """
        V, P, _ = features.shape

        # Build mask: (V, P)
        arange = torch.arange(P, device=features.device).unsqueeze(0)
        mask = arange < num_points.unsqueeze(1)

        # MLP per point
        x = self.mlp(features)  # (V, P, 16)

        # Masked SUM pooling
        x = x * mask.unsqueeze(-1).float()
        x = x.sum(dim=1)  # (V, 16)

        # LayerNorm
        x = self.norm(x)  # (V, 16)

        # Scatter into dense 3D volume
        # Output: (B, H, W, Z, 16)
        dense_volume = torch.zeros(
            batch_size, grid_h, grid_w, grid_z, self.out_dim,
            device=x.device, dtype=x.dtype
        )

        # Extract coordinates
        batch_idx = coords[:, 0].long()
        ix = coords[:, 1].long()
        iy = coords[:, 2].long()
        iz = coords[:, 3].long()

        # Scatter features into dense volume
        dense_volume[batch_idx, ix, iy, iz] = x

        return dense_volume
