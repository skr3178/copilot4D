"""VoxelPointNet: per-voxel point cloud feature extraction.

Following the CoPilot4D paper (Section 4.1, Appendix A.2.1):
- Encodes the distance of each point to its corresponding voxel center
- Uses sum operation + LayerNorm (instead of max pooling as in original PointNet)
- This is a permutation-invariant aggregation function
- Output feature dimension: 64
"""

import torch
import torch.nn as nn


class VoxelPointNet(nn.Module):
    """PointNet-style encoder for individual voxels.
    
    Paper specification:
    - Input: Points with their distances to voxel center
    - MLP applied per-point
    - Sum pooling (not max pooling as in original PointNet) + LayerNorm
    - Output: 64-dim voxel features
    
    The 3D feature volume shape after this layer: 1024 x 1024 x 64 x 64
    (H x W x Z x C) where H=W=1024, Z=64, C=64
    """

    def __init__(self, in_dim: int = 4, hidden_dim: int = 16, out_dim: int = 16):
        """
        Args:
            in_dim: Input feature dimension (4 = distance to voxel center xyz + reflectance)
            hidden_dim: Hidden layer dimension (paper: 16)
            out_dim: Output feature dimension (paper: 16, then aggregated to BEV with 64)
        """
        super().__init__()
        self.out_dim = out_dim
        
        # MLP with LayerNorm inside (matches paper architecture)
        # Paper: Linear(4, 16) -> LN -> ReLU -> Linear(16, 16)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),  # 4 -> 16
            nn.LayerNorm(hidden_dim),        # LN(16)
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),  # 16 -> 16 (no ReLU after, closer to paper)
        )
        # Final LayerNorm after sum aggregation (paper: "sum operation + LayerNorm")
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, features: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (V, P, 4) padded point features [dx, dy, dz, reflectance]
                      where (dx, dy, dz) is distance to voxel center
            num_points: (V,) number of real points per voxel

        Returns:
            (V, out_dim) voxel features after sum pooling + LayerNorm
        """
        V, P, _ = features.shape

        # Build mask: (V, P)
        arange = torch.arange(P, device=features.device).unsqueeze(0)  # (1, P)
        mask = arange < num_points.unsqueeze(1)  # (V, P)

        # MLP per point
        x = self.mlp(features)  # (V, P, out_dim)

        # Masked SUM pooling (paper modification: use sum instead of max)
        x = x * mask.unsqueeze(-1).float()  # zero out padded points
        x = x.sum(dim=1)  # (V, out_dim) - sum pooling

        # LayerNorm (paper modification)
        x = self.norm(x)

        return x


class VoxelCenterComputer:
    """Compute distance from each point to its voxel center.
    
    This is used to prepare input features for VoxelPointNet.
    Paper: "encodes the distance of each point to its corresponding voxel center"
    """
    
    @staticmethod
    def compute_point_to_center_offset(
        points: torch.Tensor,
        voxel_coords: torch.Tensor,
        voxel_size_x: float,
        voxel_size_y: float,
        voxel_size_z: float,
        x_min: float,
        y_min: float,
        z_min: float,
    ) -> torch.Tensor:
        """Compute offset of each point from its voxel center.
        
        Args:
            points: (N, 3+) point coordinates [x, y, z, ...]
            voxel_coords: (N, 3) voxel indices [ix, iy, iz]
            voxel_size_x, voxel_size_y, voxel_size_z: voxel sizes
            x_min, y_min, z_min: ROI bounds
            
        Returns:
            offsets: (N, 3) distance to voxel center [dx, dy, dz]
        """
        # Compute voxel center in world coordinates
        # center_x = x_min + (ix + 0.5) * voxel_size_x
        center_x = x_min + (voxel_coords[:, 0].float() + 0.5) * voxel_size_x
        center_y = y_min + (voxel_coords[:, 1].float() + 0.5) * voxel_size_y
        center_z = z_min + (voxel_coords[:, 2].float() + 0.5) * voxel_size_z
        
        # Compute offset from center
        dx = points[:, 0] - center_x
        dy = points[:, 1] - center_y
        dz = points[:, 2] - center_z
        
        return torch.stack([dx, dy, dz], dim=1)


def convert_points_to_voxel_center_features(
    points: torch.Tensor,
    voxel_indices: torch.Tensor,
    reflectance: torch.Tensor,
    cfg,
) -> torch.Tensor:
    """Convert raw points to features relative to voxel centers.
    
    Paper: "encodes the distance of each point to its corresponding voxel center"
    
    Args:
        points: (N, 3) point coordinates [x, y, z]
        voxel_indices: (N, 3) voxel grid indices [ix, iy, iz]
        reflectance: (N,) reflectance values
        cfg: TokenizerConfig with voxel sizes and ROI bounds
        
    Returns:
        features: (N, 4) [dx, dy, dz, reflectance] distances to voxel center
    """
    computer = VoxelCenterComputer()
    
    offsets = computer.compute_point_to_center_offset(
        points,
        voxel_indices,
        cfg.voxel_size_x,
        cfg.voxel_size_y,
        cfg.voxel_size_z,
        cfg.x_min,
        cfg.y_min,
        cfg.z_min,
    )
    
    # Combine with reflectance: [dx, dy, dz, reflectance]
    features = torch.cat([offsets, reflectance.unsqueeze(-1)], dim=1)
    
    return features
