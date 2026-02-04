"""BEVPillarPooling: scatter dense 3D voxel features into dense BEV feature map.

Following CoPilot4D spec:
"z-axis embedding concatenated to voxel features, then MLP projects to dim=64"
"pool the 3D feature volume into a 2D Bird-Eye View (BEV) representation,
using our aggregation function (sum operation + LayerNorm) on the z-axis"
"""

import torch
import torch.nn as nn


class BEVPillarPooling(nn.Module):
    """Project dense 3D voxel features to dense BEV grid.
    
    Steps:
    1. Concat z-embedding to voxel features (per-location)
    2. MLP to project to BEV feature dim
    3. Sum across z-axis (vertical pillars)
    4. LayerNorm
    """

    def __init__(
        self,
        voxel_dim: int = 16,
        z_bins: int = 64,
        bev_dim: int = 64,
    ):
        """
        Args:
            voxel_dim: Input voxel feature dimension (16 from PointNet)
            z_bins: Number of z-height bins (64)
            bev_dim: Output BEV feature dimension
        """
        super().__init__()
        self.voxel_dim = voxel_dim
        self.z_bins = z_bins
        self.bev_dim = bev_dim
        
        # Learnable z-axis embedding
        self.z_embed = nn.Embedding(z_bins, voxel_dim)
        
        # MLP: [voxel_features; z_embed] = 32-dim -> bev_dim
        # Spec: "z-axis embedding concatenated... then MLP projects to dim=64"
        self.mlp = nn.Sequential(
            nn.Linear(voxel_dim * 2, bev_dim),  # 16+16=32 -> 64
            nn.ReLU(inplace=True),
        )
        
        self.norm = nn.LayerNorm(bev_dim)

    def forward(
        self,
        dense_volume: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            dense_volume: (B, H, W, Z, voxel_dim) dense 3D features from PointNet

        Returns:
            bev: (B, H, W, bev_dim) dense BEV features
        """
        B, H, W, Z, C = dense_volume.shape
        
        # Create z-coordinate indices for embedding lookup
        # z_indices: (Z,) -> [0, 1, 2, ..., 63]
        z_indices = torch.arange(Z, device=dense_volume.device)
        
        # Get z-embeddings: (Z, voxel_dim)
        z_emb = self.z_embed(z_indices)
        
        # Expand to match volume shape: (B, H, W, Z, voxel_dim)
        z_emb = z_emb.view(1, 1, 1, Z, C).expand(B, H, W, Z, C)
        
        # CONCAT z-embedding with features (spec: "concatenated")
        # (B, H, W, Z, 16) + (B, H, W, Z, 16) -> (B, H, W, Z, 32)
        x = torch.cat([dense_volume, z_emb], dim=-1)
        
        # MLP projects to bev_dim
        x = self.mlp(x)  # (B, H, W, Z, 64)
        
        # Sum across z-axis (vertical pillars) - paper: "sum operation"
        x = x.sum(dim=3)  # (B, H, W, 64)
        
        # LayerNorm - paper: "+ LayerNorm"
        x = self.norm(x)
        
        return x
