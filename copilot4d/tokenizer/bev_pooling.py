"""BEVPillarPooling: scatter sparse voxel features into dense BEV feature map."""

import torch
import torch.nn as nn


class BEVPillarPooling(nn.Module):
    """Project sparse voxel features to dense BEV grid via scatter_add.

    1. Add learned z-embedding
    2. Project to BEV feature dim
    3. Scatter-add into dense (B, H, W, C) grid
    4. LayerNorm

    Never materializes a (H, W, Z) tensor -- uses flat indexing.
    """

    def __init__(
        self,
        voxel_dim: int = 16,
        z_bins: int = 32,
        bev_dim: int = 64,
    ):
        super().__init__()
        self.z_embed = nn.Embedding(z_bins, voxel_dim)
        self.proj = nn.Linear(voxel_dim, bev_dim)
        self.norm = nn.LayerNorm(bev_dim)
        self.bev_dim = bev_dim

    def forward(
        self,
        voxel_features: torch.Tensor,
        coords: torch.Tensor,
        batch_size: int,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """
        Args:
            voxel_features: (V, voxel_dim) from VoxelPointNet
            coords: (V, 3) int [batch_idx, ix, iy]
            batch_size: B
            grid_h: H of BEV grid
            grid_w: W of BEV grid

        Returns:
            bev: (B, H, W, C) dense BEV features
        """
        # NOTE: z-embedding is not indexed by z-coord since we do pillar pooling
        # (all z-slices collapsed). We use a constant z=0 embedding for simplicity,
        # or we can skip the z-embed for pillar-based approach.
        # In the CoPilot4D paper, voxels have z-coords and z-embed is added before
        # scatter. For pillar approach, we just project directly.

        x = self.proj(voxel_features)  # (V, bev_dim)

        # Compute flat indices: batch_idx * H * W + ix * W + iy
        batch_idx = coords[:, 0].long()
        ix = coords[:, 1].long()
        iy = coords[:, 2].long()
        flat_idx = batch_idx * (grid_h * grid_w) + ix * grid_w + iy  # (V,)

        # Scatter add into dense grid
        bev_flat = torch.zeros(
            batch_size * grid_h * grid_w,
            self.bev_dim,
            device=x.device,
            dtype=x.dtype,
        )
        bev_flat.scatter_add_(0, flat_idx.unsqueeze(1).expand_as(x), x)

        # Reshape to (B, H, W, C)
        bev = bev_flat.view(batch_size, grid_h, grid_w, self.bev_dim)

        # LayerNorm
        bev = self.norm(bev)

        return bev
