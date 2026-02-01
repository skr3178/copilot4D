"""Spatial Skip Branch: predict binary occupancy for full BEV resolution."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from copilot4d.utils.config import TokenizerConfig


class SpatialSkipBranch(nn.Module):
    """Spatial skip connection: upsample decoder features to full BEV and predict occupancy.

    1. LN -> Linear(decoder_dim -> skip_upsample_factor^2 * Z)
    2. Reshape to (B, skip_grid, skip_grid, Z)
    3. Sigmoid -> binary occupancy logits

    Used for auxiliary BCE loss against GT occupancy computed from point cloud.
    """

    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        self.cfg = cfg

        dec_out_dim = cfg.dec_output_dim
        uf = cfg.skip_upsample_factor
        z_bins = cfg.voxel_grid_z
        self.skip_grid = cfg.skip_grid_xy

        self.head = nn.Sequential(
            nn.LayerNorm(dec_out_dim),
            nn.Linear(dec_out_dim, uf * uf * z_bins),
        )
        self.uf = uf
        self.z_bins = z_bins

    def forward(self, decoder_output: torch.Tensor) -> torch.Tensor:
        """Predict occupancy logits.

        Args:
            decoder_output: (B, dec_grid^2, dec_out_dim)

        Returns:
            skip_logits: (B, skip_grid, skip_grid, Z) logits (before sigmoid)
        """
        B = decoder_output.shape[0]
        dec_grid = int(decoder_output.shape[1] ** 0.5)

        x = self.head(decoder_output)  # (B, dec_grid^2, uf*uf*Z)
        x = x.view(B, dec_grid, dec_grid, self.uf, self.uf, self.z_bins)

        # Rearrange to (B, dec_grid*uf, dec_grid*uf, Z) = (B, skip_grid, skip_grid, Z)
        x = x.permute(0, 1, 3, 2, 4, 5)  # (B, dec_grid, uf, dec_grid, uf, Z)
        x = x.reshape(B, self.skip_grid, self.skip_grid, self.z_bins)

        return x


def compute_gt_occupancy_sparse(
    coords: torch.Tensor,
    grid_h: int,
    grid_w: int,
    grid_z: int,
) -> torch.Tensor:
    """Compute dense occupancy grid from sparse voxel coordinates.

    Args:
        coords: (V, 3) [batch_idx, ix, iy] voxel coordinates
        grid_h: H dimension
        grid_w: W dimension
        grid_z: Z dimension

    Returns:
        occupancy: (B, H, W, Z) binary occupancy (0 or 1)
    """
    batch_idx = coords[:, 0].long()
    ix = coords[:, 1].long()
    iy = coords[:, 2].long()

    B = batch_idx.max().item() + 1

    # For pillar-based representation, we need z-coordinates
    # Here we assume all z-slices are occupied if pillar has points
    # In practice, the dataset returns GT occupancy computed from points
    # This is a fallback for sparse voxel data

    occupancy = torch.zeros(B, grid_h, grid_w, grid_z, device=coords.device)
    
    # Mark occupied pillars (all z-slices for simplicity in sparse case)
    # In full implementation, z-coords should be passed
    for b in range(B):
        mask = batch_idx == b
        if mask.any():
            occ_i = ix[mask]
            occ_j = iy[mask]
            occupancy[b, occ_i, occ_j, :] = 1.0

    return occupancy
