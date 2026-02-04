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

        self.norm = nn.LayerNorm(dec_out_dim)
        self.linear = nn.Linear(dec_out_dim, uf * uf * z_bins)
        # Paper: bias initialized to -5.0 since most voxels are empty
        nn.init.constant_(self.linear.bias, -5.0)
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

        x = self.norm(decoder_output)  # LayerNorm
        x = self.linear(x)  # (B, dec_grid^2, uf*uf*Z)
        x = x.view(B, dec_grid, dec_grid, self.uf, self.uf, self.z_bins)

        # Rearrange to (B, dec_grid*uf, dec_grid*uf, Z) = (B, skip_grid, skip_grid, Z)
        x = x.permute(0, 1, 3, 2, 4, 5)  # (B, dec_grid, uf, dec_grid, uf, Z)
        x = x.reshape(B, self.skip_grid, self.skip_grid, self.z_bins)

        return x


def compute_spatial_skip_mask(
    skip_logits: torch.Tensor,
    pool_factor: int = 8,
    threshold: float = 0.5,
    add_noise: bool = False,
    noise_scale: float = 0.1,
) -> torch.Tensor:
    """Compute spatial skip mask from skip logits for efficient ray sampling.
    
    Implements the inference-time spatial skipping from the paper (Appendix A.2.1):
    1. Apply sigmoid to get probabilities
    2. Optionally add Logistic noise for stochasticity
    3. Threshold to get binary mask
    4. Max pool to get coarser mask (increases recall)
    
    Args:
        skip_logits: (B, H, W, Z) logits from SpatialSkipBranch
        pool_factor: Max pooling factor in BEV (paper uses 8)
        threshold: Probability threshold for binary mask
        add_noise: Whether to add Logistic noise (paper mentions this)
        noise_scale: Scale of Logistic noise
        
    Returns:
        coarse_mask: (B, H//pool_factor, W//pool_factor, Z) binary mask
                     indicating which regions to sample
    """
    # Get probabilities
    probs = torch.sigmoid(skip_logits)  # (B, H, W, Z)
    
    # Add Logistic noise if requested (mentioned in paper, Maddison et al. 2016)
    if add_noise and probs.requires_grad is False:  # Only during inference
        # Logistic noise: sample from Uniform, apply logit transform
        uniform = torch.rand_like(probs)
        # Add small epsilon to avoid log(0)
        noise = torch.log(uniform + 1e-8) - torch.log(1 - uniform + 1e-8)
        probs = probs + noise_scale * noise
        probs = torch.clamp(probs, 0, 1)
    
    # Threshold to get binary mask
    binary_mask = (probs > threshold).float()  # (B, H, W, Z)
    
    # Max pooling in BEV to get coarser mask (increases recall)
    # Paper: "max pooling factor of 8 in Bird-Eye View"
    if pool_factor > 1:
        B, H, W, Z = binary_mask.shape
        # Reshape for pooling: (B, Z, H, W) -> pool in H, W
        binary_mask = binary_mask.permute(0, 3, 1, 2)  # (B, Z, H, W)
        
        # Max pool with kernel_size=pool_factor, stride=pool_factor
        coarse_mask = F.max_pool2d(
            binary_mask, 
            kernel_size=pool_factor, 
            stride=pool_factor
        )  # (B, Z, H//pool, W//pool)
        
        coarse_mask = coarse_mask.permute(0, 2, 3, 1)  # (B, H//pool, W//pool, Z)
    else:
        coarse_mask = binary_mask
        
    return coarse_mask  # Binary mask: 1 = has points (sample here), 0 = empty (skip)


def filter_rays_by_spatial_skip(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    coarse_mask: torch.Tensor,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> tuple:
    """Filter rays to only sample in regions marked by coarse_mask.
    
    For each ray, check if it intersects any occupied coarse voxel.
    Only return rays that need sampling (for which coarse_mask indicates occupancy).
    
    Args:
        ray_origins: (B, R, 3) ray starting points
        ray_directions: (B, R, 3) ray directions
        coarse_mask: (B, H, W, Z) binary mask from compute_spatial_skip_mask
        x_min, x_max, y_min, y_max, z_min, z_max: ROI bounds
        
    Returns:
        filtered_origins: (B, R', 3) rays that intersect occupied regions
        filtered_directions: (B, R', 3) corresponding directions
        ray_indices: (B, R') original indices of kept rays
    """
    B, R, _ = ray_origins.shape
    device = ray_origins.device
    
    H, W, Z = coarse_mask.shape[1:]
    
    # For each ray, check intersection with occupied coarse voxels
    # This is a simplified version - full implementation would do proper ray-voxel intersection
    
    # Sample a few points along each ray and check if any falls in occupied region
    num_test_samples = 16
    t_vals = torch.linspace(0, 1, num_test_samples, device=device)
    
    # Compute test points along rays (normalized depth)
    test_depths = 2.0 + 78.0 * t_vals  # Sample from 2m to 80m
    test_points = ray_origins.unsqueeze(2) + ray_directions.unsqueeze(2) * test_depths.view(1, 1, -1, 1)
    # (B, R, num_test_samples, 3)
    
    # Normalize to grid coordinates
    x_grid = ((test_points[..., 0] - x_min) / (x_max - x_min) * W).long()
    y_grid = ((test_points[..., 1] - y_min) / (y_max - y_min) * H).long()
    z_grid = ((test_points[..., 2] - z_min) / (z_max - z_min) * Z).long()
    
    # Clamp to valid range
    x_grid = torch.clamp(x_grid, 0, W - 1)
    y_grid = torch.clamp(y_grid, 0, H - 1)
    z_grid = torch.clamp(z_grid, 0, Z - 1)
    
    # Check if any test point falls in occupied voxel
    occupied = torch.zeros(B, R, dtype=torch.bool, device=device)
    for b in range(B):
        for r in range(R):
            # Sample points for this ray
            xs = x_grid[b, r]
            ys = y_grid[b, r]
            zs = z_grid[b, r]
            # Check if any point is in occupied voxel
            occupied[b, r] = (coarse_mask[b, ys, xs, zs] > 0).any()
    
    # Filter rays
    batch_indices = []
    ray_indices = []
    filtered_origins_list = []
    filtered_dirs_list = []
    
    for b in range(B):
        mask = occupied[b]
        if mask.any():
            batch_indices.extend([b] * mask.sum().item())
            ray_indices.append(torch.where(mask)[0])
            filtered_origins_list.append(ray_origins[b, mask])
            filtered_dirs_list.append(ray_directions[b, mask])
    
    if len(filtered_origins_list) == 0:
        # No rays intersect occupied regions - return empty
        return (
            torch.zeros(B, 0, 3, device=device),
            torch.zeros(B, 0, 3, device=device),
            [torch.tensor([], dtype=torch.long, device=device) for _ in range(B)]
        )
    
    # Stack results
    filtered_origins = torch.stack([
        filtered_origins_list[b] if b < len(filtered_origins_list) else torch.zeros(0, 3, device=device)
        for b in range(B)
    ])
    filtered_directions = torch.stack([
        filtered_dirs_list[b] if b < len(filtered_dirs_list) else torch.zeros(0, 3, device=device)
        for b in range(B)
    ])
    
    return filtered_origins, filtered_directions, ray_indices


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
