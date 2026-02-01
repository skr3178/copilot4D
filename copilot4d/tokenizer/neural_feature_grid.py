"""Neural Feature Grid (NFG): build 3D feature volume, trilinear query, volume rendering."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from copilot4d.utils.config import TokenizerConfig


class NeuralFeatureGrid(nn.Module):
    """Decode BEV features into a 3D neural feature grid and render depth via volume rendering.

    1. Reshape decoder output -> NFG of shape (B, nfg_H, nfg_W, nfg_Z, nfg_feat_dim)
    2. Query along rays via trilinear interpolation (F.grid_sample on 5D)
    3. MLP: feature -> occupancy alpha
    4. Volume rendering: weighted sum of depths
    """

    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        self.cfg = cfg

        dec_out_dim = cfg.dec_output_dim
        nfg_h = cfg.nfg_spatial_size
        nfg_w = cfg.nfg_spatial_size
        nfg_z = cfg.nfg_z_bins
        nfg_f = cfg.nfg_feat_dim

        # Head: project decoder output to NFG voxels
        # Each decoder token covers a (upsample_factor x upsample_factor x nfg_z) patch of NFG
        uf = cfg.nfg_upsample_factor
        self.nfg_head = nn.Sequential(
            nn.LayerNorm(dec_out_dim),
            nn.Linear(dec_out_dim, uf * uf * nfg_z * nfg_f),
        )
        self.uf = uf
        self.nfg_z = nfg_z
        self.nfg_f = nfg_f
        self.nfg_h = nfg_h
        self.nfg_w = nfg_w

        # Occupancy MLP: feature -> alpha
        self.occ_mlp = nn.Sequential(
            nn.Linear(nfg_f, cfg.nfg_mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.nfg_mlp_hidden, 1),
        )

        self.num_depth_samples = cfg.num_depth_samples
        self.ray_chunk_size = cfg.ray_chunk_size

    def build_nfg(self, decoder_output: torch.Tensor) -> torch.Tensor:
        """Build 3D NFG from decoder output.

        Args:
            decoder_output: (B, dec_grid^2, dec_out_dim) where dec_grid = decoder_output_grid_size

        Returns:
            nfg: (B, nfg_feat_dim, nfg_z, nfg_h, nfg_w) -- 5D for grid_sample
        """
        B = decoder_output.shape[0]
        dec_grid = int(decoder_output.shape[1] ** 0.5)

        x = self.nfg_head(decoder_output)  # (B, dec_grid^2, uf*uf*Z*F)
        x = x.view(B, dec_grid, dec_grid, self.uf, self.uf, self.nfg_z, self.nfg_f)

        # Rearrange to (B, dec_grid*uf, dec_grid*uf, Z, F) = (B, nfg_h, nfg_w, Z, F)
        x = x.permute(0, 1, 3, 2, 4, 5, 6)  # (B, dec_grid, uf, dec_grid, uf, Z, F)
        x = x.reshape(B, self.nfg_h, self.nfg_w, self.nfg_z, self.nfg_f)

        # Permute to (B, F, Z, H, W) for F.grid_sample(input=5D)
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        return x

    def query_rays(
        self,
        nfg: torch.Tensor,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        depth_min: float,
        depth_max: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query NFG along rays and perform volume rendering.

        Args:
            nfg: (B, F, Z, H, W) neural feature grid
            ray_origins: (B, R, 3)
            ray_directions: (B, R, 3)
            depth_min: minimum sample depth
            depth_max: maximum sample depth

        Returns:
            pred_depths: (B, R) predicted depth per ray
            weights: (B, R, S) volume rendering weights
        """
        B, R = ray_origins.shape[:2]
        device = nfg.device

        # Process rays in chunks
        all_depths = []
        all_weights = []

        for start in range(0, R, self.ray_chunk_size):
            end = min(start + self.ray_chunk_size, R)
            chunk_origins = ray_origins[:, start:end]      # (B, C, 3)
            chunk_dirs = ray_directions[:, start:end]       # (B, C, 3)
            chunk_R = end - start

            d, w = self._render_chunk(nfg, chunk_origins, chunk_dirs, depth_min, depth_max)
            all_depths.append(d)
            all_weights.append(w)

        pred_depths = torch.cat(all_depths, dim=1)  # (B, R)
        weights = torch.cat(all_weights, dim=1)      # (B, R, S)
        return pred_depths, weights

    def _render_chunk(
        self,
        nfg: torch.Tensor,
        origins: torch.Tensor,
        directions: torch.Tensor,
        depth_min: float,
        depth_max: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Volume render a chunk of rays.

        Args:
            nfg: (B, F, Z, H, W)
            origins: (B, C, 3)
            directions: (B, C, 3)
            depth_min, depth_max: depth sampling range

        Returns:
            pred_depth: (B, C)
            weights: (B, C, S)
        """
        B, C, _ = origins.shape
        S = self.num_depth_samples
        device = nfg.device

        # Sample depths uniformly
        t_vals = torch.linspace(0, 1, S, device=device)
        sample_depths = depth_min + (depth_max - depth_min) * t_vals  # (S,)
        sample_depths = sample_depths.view(1, 1, S).expand(B, C, S)  # (B, C, S)

        # Compute 3D sample positions: origin + t * direction
        # origins: (B,C,3), directions: (B,C,3), sample_depths: (B,C,S)
        pts = origins.unsqueeze(2) + directions.unsqueeze(2) * sample_depths.unsqueeze(3)
        # pts: (B, C, S, 3) -- xyz coordinates

        # Normalize coordinates to [-1, 1] for grid_sample
        cfg = self.cfg
        x_norm = 2.0 * (pts[..., 0] - cfg.x_min) / (cfg.x_max - cfg.x_min) - 1.0
        y_norm = 2.0 * (pts[..., 1] - cfg.y_min) / (cfg.y_max - cfg.y_min) - 1.0
        z_norm = 2.0 * (pts[..., 2] - cfg.z_min) / (cfg.z_max - cfg.z_min) - 1.0

        # grid_sample expects grid of shape (B, D_out, H_out, W_out, 3) for 5D input
        # nfg is (B, F, Z, H, W) -- depth=Z, height=H, width=W
        # grid coords: (x -> W, y -> H, z -> D) but grid_sample expects (x, y, z) = (W, H, D)
        grid = torch.stack([y_norm, x_norm, z_norm], dim=-1)  # (B, C, S, 3)
        # Reshape to (B, C, S, 1, 3) -- treating as (B, D_out=C, H_out=S, W_out=1, 3)
        grid = grid.unsqueeze(3)  # (B, C, S, 1, 3)

        # Query: nfg is (B, F, Z, H, W)
        # grid_sample with 5D: input=(B,C,D,H,W), grid=(B,D',H',W',3)
        features = F.grid_sample(
            nfg, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        # output: (B, F, C, S, 1)
        features = features.squeeze(-1).permute(0, 2, 3, 1)  # (B, C, S, F)

        # Occupancy MLP -> alpha
        alpha = self.occ_mlp(features).squeeze(-1)  # (B, C, S)
        alpha = torch.sigmoid(alpha)

        # Volume rendering
        # weights_i = alpha_i * cumprod(1 - alpha_j, j < i)
        one_minus_alpha = 1.0 - alpha + 1e-10
        # Exclusive cumprod: shift right and prepend 1
        transmittance = torch.cumprod(
            torch.cat([torch.ones(B, C, 1, device=device), one_minus_alpha[..., :-1]], dim=-1),
            dim=-1,
        )
        weights = alpha * transmittance  # (B, C, S)

        # Expected depth
        pred_depth = (weights * sample_depths).sum(dim=-1)  # (B, C)

        return pred_depth, weights

    def forward(
        self,
        decoder_output: torch.Tensor,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full NFG pipeline: build grid, query rays, volume render.

        Args:
            decoder_output: (B, dec_grid^2, dec_out_dim)
            ray_origins: (B, R, 3)
            ray_directions: (B, R, 3)

        Returns:
            pred_depths: (B, R)
            weights: (B, R, S)
            nfg: (B, F, Z, H, W)
        """
        nfg = self.build_nfg(decoder_output)
        pred_depths, weights = self.query_rays(
            nfg,
            ray_origins,
            ray_directions,
            depth_min=self.cfg.ray_depth_min,
            depth_max=self.cfg.ray_depth_max,
        )
        return pred_depths, weights, nfg
