"""CoPilot4DTokenizer: Full VQVAE tokenizer for LiDAR point clouds."""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional

from copilot4d.utils.config import TokenizerConfig
from copilot4d.tokenizer.voxel_encoder_dense import DenseVoxelPointNet
from copilot4d.tokenizer.bev_pooling import BEVPillarPooling
from copilot4d.tokenizer.swin_transformer import SwinEncoder, SwinDecoder
from copilot4d.tokenizer.vector_quantizer import VectorQuantizer
from copilot4d.tokenizer.neural_feature_grid import NeuralFeatureGrid
from copilot4d.tokenizer.spatial_skipping import SpatialSkipBranch, compute_spatial_skip_mask, filter_rays_by_spatial_skip
from copilot4d.tokenizer.tokenizer_losses import tokenizer_total_loss


class CoPilot4DTokenizer(nn.Module):
    """Full VQVAE tokenizer: LiDAR -> discrete BEV tokens -> depth rendering.

    Architecture:
        1. VoxelPointNet: per-voxel point features
        2. BEVPillarPooling: sparse -> dense BEV
        3. SwinEncoder: BEV -> token grid
        4. VectorQuantizer: continuous -> discrete tokens
        5. SwinDecoder: tokens -> upsampled features
        6. NeuralFeatureGrid: features -> 3D grid -> rendered depth
        7. SpatialSkipBranch: auxiliary occupancy prediction
    """

    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        self.cfg = cfg

        # Point cloud encoder (outputs dense 3D volume per spec)
        self.voxel_encoder = DenseVoxelPointNet(
            in_dim=4,
            hidden_dim=cfg.voxel_feat_dim,
            out_dim=cfg.voxel_feat_dim,
        )

        # BEV pooling (concat z-embedding per spec)
        self.bev_pooling = BEVPillarPooling(
            voxel_dim=cfg.voxel_feat_dim,
            z_bins=cfg.voxel_grid_z,
            bev_dim=cfg.bev_feat_dim,
        )

        # Swin encoder/decoder
        self.encoder = SwinEncoder(cfg)
        self.decoder = SwinDecoder(cfg)

        # Vector quantization (paper: memory bank + K-Means re-init)
        self.vq = VectorQuantizer(
            dim=cfg.vq_dim,
            codebook_size=cfg.vq_codebook_size,
            codebook_dim=cfg.vq_codebook_dim,
            commitment_cost=cfg.vq_commitment_cost,  # lambda_1 = 0.25
            codebook_cost=cfg.vq_codebook_cost,       # lambda_2 = 1.0
            kmeans_iters=cfg.vq_kmeans_iters,
            dead_threshold=cfg.vq_dead_threshold,     # 256 iterations
            dead_percentage=cfg.vq_dead_percentage,   # 3%
            min_iterations=cfg.vq_min_iterations,     # 200 iterations
        )

        # Neural feature grid for depth rendering
        self.nfg = NeuralFeatureGrid(cfg)

        # Spatial skip branch
        self.spatial_skip = SpatialSkipBranch(cfg)

    def encode_voxels(
        self,
        features: torch.Tensor,
        num_points: torch.Tensor,
        coords: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Encode sparse voxels to dense BEV features.

        Args:
            features: (V, max_pts, 4) padded point features per voxel
            num_points: (V,) number of real points per voxel
            coords: (V, 4) [batch_idx, ix, iy, iz] voxel coordinates (3D)
            batch_size: B

        Returns:
            bev: (B, C, H, W) BEV feature map
        """
        # DenseVoxelPointNet: outputs dense 3D volume (B, H, W, Z, 16)
        dense_volume = self.voxel_encoder(
            features, 
            num_points,
            coords,
            batch_size,
            self.cfg.voxel_grid_xy,
            self.cfg.voxel_grid_xy,
            self.cfg.voxel_grid_z,
        )  # (B, 1024, 1024, 64, 16)

        # BEV pooling: concat z-embedding, MLP, sum over Z
        bev = self.bev_pooling(dense_volume)  # (B, 1024, 1024, 64)

        # Permute to (B, C, H, W) for Swin
        bev = bev.permute(0, 3, 1, 2).contiguous()

        return bev

    def encode(
        self,
        features: torch.Tensor,
        num_points: torch.Tensor,
        coords: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full encode: voxels -> BEV -> tokens -> quantized tokens.

        Args:
            features: (V, max_pts, 4)
            num_points: (V,)
            coords: (V, 3)
            batch_size: B

        Returns:
            quantized: (B, num_tokens, vq_dim)
            indices: (B, token_grid, token_grid) discrete token indices
        """
        bev = self.encode_voxels(features, num_points, coords, batch_size)
        encoder_out = self.encoder(bev)  # (B, num_tokens, vq_dim)
        quantized, indices, _ = self.vq(encoder_out)  # (B, num_tokens, vq_dim), (B, num_tokens)

        # Reshape indices to spatial grid
        token_grid = self.cfg.token_grid_size
        indices = indices.view(batch_size, token_grid, token_grid)

        return quantized, indices

    def decode(
        self,
        quantized: torch.Tensor,
        ray_origins: Optional[torch.Tensor] = None,
        ray_directions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Decode quantized tokens to depth predictions and occupancy.

        Args:
            quantized: (B, num_tokens, vq_dim)
            ray_origins: (B, R, 3) optional, for depth rendering
            ray_directions: (B, R, 3) optional, for depth rendering

        Returns:
            dict with:
                decoder_output: (B, dec_grid^2, dec_output_dim)
                pred_depths: (B, R) if rays provided
                weights: (B, R, S) volume rendering weights if rays provided
                nfg: (B, F, Z, H, W) neural feature grid
                skip_logits: (B, skip_grid, skip_grid, Z) spatial skip logits
        """
        decoder_output = self.decoder(quantized)  # (B, dec_grid^2, dec_output_dim)

        # Spatial skip prediction
        skip_logits = self.spatial_skip(decoder_output)  # (B, skip_grid, skip_grid, Z)

        # Neural feature grid
        if ray_origins is not None and ray_directions is not None:
            pred_depths, weights, nfg = self.nfg(
                decoder_output, ray_origins, ray_directions
            )
            return {
                "decoder_output": decoder_output,
                "pred_depths": pred_depths,
                "weights": weights,
                "nfg": nfg,
                "skip_logits": skip_logits,
            }
        else:
            # Just build NFG without rendering
            nfg = self.nfg.build_nfg(decoder_output)
            return {
                "decoder_output": decoder_output,
                "nfg": nfg,
                "skip_logits": skip_logits,
            }

    def forward(
        self,
        features: torch.Tensor,
        num_points: torch.Tensor,
        coords: torch.Tensor,
        batch_size: int,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        gt_depths: Optional[torch.Tensor] = None,
        gt_occupancy: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass with optional loss computation.

        Args:
            features: (V, max_pts, 4) voxel point features
            num_points: (V,) points per voxel
            coords: (V, 3) voxel coordinates
            batch_size: B
            ray_origins: (B, R, 3)
            ray_directions: (B, R, 3)
            gt_depths: (B, R) optional ground truth depths
            gt_occupancy: (B, H, W, Z) optional ground truth occupancy

        Returns:
            dict with outputs and optional losses
        """
        # Encode
        bev = self.encode_voxels(features, num_points, coords, batch_size)
        encoder_out = self.encoder(bev)  # (B, num_tokens, vq_dim)

        # VQ
        quantized, indices, vq_loss = self.vq(encoder_out)

        # Reshape indices to spatial grid
        token_grid = self.cfg.token_grid_size
        indices = indices.view(batch_size, token_grid, token_grid)

        # Decode
        decode_out = self.decode(quantized, ray_origins, ray_directions)

        result = {
            "encoder_out": encoder_out,
            "quantized": quantized,
            "indices": indices,
            "vq_loss": vq_loss,
            **decode_out,
        }

        # Compute losses if GT provided
        if gt_depths is not None and gt_occupancy is not None:
            # Need sample depths for surface concentration loss
            # Reconstruct from NFG forward
            pred_depths = decode_out["pred_depths"]
            weights = decode_out["weights"]

            # Build sample depths (same as in NFG._render_chunk)
            S = self.cfg.num_depth_samples
            device = pred_depths.device
            t_vals = torch.linspace(0, 1, S, device=device)
            sample_depths = (
                self.cfg.ray_depth_min
                + (self.cfg.ray_depth_max - self.cfg.ray_depth_min) * t_vals
            )
            sample_depths = sample_depths.view(1, 1, S).expand(
                batch_size, ray_origins.shape[1], S
            )

            losses = tokenizer_total_loss(
                pred_depths=pred_depths,
                gt_depths=gt_depths,
                weights=weights,
                sample_depths=sample_depths,
                vq_loss=vq_loss,
                skip_logits=decode_out["skip_logits"],
                gt_occupancy=gt_occupancy,
                surface_conc_eps=self.cfg.surface_conc_eps,
                vq_weight=1.0,
            )
            result["losses"] = losses

        return result

    def get_tokens(
        self,
        features: torch.Tensor,
        num_points: torch.Tensor,
        coords: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Get discrete tokens for a point cloud (inference mode).

        Args:
            features: (V, max_pts, 4)
            num_points: (V,)
            coords: (V, 3)
            batch_size: B

        Returns:
            indices: (B, token_grid, token_grid) discrete token indices
        """
        _, indices = self.encode(features, num_points, coords, batch_size)
        return indices

    def render_from_tokens(
        self,
        indices: torch.Tensor,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Render depths from discrete tokens (inference mode).

        Args:
            indices: (B, token_grid, token_grid) discrete token indices
            ray_origins: (B, R, 3)
            ray_directions: (B, R, 3)

        Returns:
            dict with pred_depths, weights, nfg, skip_logits
        """
        # Look up codebook entries
        B = indices.shape[0]
        flat_indices = indices.view(B, -1)  # (B, num_tokens)
        quantized = self.vq.get_codebook_entry(flat_indices)

        return self.decode(quantized, ray_origins, ray_directions)

    def render_with_spatial_skip(
        self,
        indices: torch.Tensor,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        pool_factor: int = 8,
        threshold: float = 0.5,
        return_all_rays: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Render depths with spatial skipping for efficient inference.
        
        Implements the paper's spatial skipping (Appendix A.2.1):
        1. Decode tokens to get NFG and skip_logits
        2. Compute coarse occupancy mask from skip_logits
        3. Filter rays to only sample in occupied regions
        4. Render only filtered rays through NFG
        
        Args:
            indices: (B, token_grid, token_grid) discrete token indices
            ray_origins: (B, R, 3) all query rays
            ray_directions: (B, R, 3) all query ray directions
            pool_factor: Max pooling factor for spatial skip (paper uses 8)
            threshold: Probability threshold for binary mask
            return_all_rays: If True, return depths for all rays (set skipped to -1)
            
        Returns:
            dict with:
                pred_depths: (B, R) predicted depths (-1 for skipped rays if return_all_rays)
                nfg: (B, F, Z, H, W) neural feature grid
                skip_logits: (B, skip_H, skip_W, Z) spatial skip logits
                coarse_mask: (B, H//pool, W//pool, Z) binary skip mask
                num_sampled_rays: number of rays actually sampled
        """
        from copilot4d.tokenizer.spatial_skipping import compute_spatial_skip_mask
        
        B, R, _ = ray_origins.shape
        device = ray_origins.device
        
        # Decode tokens to get NFG and skip_logits
        flat_indices = indices.view(B, -1)
        quantized = self.vq.get_codebook_entry(flat_indices)
        
        # Get decoder output
        decoder_output = self.decoder(quantized)  # (B, dec_grid^2, dec_output_dim)
        
        # Get spatial skip logits
        skip_logits = self.spatial_skip(decoder_output)  # (B, skip_H, skip_W, Z)
        
        # Compute coarse occupancy mask for spatial skipping
        coarse_mask = compute_spatial_skip_mask(
            skip_logits,
            pool_factor=pool_factor,
            threshold=threshold,
            add_noise=False,  # No noise during inference
        )  # (B, H//pool, W//pool, Z)
        
        # Build NFG
        nfg = self.nfg.build_nfg(decoder_output)  # (B, F, Z, H, W)
        
        # Check which rays intersect occupied regions
        # Simplified: for each ray, check if origin or a near point is in occupied region
        cfg = self.cfg
        
        # Sample test points along rays
        num_test = 8
        test_depths = torch.linspace(2.0, 80.0, num_test, device=device)
        test_points = ray_origins.unsqueeze(2) + ray_directions.unsqueeze(2) * test_depths.view(1, 1, -1, 1)
        # (B, R, num_test, 3)
        
        # Normalize to coarse_mask coordinates
        H, W, Z = coarse_mask.shape[1:]
        x_grid = ((test_points[..., 0] - cfg.x_min) / (cfg.x_max - cfg.x_min) * W).long()
        y_grid = ((test_points[..., 1] - cfg.y_min) / (cfg.y_max - cfg.y_min) * H).long()
        z_grid = ((test_points[..., 2] - cfg.z_min) / (cfg.z_max - cfg.z_min) * Z).long()
        
        x_grid = torch.clamp(x_grid, 0, W - 1)
        y_grid = torch.clamp(y_grid, 0, H - 1)
        z_grid = torch.clamp(z_grid, 0, Z - 1)
        
        # Check occupancy
        ray_should_sample = torch.zeros(B, R, dtype=torch.bool, device=device)
        for b in range(B):
            for r in range(R):
                xs, ys, zs = x_grid[b, r], y_grid[b, r], z_grid[b, r]
                ray_should_sample[b, r] = (coarse_mask[b, ys, xs, zs] > 0).any()
        
        # Initialize output depths
        if return_all_rays:
            pred_depths = torch.full((B, R), -1.0, device=device)
        
        # Render only rays that need sampling
        num_sampled = 0
        all_weights = []
        
        for b in range(B):
            mask = ray_should_sample[b]
            if mask.sum() > 0:
                num_sampled += mask.sum().item()
                origins_b = ray_origins[b, mask].unsqueeze(0)  # (1, R', 3)
                dirs_b = ray_directions[b, mask].unsqueeze(0)  # (1, R', 3)
                
                # Render this batch
                depths_b, weights_b = self.nfg.query_rays(
                    nfg[b:b+1], origins_b, dirs_b,
                    cfg.ray_depth_min, cfg.ray_depth_max
                )
                
                if return_all_rays:
                    pred_depths[b, mask] = depths_b[0]
                else:
                    # Store separately
                    if b == 0:
                        pred_depths = depths_b[0]
                    else:
                        pred_depths = torch.cat([pred_depths, depths_b[0]], dim=0)
        
        return {
            "pred_depths": pred_depths,
            "nfg": nfg,
            "skip_logits": skip_logits,
            "coarse_mask": coarse_mask,
            "num_sampled_rays": num_sampled,
            "total_rays": B * R,
            "skip_ratio": 1.0 - (num_sampled / (B * R)),
        }
