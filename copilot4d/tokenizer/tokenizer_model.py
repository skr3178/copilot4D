"""CoPilot4DTokenizer: Full VQVAE tokenizer for LiDAR point clouds."""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional

from copilot4d.utils.config import TokenizerConfig
from copilot4d.tokenizer.voxel_encoder import VoxelPointNet
from copilot4d.tokenizer.bev_pooling import BEVPillarPooling
from copilot4d.tokenizer.swin_transformer import SwinEncoder, SwinDecoder
from copilot4d.tokenizer.vector_quantizer import VectorQuantizer
from copilot4d.tokenizer.neural_feature_grid import NeuralFeatureGrid
from copilot4d.tokenizer.spatial_skipping import SpatialSkipBranch
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

        # Point cloud encoder
        self.voxel_encoder = VoxelPointNet(
            in_dim=4,
            hidden_dim=cfg.voxel_feat_dim,
            out_dim=cfg.voxel_feat_dim,
        )

        # BEV pooling
        self.bev_pooling = BEVPillarPooling(
            voxel_dim=cfg.voxel_feat_dim,
            z_bins=cfg.voxel_grid_z,
            bev_dim=cfg.bev_feat_dim,
        )

        # Swin encoder/decoder
        self.encoder = SwinEncoder(cfg)
        self.decoder = SwinDecoder(cfg)

        # Vector quantization
        self.vq = VectorQuantizer(
            dim=cfg.vq_dim,
            codebook_size=cfg.vq_codebook_size,
            codebook_dim=cfg.vq_codebook_dim,
            commitment_cost=cfg.vq_commitment_cost,
            decay=cfg.vq_decay,
            kmeans_init=cfg.vq_kmeans_init,
            kmeans_iters=cfg.vq_kmeans_iters,
            threshold_ema_dead_code=cfg.vq_threshold_ema_dead_code,
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
            coords: (V, 3) [batch_idx, ix, iy] voxel coordinates
            batch_size: B

        Returns:
            bev: (B, C, H, W) BEV feature map
        """
        # VoxelPointNet
        voxel_feats = self.voxel_encoder(features, num_points)  # (V, voxel_dim)

        # BEV pooling
        bev = self.bev_pooling(
            voxel_feats,
            coords,
            batch_size,
            self.cfg.voxel_grid_xy,
            self.cfg.voxel_grid_xy,
        )  # (B, H, W, C)

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
