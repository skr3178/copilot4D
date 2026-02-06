"""CoPilot4D World Model: U-Net Spatio-Temporal Transformer.

Operates on discrete BEV tokens from the frozen tokenizer. Uses discrete
diffusion (improved MaskGIT) to predict future observations conditioned
on past observations and ego-vehicle actions.

Architecture (Figure 7, Section 4.4, Appendix A.2.2):
  Input: (B, T, 128, 128) discrete token indices + (B, T, 16) actions
  -> Embeddings (token + spatial + temporal + action)
  -> Encoder L1 (128x128, 256d) -> PatchMerge -> Encoder L2 (64x64, 384d)
  -> PatchMerge -> Bottleneck L3 (32x32, 512d)
  -> LevelMerge + skip -> Decoder L2 (64x64, 384d)
  -> LevelMerge + skip -> Decoder L1 (128x128, 256d)
  -> LayerNorm -> weight-tied Linear -> logits (B, T, H*W, 1025)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from copilot4d.utils.config import WorldModelConfig
from copilot4d.world_model.input_embeddings import WorldModelInputEmbedding
from copilot4d.world_model.spatio_temporal_block import SpatioTemporalBlock
from copilot4d.world_model.patch_merging import WorldModelPatchMerging
from copilot4d.world_model.level_merging import LevelMerging


class CoPilot4DWorldModel(nn.Module):
    """Complete U-Net Spatio-Temporal Transformer world model.

    forward(token_indices: (B,T,H,W), actions: (B,T,16), temporal_mask: (T,T))
        -> logits: (B, T, H*W, vocab_size)
    """

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        dims = cfg.level_dims       # (256, 384, 512)
        heads = cfg.level_heads     # (8, 12, 16)
        windows = cfg.level_windows  # (8, 8, 16)
        H = cfg.token_grid_h        # 128
        W = cfg.token_grid_w        # 128

        # Input embeddings
        self.input_embed = WorldModelInputEmbedding(cfg)

        # --- Encoder ---
        # Level 1: 128x128, dim=256
        res1 = H  # 128
        self.enc_level1 = nn.ModuleList([
            SpatioTemporalBlock(
                dim=dims[0], spatial_resolution=res1,
                num_heads=heads[0], window_size=windows[0],
                mlp_ratio=cfg.mlp_ratio, drop_path=cfg.drop_path_rate,
            )
            for _ in range(cfg.enc_st_blocks[0])
        ])
        # PatchMerge 1->2: 128x128, 256 -> 64x64, 384
        self.patch_merge1 = WorldModelPatchMerging(res1, dims[0], dims[1])

        # Level 2: 64x64, dim=384
        res2 = res1 // 2  # 64
        self.enc_level2 = nn.ModuleList([
            SpatioTemporalBlock(
                dim=dims[1], spatial_resolution=res2,
                num_heads=heads[1], window_size=windows[1],
                mlp_ratio=cfg.mlp_ratio, drop_path=cfg.drop_path_rate,
            )
            for _ in range(cfg.enc_st_blocks[1])
        ])
        # PatchMerge 2->3: 64x64, 384 -> 32x32, 512
        self.patch_merge2 = WorldModelPatchMerging(res2, dims[1], dims[2])

        # --- Bottleneck ---
        # Level 3: 32x32, dim=512
        res3 = res2 // 2  # 32
        self.bottleneck = nn.ModuleList([
            SpatioTemporalBlock(
                dim=dims[2], spatial_resolution=res3,
                num_heads=heads[2], window_size=windows[2],
                mlp_ratio=cfg.mlp_ratio, drop_path=cfg.drop_path_rate,
            )
            for _ in range(cfg.enc_st_blocks[2])
        ])

        # --- Decoder ---
        # LevelMerge 3->2: upsample 32x32,512 -> 64x64,384 + skip
        self.level_merge2 = LevelMerging(dims[2], dims[1], res3)
        self.dec_level2 = nn.ModuleList([
            SpatioTemporalBlock(
                dim=dims[1], spatial_resolution=res2,
                num_heads=heads[1], window_size=windows[1],
                mlp_ratio=cfg.mlp_ratio, drop_path=cfg.drop_path_rate,
            )
            for _ in range(cfg.dec_st_blocks[0])
        ])

        # LevelMerge 2->1: upsample 64x64,384 -> 128x128,256 + skip
        self.level_merge1 = LevelMerging(dims[1], dims[0], res2)
        self.dec_level1 = nn.ModuleList([
            SpatioTemporalBlock(
                dim=dims[0], spatial_resolution=res1,
                num_heads=heads[0], window_size=windows[0],
                mlp_ratio=cfg.mlp_ratio, drop_path=cfg.drop_path_rate,
            )
            for _ in range(cfg.dec_st_blocks[1])
        ])

        # --- Output ---
        self.output_norm = nn.LayerNorm(dims[0])
        # Weight-tied output bias
        self.output_bias = nn.Parameter(torch.zeros(cfg.vocab_size))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Paper A.3 initialization.

        Fan-in: Normal(0, sqrt(1/(3*H))) where H = input dim.
        Residual scaling: per level, count L transformer blocks * 2,
        scale residual output projections by sqrt(1/L).
        Bias = False everywhere except QKV in Swin WindowAttention.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fan_in = m.weight.shape[1]
                std = math.sqrt(1.0 / (3.0 * fan_in))
                nn.init.normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.ConvTranspose2d):
                fan_in = m.weight.shape[0]  # out_channels for ConvTranspose2d
                std = math.sqrt(1.0 / (3.0 * fan_in))
                nn.init.normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Residual scaling: scale output projections of attention and MLP
        # in each transformer block by sqrt(1/L) where L = total sub-layers
        cfg = self.cfg
        self._apply_residual_scaling(self.enc_level1, cfg.enc_st_blocks[0])
        self._apply_residual_scaling(self.enc_level2, cfg.enc_st_blocks[1])
        self._apply_residual_scaling(self.bottleneck, cfg.enc_st_blocks[2])
        self._apply_residual_scaling(self.dec_level2, cfg.dec_st_blocks[0])
        self._apply_residual_scaling(self.dec_level1, cfg.dec_st_blocks[1])

    def _apply_residual_scaling(self, blocks: nn.ModuleList, num_blocks: int):
        """Scale residual projections by sqrt(1/L).

        Each ST-block has: 2 Swin blocks (each with attn.proj + mlp.fc2)
        + 1 temporal block (attn output proj + mlp[-1]).
        So L = num_blocks * (2*2 + 2) = num_blocks * 6.
        """
        L = max(num_blocks * 6, 1)
        scale = math.sqrt(1.0 / L)
        for block in blocks:
            # Swin blocks
            for swin in [block.swin1, block.swin2]:
                swin.attn.proj.weight.data *= scale
                swin.mlp.fc2.weight.data *= scale
            # Temporal block
            # attn output projection
            block.temporal.attn.out_proj.weight.data *= scale
            # MLP last layer (Sequential: Linear, GELU, Linear)
            block.temporal.mlp[-1].weight.data *= scale

    def forward(
        self,
        token_indices: torch.Tensor,
        actions: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            token_indices: (B, T, H, W) long — discrete token indices (or masked)
            actions: (B, T, action_dim) float — flattened SE(3) transforms
            temporal_mask: (T, T) additive float mask (causal or identity)

        Returns:
            logits: (B, T, H*W, vocab_size) float
        """
        B, T, H, W = token_indices.shape
        N = H * W

        # Flatten spatial dims
        tokens_flat = token_indices.reshape(B, T, N)

        # Embed
        x = self.input_embed(tokens_flat, actions)  # (B, T, N, 256)

        # --- Encoder Level 1: 128x128, 256 ---
        for block in self.enc_level1:
            x = block(x, temporal_mask)
        skip_l1 = x  # (B, T, 16384, 256)

        # PatchMerge 1->2: process per-frame
        x = x.reshape(B * T, N, self.cfg.level_dims[0])
        x = self.patch_merge1(x)  # (B*T, 4096, 384)
        N2 = x.shape[1]
        x = x.reshape(B, T, N2, self.cfg.level_dims[1])

        # --- Encoder Level 2: 64x64, 384 ---
        for block in self.enc_level2:
            x = block(x, temporal_mask)
        skip_l2 = x  # (B, T, 4096, 384)

        # PatchMerge 2->3: process per-frame
        x = x.reshape(B * T, N2, self.cfg.level_dims[1])
        x = self.patch_merge2(x)  # (B*T, 1024, 512)
        N3 = x.shape[1]
        x = x.reshape(B, T, N3, self.cfg.level_dims[2])

        # --- Bottleneck Level 3: 32x32, 512 ---
        for block in self.bottleneck:
            x = block(x, temporal_mask)

        # --- Decoder Level 2: upsample + skip ---
        x = self.level_merge2(x, skip_l2)  # (B, T, 4096, 384)
        for block in self.dec_level2:
            x = block(x, temporal_mask)

        # --- Decoder Level 1: upsample + skip ---
        x = self.level_merge1(x, skip_l1)  # (B, T, 16384, 256)
        for block in self.dec_level1:
            x = block(x, temporal_mask)

        # --- Output: weight-tied linear ---
        x = self.output_norm(x)  # (B, T, N, 256)
        logits = F.linear(x, self.input_embed.token_embedding.weight, self.output_bias)
        # (B, T, N, vocab_size=1025)

        return logits
