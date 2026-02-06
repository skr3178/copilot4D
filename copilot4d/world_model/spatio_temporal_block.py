"""Spatio-Temporal Block: 2 Swin blocks (spatial) + 1 temporal block.

Each ST-block applies spatial attention per-frame via Swin, then
temporal attention across frames at each spatial location.
"""

import torch
import torch.nn as nn

from copilot4d.tokenizer.swin_transformer import SwinTransformerBlock
from copilot4d.world_model.temporal_block import TemporalBlock


class SpatioTemporalBlock(nn.Module):
    """2 Swin spatial blocks + 1 temporal block.

    forward(x: (B,T,N,C), temporal_mask: (T,T)) -> (B,T,N,C)

    Swin blocks operate per-frame: reshape (B,T,N,C) -> (B*T,N,C).
    Temporal block operates across time at each spatial location.
    Shift sizes alternate: 0 for first Swin block, window_size//2 for second.
    """

    def __init__(
        self,
        dim: int,
        spatial_resolution: int,
        num_heads: int,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        res = (spatial_resolution, spatial_resolution)

        self.swin1 = SwinTransformerBlock(
            dim=dim,
            input_resolution=res,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
        )
        self.swin2 = SwinTransformerBlock(
            dim=dim,
            input_resolution=res,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
        )
        self.temporal = TemporalBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
        )

    def forward(self, x: torch.Tensor, temporal_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, C)
            temporal_mask: (T, T) additive float mask

        Returns:
            (B, T, N, C)
        """
        B, T, N, C = x.shape

        # Spatial: flatten batch and time -> (B*T, N, C)
        x_flat = x.reshape(B * T, N, C)
        x_flat = self.swin1(x_flat)
        x_flat = self.swin2(x_flat)

        # Unflatten back to (B, T, N, C)
        x = x_flat.reshape(B, T, N, C)

        # Temporal: across T at each spatial location
        x = self.temporal(x, temporal_mask)

        return x
