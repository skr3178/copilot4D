"""Patch Merging for the world model U-Net encoder.

Same spatial grouping as tokenizer's PatchMerging but with custom target dim.
The tokenizer's PatchMerging does 4*dim -> 2*dim. The world model needs
4*256->384 and 4*384->512.
"""

import torch
import torch.nn as nn


class WorldModelPatchMerging(nn.Module):
    """2x2 patch grouping -> concat -> LN -> Linear to target_dim.

    forward(x: (B, H*W, C)) -> (B, H/2*W/2, target_dim)
    """

    def __init__(self, input_resolution: int, dim: int, target_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.target_dim = target_dim

        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, target_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, C) where H = W = input_resolution

        Returns:
            (B, H/2*W/2, target_dim)
        """
        H = W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W
        assert H % 2 == 0 and W % 2 == 0

        x = x.view(B, H, W, C)
        # Group 2x2 patches
        x0 = x[:, 0::2, 0::2, :]  # top-left
        x1 = x[:, 1::2, 0::2, :]  # bottom-left
        x2 = x[:, 0::2, 1::2, :]  # top-right
        x3 = x[:, 1::2, 1::2, :]  # bottom-right
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4*C)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        return x
