"""GPT-2 style causal temporal attention block.

Applies temporal attention across the T dimension at each spatial location
independently. Pre-norm architecture (LN before attention and MLP).
"""

import torch
import torch.nn as nn
import math

from copilot4d.tokenizer.swin_transformer import DropPath


def make_causal_mask(T: int, device: torch.device = None) -> torch.Tensor:
    """Lower-triangular causal mask (0 = attend, -inf = block).

    Returns: (T, T) float tensor
    """
    mask = torch.full((T, T), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


def make_identity_mask(T: int, device: torch.device = None) -> torch.Tensor:
    """Diagonal-only mask: each frame attends only to itself.

    Returns: (T, T) float tensor
    """
    mask = torch.full((T, T), float("-inf"), device=device)
    mask.fill_diagonal_(0.0)
    return mask


class TemporalBlock(nn.Module):
    """GPT-2 style causal temporal attention.

    forward(x: (B,T,N,C), temporal_mask: (T,T)) -> (B,T,N,C)

    Internally reshapes to (B*N, T, C) for temporal attention across T
    at each spatial location independently.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 drop_path: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, bias=True,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=False),
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

        # Reshape: (B, T, N, C) -> (B*N, T, C) for temporal attention
        x_r = x.permute(0, 2, 1, 3).reshape(B * N, T, C)

        # Pre-norm temporal attention
        residual = x_r
        x_normed = self.norm1(x_r)
        attn_out, _ = self.attn(
            x_normed, x_normed, x_normed,
            attn_mask=temporal_mask,
            need_weights=False,
        )
        x_r = residual + self.drop_path(attn_out)

        # Pre-norm MLP
        x_r = x_r + self.drop_path(self.mlp(self.norm2(x_r)))

        # Reshape back: (B*N, T, C) -> (B, T, N, C)
        x_r = x_r.reshape(B, N, T, C).permute(0, 2, 1, 3)

        return x_r
