"""Level Merging for the world model U-Net decoder.

Paper (A.2.2): "first we use a linear layer to output the 2x upsampled feature
map (similar to a deconvolution layer), concatenate with the lower-level feature
map, applies LayerNorm on every feature, and uses a linear projection to reduce
the feature dimension. A residual connection is then applied."
"""

import torch
import torch.nn as nn


class LevelMerging(nn.Module):
    """Decoder upsampling with skip connection.

    forward(x_up: (B,T,N_up,C_up), x_skip: (B,T,N_skip,C_skip))
        -> (B,T,N_skip,C_skip)
    where N_skip = 4 * N_up (2x spatial in each dim).
    """

    def __init__(self, in_dim: int, out_dim: int, spatial_resolution_in: int):
        """
        Args:
            in_dim: channel dim of the input from deeper level (C_up)
            out_dim: channel dim of the skip connection (C_skip), also output dim
            spatial_resolution_in: H=W of the input (before upsampling)
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.spatial_in = spatial_resolution_in

        # 2x spatial upsampling via transposed convolution
        self.upsample = nn.ConvTranspose2d(
            in_dim, out_dim, kernel_size=2, stride=2, bias=False,
        )

        # After concatenation with skip: 2*out_dim -> out_dim
        self.norm = nn.LayerNorm(2 * out_dim)
        self.proj = nn.Linear(2 * out_dim, out_dim, bias=False)

    def forward(
        self,
        x_up: torch.Tensor,
        x_skip: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_up: (B, T, N_up, C_up) from deeper level
            x_skip: (B, T, N_skip, C_skip) encoder skip connection

        Returns:
            (B, T, N_skip, C_skip)
        """
        B, T, N_up, C_up = x_up.shape
        H_in = W_in = self.spatial_in
        H_out = H_in * 2
        W_out = W_in * 2

        # Process each frame: reshape to spatial for ConvTranspose2d
        x_flat = x_up.reshape(B * T, H_in, W_in, C_up).permute(0, 3, 1, 2)
        x_flat = self.upsample(x_flat)  # (B*T, C_skip, H_out, W_out)
        x_flat = x_flat.permute(0, 2, 3, 1).reshape(B, T, H_out * W_out, self.out_dim)

        # Concatenate with skip connection
        x_cat = torch.cat([x_flat, x_skip], dim=-1)  # (B, T, N_skip, 2*C_skip)

        # Norm + project + residual
        x_out = self.proj(self.norm(x_cat))           # (B, T, N_skip, C_skip)
        x_out = x_out + x_skip                        # residual connection

        return x_out
