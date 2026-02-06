"""Input embeddings for the world model.

Converts discrete tokens + spatial/temporal positions + actions into
continuous embeddings. Paper (A.2.2): "After the embedding layer, we
additionally apply Linear -> LayerNorm -> Linear."
"""

import torch
import torch.nn as nn
import math

from copilot4d.utils.config import WorldModelConfig


class WorldModelInputEmbedding(nn.Module):
    """Token + spatial + temporal + action embeddings.

    forward(token_indices: (B,T,N), actions: (B,T,16)) -> (B,T,N,C)
    where N = H*W = num_tokens_per_frame, C = level_dims[0].
    """

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        dim = cfg.level_dims[0]  # 256
        N = cfg.num_tokens_per_frame
        T_max = cfg.num_frames

        # Token embedding: vocab_size (1025) -> dim
        self.token_embedding = nn.Embedding(cfg.vocab_size, dim)

        # Post-embedding projection: Linear -> LN -> Linear (no bias)
        self.embed_proj = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim, bias=False),
        )

        # Spatial positional embedding: learnable, shared across all frames
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, N, dim))

        # Temporal positional embedding: learnable, broadcast to spatial
        self.temporal_pos = nn.Parameter(torch.zeros(1, T_max, 1, dim))

        # Action projection: Linear -> LN -> Linear (no bias)
        self.action_proj = nn.Sequential(
            nn.Linear(cfg.action_dim, dim, bias=False),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim, bias=False),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.token_embedding.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fan_in = m.weight.shape[1]
                std = math.sqrt(1.0 / (3.0 * fan_in))
                nn.init.normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        token_indices: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            token_indices: (B, T, N) long — discrete token indices
            actions: (B, T, action_dim) float — flattened SE(3)

        Returns:
            (B, T, N, C) float — embedded tokens
        """
        B, T, N = token_indices.shape

        # Token embedding + projection
        x = self.token_embedding(token_indices)  # (B, T, N, C)
        x = self.embed_proj(x)                   # (B, T, N, C)

        # Add spatial position (broadcast over B, T)
        x = x + self.spatial_pos[:, :, :N, :]

        # Add temporal position (broadcast over B, N)
        x = x + self.temporal_pos[:, :T, :, :]

        # Add action conditioning (broadcast to all spatial positions)
        act = self.action_proj(actions)           # (B, T, C)
        x = x + act.unsqueeze(2)                  # (B, T, N, C)

        return x
