"""Simple transformer model for discrete diffusion on Moving MNIST.

This is a simplified version of the CoPilot4D world model for testing
the discrete diffusion training on a smaller scale.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class TemporalAttention(nn.Module):
    """Temporal self-attention with configurable mask."""
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, N, D) - batch, time, tokens, dim
            mask: (T, T) attention mask or None
        """
        B, T, N, D = x.shape
        
        # Reshape for attention: (B*N, T, D)
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        
        qkv = self.qkv(x).reshape(B * N, T, 3, self.num_heads, D // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*N, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B*N, heads, T, T)
        
        # Apply mask if provided
        if mask is not None:
            attn = attn + mask[None, None, :, :]
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B * N, T, D)
        out = self.proj(out)
        
        # Reshape back: (B, T, N, D)
        out = out.reshape(B, N, T, D).permute(0, 2, 1, 3)
        return out


class SpatialAttention(nn.Module):
    """Spatial self-attention (within each frame)."""
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (B, T, N, D)
        """
        B, T, N, D = x.shape
        
        # Reshape: (B*T, N, D)
        x = x.reshape(B * T, N, D)
        
        qkv = self.qkv(x).reshape(B * T, N, 3, self.num_heads, D // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B * T, N, D)
        out = self.proj(out)
        
        # Reshape back
        out = out.reshape(B, T, N, D)
        return out


class TransformerBlock(nn.Module):
    """Combined spatial + temporal transformer block."""
    
    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.spatial_attn = SpatialAttention(dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        self.temporal_attn = TemporalAttention(dim, num_heads, dropout)
        
        self.norm3 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x, temporal_mask=None):
        """
        Args:
            x: (B, T, N, D)
            temporal_mask: (T, T) or None
        """
        # Spatial attention
        x = x + self.spatial_attn(self.norm1(x))
        
        # Temporal attention
        x = x + self.temporal_attn(self.norm2(x), temporal_mask)
        
        # MLP
        x = x + self.mlp(self.norm3(x))
        
        return x


class SimpleVideoTransformer(nn.Module):
    """Simple transformer for discrete video diffusion on Moving MNIST.
    
    Args:
        vocab_size: Number of discrete token values (e.g., 16 for 4-bit)
        mask_token_id: Special token ID for masked positions
        num_frames: Number of frames in sequence
        height: Frame height
        width: Frame width
        embed_dim: Embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        action_dim: Dimension of action vectors
    """
    
    def __init__(
        self,
        vocab_size=16,
        mask_token_id=16,  # vocab_size is the mask token
        num_frames=20,
        height=64,
        width=64,
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        action_dim=4,
        dropout=0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.num_tokens = height * width
        
        # Token embedding (including mask token)
        self.token_embed = nn.Embedding(vocab_size + 1, embed_dim)
        
        # Action embedding (project action vectors to embed_dim)
        self.action_proj = nn.Linear(action_dim, embed_dim)
        
        # Positional embeddings
        self.temporal_pos = SinusoidalPosEmb(embed_dim)
        # Learnable spatial positions
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, self.num_tokens, embed_dim) * 0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output projection (weight-tied with input)
        self.output_proj = nn.Linear(embed_dim, vocab_size + 1, bias=False)
        # Tie weights
        self.output_proj.weight = self.token_embed.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def forward(self, token_indices, actions, temporal_mask=None):
        """
        Args:
            token_indices: (B, T, H, W) discrete token indices
            actions: (B, T, action_dim) action vectors
            temporal_mask: (T, T) attention mask or None
            
        Returns:
            logits: (B, T, H*W, vocab_size+1) token logits
        """
        B, T, H, W = token_indices.shape
        
        # Embed tokens: (B, T, H*W, D)
        tokens = token_indices.reshape(B, T, -1)
        x = self.token_embed(tokens)  # (B, T, N, D)
        
        # Add action embeddings
        # Project actions and add to each token
        action_emb = self.action_proj(actions)  # (B, T, D)
        x = x + action_emb[:, :, None, :]
        
        # Add positional embeddings
        # Temporal: add same embedding to all spatial positions
        t_pos = self.temporal_pos(torch.arange(T, device=x.device))  # (T, D)
        x = x + t_pos[None, :, None, :]
        
        # Spatial: add to all temporal frames
        x = x + self.spatial_pos
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, temporal_mask)
        
        # Output
        x = self.norm(x)
        logits = self.output_proj(x)  # (B, T, N, vocab_size+1)
        
        return logits


if __name__ == "__main__":
    # Test the model
    print("Testing SimpleVideoTransformer...")
    
    model = SimpleVideoTransformer(
        vocab_size=16,
        mask_token_id=16,
        num_frames=20,
        height=64,
        width=64,
        embed_dim=256,
        num_layers=4,
        num_heads=8,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    B, T, H, W = 2, 20, 64, 64
    tokens = torch.randint(0, 17, (B, T, H, W))
    actions = torch.randn(B, T, 4)
    
    # Test with causal mask
    causal_mask = torch.full((T, T), float("-inf"))
    causal_mask = torch.triu(causal_mask, diagonal=1)
    
    logits = model(tokens, actions, temporal_mask=causal_mask)
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected: ({B}, {T}, {H*W}, 17)")
    
    # Test with identity mask
    identity_mask = torch.full((T, T), float("-inf"))
    identity_mask.fill_diagonal_(0.0)
    
    logits_id = model(tokens, actions, temporal_mask=identity_mask)
    print(f"Identity mask output shape: {logits_id.shape}")
    
    print("\nSimpleVideoTransformer test passed!")
