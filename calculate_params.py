"""Calculate tokenizer model parameters without loading the model (CPU-only analysis)."""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TokenizerConfig:
    # Voxel grid
    voxel_grid_xy: int = 1024
    voxel_grid_z: int = 64
    
    # Encoder
    voxel_feat_dim: int = 16
    bev_feat_dim: int = 64
    patch_size: int = 4
    enc_embed_dim: int = 128
    enc_stage1_depth: int = 2
    enc_stage1_heads: int = 8
    enc_stage2_dim: int = 256
    enc_stage2_depth: int = 6
    enc_stage2_heads: int = 16
    window_size: int = 8
    mlp_ratio: float = 4.0
    
    # VQ
    vq_dim: int = 256
    vq_codebook_size: int = 1024
    vq_codebook_dim: int = 1024
    
    # Decoder
    dec_stage1_depth: int = 6
    dec_stage1_heads: int = 8
    dec_stage2_depth: int = 2
    dec_stage2_heads: int = 8
    dec_output_dim: int = 128
    
    # NFG
    nfg_feat_dim: int = 16
    nfg_upsample_factor: int = 2
    nfg_z_bins: int = 64
    nfg_mlp_hidden: int = 32
    
    # Spatial skip
    skip_upsample_factor: int = 4
    
    @property
    def token_grid_size(self) -> int:
        return self.voxel_grid_xy // (self.patch_size * 2)
    
    @property
    def decoder_output_grid_size(self) -> int:
        return self.token_grid_size * 2
    
    @property
    def nfg_spatial_size(self) -> int:
        return self.decoder_output_grid_size * self.nfg_upsample_factor
    
    @property
    def skip_grid_xy(self) -> int:
        return self.decoder_output_grid_size * self.skip_upsample_factor


def count_linear(in_features, out_features, bias=True):
    """Count parameters in a Linear layer."""
    params = in_features * out_features
    if bias:
        params += out_features
    return params


def count_layernorm(normalized_shape):
    """Count parameters in LayerNorm."""
    if isinstance(normalized_shape, int):
        return 2 * normalized_shape
    return 2 * normalized_shape[0]


def count_conv2d(in_ch, out_ch, kernel, stride=1, bias=True):
    """Count parameters in Conv2d."""
    params = in_ch * out_ch * kernel * kernel
    if bias:
        params += out_ch
    return params


def count_convtranspose2d(in_ch, out_ch, kernel, stride=1, bias=True):
    """Count parameters in ConvTranspose2d."""
    params = in_ch * out_ch * kernel * kernel
    if bias:
        params += out_ch
    return params


def count_embedding(num_embeddings, embedding_dim):
    """Count parameters in Embedding."""
    return num_embeddings * embedding_dim


def count_window_attention(dim, num_heads, window_size):
    """Count parameters in WindowAttention."""
    # qkv: Linear(dim, dim*3)
    qkv = count_linear(dim, dim * 3)
    # proj: Linear(dim, dim)
    proj = count_linear(dim, dim)
    # relative_position_bias_table: (2*window_size-1)^2 * num_heads
    rel_pos_bias = (2 * window_size - 1) ** 2 * num_heads
    return qkv + proj + rel_pos_bias


def count_mlp(in_features, hidden_features, out_features):
    """Count parameters in MLP."""
    fc1 = count_linear(in_features, hidden_features)
    fc2 = count_linear(hidden_features, out_features)
    return fc1 + fc2


def count_swin_block(dim, num_heads, window_size, mlp_ratio):
    """Count parameters in SwinTransformerBlock."""
    # norm1: LayerNorm(dim)
    norm1 = count_layernorm(dim)
    # attn: WindowAttention
    attn = count_window_attention(dim, num_heads, window_size)
    # norm2: LayerNorm(dim)
    norm2 = count_layernorm(dim)
    # mlp: Mlp
    mlp_hidden = int(dim * mlp_ratio)
    mlp = count_mlp(dim, mlp_hidden, dim)
    return norm1 + attn + norm2 + mlp


def count_patch_merging(input_resolution, dim):
    """Count parameters in PatchMerging."""
    # reduction: Linear(4*dim, 2*dim, bias=False)
    reduction = count_linear(4 * dim, 2 * dim, bias=False)
    # norm: LayerNorm(4*dim)
    norm = count_layernorm(4 * dim)
    return reduction + norm


def count_patch_upsample(input_resolution, dim):
    """Count parameters in PatchUpsample."""
    # deconv: ConvTranspose2d(dim, dim, kernel=2, stride=2, bias=False)
    deconv = count_convtranspose2d(dim, dim, 2, bias=False)
    # norm: LayerNorm(dim)
    norm = count_layernorm(dim)
    # proj: Linear(dim, dim//2, bias=False)
    proj = count_linear(dim, dim // 2, bias=False)
    return deconv + norm + proj


def count_voxel_encoder(cfg):
    """Count parameters in DenseVoxelPointNet."""
    # MLP: Linear(4, 16) + LN(16) + Linear(16, 16)
    mlp = count_linear(4, 16) + count_layernorm(16) + count_linear(16, 16)
    # Final norm: LayerNorm(16)
    final_norm = count_layernorm(16)
    total = mlp + final_norm
    print(f"  DenseVoxelPointNet: {total:,}")
    return total


def count_bev_pooling(cfg):
    """Count parameters in BEVPillarPooling."""
    # z_embed: Embedding(64, 16)
    z_embed = count_embedding(cfg.voxel_grid_z, cfg.voxel_feat_dim)
    # mlp: Linear(32, 64)
    mlp = count_linear(cfg.voxel_feat_dim * 2, cfg.bev_feat_dim)
    # norm: LayerNorm(64)
    norm = count_layernorm(cfg.bev_feat_dim)
    total = z_embed + mlp + norm
    print(f"  BEVPillarPooling: {total:,}")
    return total


def count_swin_encoder(cfg):
    """Count parameters in SwinEncoder."""
    total = 0
    
    # PatchEmbed
    patch_res = cfg.voxel_grid_xy // cfg.patch_size  # 256
    # Conv2d(64, 128, kernel=4, stride=4)
    patch_embed_proj = count_conv2d(cfg.bev_feat_dim, cfg.enc_embed_dim, cfg.patch_size, cfg.patch_size)
    patch_embed_norm = count_layernorm(cfg.enc_embed_dim)
    patch_embed = patch_embed_proj + patch_embed_norm
    print(f"    PatchEmbed: {patch_embed:,}")
    
    # Positional embedding
    num_patches = patch_res * patch_res  # 256*256=65536
    pos_embed = num_patches * cfg.enc_embed_dim
    print(f"    Positional embedding: {pos_embed:,}")
    
    # Stage 1: depth=2, dim=128, heads=8
    stage1_blocks = 0
    for i in range(cfg.enc_stage1_depth):
        block_params = count_swin_block(cfg.enc_embed_dim, cfg.enc_stage1_heads, cfg.window_size, cfg.mlp_ratio)
        stage1_blocks += block_params
    print(f"    Stage 1 blocks ({cfg.enc_stage1_depth}x): {stage1_blocks:,}")
    
    # PatchMerging
    patch_merging = count_patch_merging((patch_res, patch_res), cfg.enc_embed_dim)
    print(f"    PatchMerging: {patch_merging:,}")
    
    # Stage 2: depth=6, dim=256, heads=16
    stage2_blocks = 0
    for i in range(cfg.enc_stage2_depth):
        block_params = count_swin_block(cfg.enc_stage2_dim, cfg.enc_stage2_heads, cfg.window_size, cfg.mlp_ratio)
        stage2_blocks += block_params
    print(f"    Stage 2 blocks ({cfg.enc_stage2_depth}x): {stage2_blocks:,}")
    
    total = patch_embed + pos_embed + stage1_blocks + patch_merging + stage2_blocks
    print(f"  SwinEncoder total: {total:,}")
    return total


def count_vq(cfg):
    """Count parameters in VectorQuantizer."""
    total = 0
    
    # pre_norm: LayerNorm(256)
    pre_norm = count_layernorm(cfg.vq_dim)
    print(f"    pre_norm: {pre_norm:,}")
    
    # pre_proj: Linear(256, 1024)
    pre_proj = count_linear(cfg.vq_dim, cfg.vq_codebook_dim)
    print(f"    pre_proj: {pre_proj:,}")
    
    # post_proj: Linear(1024, 256)
    post_proj = count_linear(cfg.vq_codebook_dim, cfg.vq_dim)
    print(f"    post_proj: {post_proj:,}")
    
    # embed (codebook): 1024 * 1024 - this is a buffer, not a parameter!
    # According to VQ implementation, embed is a buffer, not nn.Parameter
    # But codebook is typically trainable in VQ-VAE... let me check
    # Looking at the code: self.register_buffer("embed", ...) - it's a buffer!
    embed = 0  # Not counted as it's a buffer
    print(f"    embed (buffer, not param): {cfg.vq_codebook_size * cfg.vq_codebook_dim:,}")
    
    total = pre_norm + pre_proj + post_proj
    print(f"  VectorQuantizer total: {total:,}")
    return total


def count_swin_decoder(cfg):
    """Count parameters in SwinDecoder."""
    total = 0
    token_grid = cfg.token_grid_size  # 128
    
    # Stage 1: depth=6, dim=256, heads=8
    stage1_blocks = 0
    for i in range(cfg.dec_stage1_depth):
        block_params = count_swin_block(cfg.enc_stage2_dim, cfg.dec_stage1_heads, cfg.window_size, cfg.mlp_ratio)
        stage1_blocks += block_params
    print(f"    Stage 1 blocks ({cfg.dec_stage1_depth}x): {stage1_blocks:,}")
    
    # PatchUpsample
    patch_upsample = count_patch_upsample((token_grid, token_grid), cfg.enc_stage2_dim)
    print(f"    PatchUpsample: {patch_upsample:,}")
    
    # Stage 2: depth=2, dim=128, heads=8
    upsampled_res = token_grid * 2  # 256
    stage2_blocks = 0
    for i in range(cfg.dec_stage2_depth):
        block_params = count_swin_block(cfg.dec_output_dim, cfg.dec_stage2_heads, cfg.window_size, cfg.mlp_ratio)
        stage2_blocks += block_params
    print(f"    Stage 2 blocks ({cfg.dec_stage2_depth}x): {stage2_blocks:,}")
    
    # Final norm
    final_norm = count_layernorm(cfg.dec_output_dim)
    print(f"    Final norm: {final_norm:,}")
    
    total = stage1_blocks + patch_upsample + stage2_blocks + final_norm
    print(f"  SwinDecoder total: {total:,}")
    return total


def count_nfg(cfg):
    """Count parameters in NeuralFeatureGrid."""
    total = 0
    
    # nfg_head: LayerNorm(128) + Linear(128, uf*uf*Z*F)
    # uf=2, Z=64, F=16 -> 2*2*64*16 = 4096
    nfg_head_output = cfg.nfg_upsample_factor ** 2 * cfg.nfg_z_bins * cfg.nfg_feat_dim
    nfg_head = count_layernorm(cfg.dec_output_dim) + count_linear(cfg.dec_output_dim, nfg_head_output)
    print(f"    nfg_head: {nfg_head:,}")
    
    # occ_mlp: Linear(16, 32) + Linear(32, 1)
    occ_mlp = count_linear(cfg.nfg_feat_dim, cfg.nfg_mlp_hidden) + count_linear(cfg.nfg_mlp_hidden, 1)
    print(f"    occ_mlp: {occ_mlp:,}")
    
    total = nfg_head + occ_mlp
    print(f"  NeuralFeatureGrid total: {total:,}")
    return total


def count_spatial_skip(cfg):
    """Count parameters in SpatialSkipBranch."""
    # norm: LayerNorm(128)
    norm = count_layernorm(cfg.dec_output_dim)
    # linear: Linear(128, uf*uf*Z) = 128 * (4*4*64) = 128 * 1024 = 131072
    linear = count_linear(cfg.dec_output_dim, cfg.skip_upsample_factor ** 2 * cfg.voxel_grid_z)
    total = norm + linear
    print(f"  SpatialSkipBranch: {total:,}")
    return total


def main():
    cfg = TokenizerConfig()
    
    print("=" * 60)
    print("CoPilot4D Tokenizer Parameter Count")
    print("=" * 60)
    print()
    
    # Key dimensions
    print("Key dimensions:")
    print(f"  token_grid_size: {cfg.token_grid_size}")
    print(f"  decoder_output_grid_size: {cfg.decoder_output_grid_size}")
    print(f"  nfg_spatial_size: {cfg.nfg_spatial_size}")
    print(f"  skip_grid_xy: {cfg.skip_grid_xy}")
    print()
    
    total = 0
    
    print("1. Voxel Encoder:")
    total += count_voxel_encoder(cfg)
    print()
    
    print("2. BEV Pooling:")
    total += count_bev_pooling(cfg)
    print()
    
    print("3. Swin Encoder:")
    total += count_swin_encoder(cfg)
    print()
    
    print("4. Vector Quantizer:")
    total += count_vq(cfg)
    print()
    
    print("5. Swin Decoder:")
    total += count_swin_decoder(cfg)
    print()
    
    print("6. Neural Feature Grid:")
    total += count_nfg(cfg)
    print()
    
    print("7. Spatial Skip Branch:")
    total += count_spatial_skip(cfg)
    print()
    
    print("=" * 60)
    print(f"TOTAL PARAMETERS: {total:,}")
    print(f"                 = {total/1e6:.2f}M")
    print("=" * 60)
    
    # If codebook is trainable (not buffer), add it
    codebook_params = cfg.vq_codebook_size * cfg.vq_codebook_dim
    print()
    print("Note: Codebook (embed) is registered as buffer, not parameter.")
    print(f"      If made trainable: +{codebook_params:,} = {(total + codebook_params)/1e6:.2f}M")


if __name__ == "__main__":
    main()
