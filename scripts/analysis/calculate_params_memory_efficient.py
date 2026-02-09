"""Calculate tokenizer model parameters for memory_efficient config."""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TokenizerConfig:
    # Memory efficient config
    voxel_grid_xy: int = 512
    voxel_grid_z: int = 16
    
    voxel_feat_dim: int = 16
    bev_feat_dim: int = 32
    patch_size: int = 4
    enc_embed_dim: int = 64
    enc_stage1_depth: int = 2
    enc_stage1_heads: int = 4
    enc_stage2_dim: int = 128
    enc_stage2_depth: int = 4
    enc_stage2_heads: int = 8
    window_size: int = 8
    mlp_ratio: float = 4.0
    
    vq_dim: int = 128
    vq_codebook_size: int = 1024
    vq_codebook_dim: int = 512
    
    dec_stage1_depth: int = 4
    dec_stage1_heads: int = 8
    dec_stage2_depth: int = 2
    dec_stage2_heads: int = 4
    dec_output_dim: int = 64
    
    nfg_feat_dim: int = 16
    nfg_upsample_factor: int = 2
    nfg_z_bins: int = 16
    nfg_mlp_hidden: int = 32
    
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
    params = in_features * out_features
    if bias:
        params += out_features
    return params


def count_layernorm(normalized_shape):
    if isinstance(normalized_shape, int):
        return 2 * normalized_shape
    return 2 * normalized_shape[0]


def count_conv2d(in_ch, out_ch, kernel, stride=1, bias=True):
    params = in_ch * out_ch * kernel * kernel
    if bias:
        params += out_ch
    return params


def count_convtranspose2d(in_ch, out_ch, kernel, stride=1, bias=True):
    params = in_ch * out_ch * kernel * kernel
    if bias:
        params += out_ch
    return params


def count_embedding(num_embeddings, embedding_dim):
    return num_embeddings * embedding_dim


def count_window_attention(dim, num_heads, window_size):
    qkv = count_linear(dim, dim * 3)
    proj = count_linear(dim, dim)
    rel_pos_bias = (2 * window_size - 1) ** 2 * num_heads
    return qkv + proj + rel_pos_bias


def count_mlp(in_features, hidden_features, out_features):
    fc1 = count_linear(in_features, hidden_features)
    fc2 = count_linear(hidden_features, out_features)
    return fc1 + fc2


def count_swin_block(dim, num_heads, window_size, mlp_ratio):
    norm1 = count_layernorm(dim)
    attn = count_window_attention(dim, num_heads, window_size)
    norm2 = count_layernorm(dim)
    mlp_hidden = int(dim * mlp_ratio)
    mlp = count_mlp(dim, mlp_hidden, dim)
    return norm1 + attn + norm2 + mlp


def count_patch_merging(input_resolution, dim):
    reduction = count_linear(4 * dim, 2 * dim, bias=False)
    norm = count_layernorm(4 * dim)
    return reduction + norm


def count_patch_upsample(input_resolution, dim):
    deconv = count_convtranspose2d(dim, dim, 2, bias=False)
    norm = count_layernorm(dim)
    proj = count_linear(dim, dim // 2, bias=False)
    return deconv + norm + proj


def count_voxel_encoder(cfg):
    mlp = count_linear(4, 16) + count_layernorm(16) + count_linear(16, 16)
    final_norm = count_layernorm(16)
    total = mlp + final_norm
    print(f"  DenseVoxelPointNet: {total:,}")
    return total


def count_bev_pooling(cfg):
    z_embed = count_embedding(cfg.voxel_grid_z, cfg.voxel_feat_dim)
    mlp = count_linear(cfg.voxel_feat_dim * 2, cfg.bev_feat_dim)
    norm = count_layernorm(cfg.bev_feat_dim)
    total = z_embed + mlp + norm
    print(f"  BEVPillarPooling: {total:,}")
    return total


def count_swin_encoder(cfg):
    total = 0
    patch_res = cfg.voxel_grid_xy // cfg.patch_size
    
    patch_embed_proj = count_conv2d(cfg.bev_feat_dim, cfg.enc_embed_dim, cfg.patch_size, cfg.patch_size)
    patch_embed_norm = count_layernorm(cfg.enc_embed_dim)
    patch_embed = patch_embed_proj + patch_embed_norm
    print(f"    PatchEmbed: {patch_embed:,}")
    
    num_patches = patch_res * patch_res
    pos_embed = num_patches * cfg.enc_embed_dim
    print(f"    Positional embedding: {pos_embed:,}")
    
    stage1_blocks = 0
    for i in range(cfg.enc_stage1_depth):
        block_params = count_swin_block(cfg.enc_embed_dim, cfg.enc_stage1_heads, cfg.window_size, cfg.mlp_ratio)
        stage1_blocks += block_params
    print(f"    Stage 1 blocks ({cfg.enc_stage1_depth}x): {stage1_blocks:,}")
    
    patch_merging = count_patch_merging((patch_res, patch_res), cfg.enc_embed_dim)
    print(f"    PatchMerging: {patch_merging:,}")
    
    stage2_blocks = 0
    for i in range(cfg.enc_stage2_depth):
        block_params = count_swin_block(cfg.enc_stage2_dim, cfg.enc_stage2_heads, cfg.window_size, cfg.mlp_ratio)
        stage2_blocks += block_params
    print(f"    Stage 2 blocks ({cfg.enc_stage2_depth}x): {stage2_blocks:,}")
    
    total = patch_embed + pos_embed + stage1_blocks + patch_merging + stage2_blocks
    print(f"  SwinEncoder total: {total:,}")
    return total


def count_vq(cfg):
    total = 0
    pre_norm = count_layernorm(cfg.vq_dim)
    pre_proj = count_linear(cfg.vq_dim, cfg.vq_codebook_dim)
    post_proj = count_linear(cfg.vq_codebook_dim, cfg.vq_dim)
    
    print(f"    pre_norm: {pre_norm:,}")
    print(f"    pre_proj: {pre_proj:,}")
    print(f"    post_proj: {post_proj:,}")
    
    total = pre_norm + pre_proj + post_proj
    print(f"  VectorQuantizer total: {total:,}")
    return total


def count_swin_decoder(cfg):
    total = 0
    token_grid = cfg.token_grid_size
    
    stage1_blocks = 0
    for i in range(cfg.dec_stage1_depth):
        block_params = count_swin_block(cfg.enc_stage2_dim, cfg.dec_stage1_heads, cfg.window_size, cfg.mlp_ratio)
        stage1_blocks += block_params
    print(f"    Stage 1 blocks ({cfg.dec_stage1_depth}x): {stage1_blocks:,}")
    
    patch_upsample = count_patch_upsample((token_grid, token_grid), cfg.enc_stage2_dim)
    print(f"    PatchUpsample: {patch_upsample:,}")
    
    upsampled_res = token_grid * 2
    stage2_blocks = 0
    for i in range(cfg.dec_stage2_depth):
        block_params = count_swin_block(cfg.dec_output_dim, cfg.dec_stage2_heads, cfg.window_size, cfg.mlp_ratio)
        stage2_blocks += block_params
    print(f"    Stage 2 blocks ({cfg.dec_stage2_depth}x): {stage2_blocks:,}")
    
    final_norm = count_layernorm(cfg.dec_output_dim)
    print(f"    Final norm: {final_norm:,}")
    
    total = stage1_blocks + patch_upsample + stage2_blocks + final_norm
    print(f"  SwinDecoder total: {total:,}")
    return total


def count_nfg(cfg):
    total = 0
    nfg_head_output = cfg.nfg_upsample_factor ** 2 * cfg.nfg_z_bins * cfg.nfg_feat_dim
    nfg_head = count_layernorm(cfg.dec_output_dim) + count_linear(cfg.dec_output_dim, nfg_head_output)
    print(f"    nfg_head: {nfg_head:,}")
    
    occ_mlp = count_linear(cfg.nfg_feat_dim, cfg.nfg_mlp_hidden) + count_linear(cfg.nfg_mlp_hidden, 1)
    print(f"    occ_mlp: {occ_mlp:,}")
    
    total = nfg_head + occ_mlp
    print(f"  NeuralFeatureGrid total: {total:,}")
    return total


def count_spatial_skip(cfg):
    norm = count_layernorm(cfg.dec_output_dim)
    linear = count_linear(cfg.dec_output_dim, cfg.skip_upsample_factor ** 2 * cfg.voxel_grid_z)
    total = norm + linear
    print(f"  SpatialSkipBranch: {total:,}")
    return total


def main():
    cfg = TokenizerConfig()
    
    print("=" * 60)
    print("CoPilot4D Tokenizer (Memory Efficient Config)")
    print("=" * 60)
    print()
    
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
    
    codebook_params = cfg.vq_codebook_size * cfg.vq_codebook_dim
    print()
    print("Note: Codebook (embed) is registered as buffer, not parameter.")
    print(f"      If made trainable: +{codebook_params:,} = {(total + codebook_params)/1e6:.2f}M")


if __name__ == "__main__":
    main()
