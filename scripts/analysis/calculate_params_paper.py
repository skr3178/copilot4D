"""Calculate tokenizer model parameters - checking paper config vs actual."""

from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    # Default config from config.py (matches paper spec)
    voxel_grid_xy: int = 1024
    voxel_grid_z: int = 64
    
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
    
    vq_dim: int = 256
    vq_codebook_size: int = 1024
    vq_codebook_dim: int = 1024
    
    dec_stage1_depth: int = 6
    dec_stage1_heads: int = 8
    dec_stage2_depth: int = 2
    dec_stage2_heads: int = 8
    dec_output_dim: int = 128
    
    nfg_feat_dim: int = 16
    nfg_upsample_factor: int = 2
    nfg_z_bins: int = 64
    nfg_mlp_hidden: int = 32
    
    skip_upsample_factor: int = 4
    
    @property
    def token_grid_size(self) -> int:
        return self.voxel_grid_xy // (self.patch_size * 2)
    
    @property
    def decoder_output_grid_size(self) -> int:
        return self.token_grid_size * 2


def count_linear(in_f, out_f, bias=True):
    return in_f * out_f + (out_f if bias else 0)


def count_layernorm(n):
    return 2 * n


def count_conv2d(in_c, out_c, k, bias=True):
    return in_c * out_c * k * k + (out_c if bias else 0)


def count_convtranspose2d(in_c, out_c, k, bias=True):
    return in_c * out_c * k * k + (out_c if bias else 0)


def count_embedding(n, dim):
    return n * dim


def count_window_attn(dim, heads, win):
    qkv = count_linear(dim, dim * 3)
    proj = count_linear(dim, dim)
    rel_pos = (2 * win - 1) ** 2 * heads
    return qkv + proj + rel_pos


def count_mlp(in_f, hid_f, out_f):
    return count_linear(in_f, hid_f) + count_linear(hid_f, out_f)


def count_swin_block(dim, heads, win, mlp_ratio):
    norm1 = count_layernorm(dim)
    attn = count_window_attn(dim, heads, win)
    norm2 = count_layernorm(dim)
    mlp = count_mlp(dim, int(dim * mlp_ratio), dim)
    return norm1 + attn + norm2 + mlp


def count_patch_merging(dim):
    return count_linear(4 * dim, 2 * dim, bias=False) + count_layernorm(4 * dim)


def count_patch_upsample(dim):
    return (count_convtranspose2d(dim, dim, 2, bias=False) + 
            count_layernorm(dim) + 
            count_linear(dim, dim // 2, bias=False))


def main():
    cfg = TokenizerConfig()
    
    print("=" * 70)
    print("CoPilot4D Tokenizer Parameter Analysis (Paper Config)")
    print("=" * 70)
    print(f"\nToken grid size: {cfg.token_grid_size}x{cfg.token_grid_size}")
    print(f"Decoder output grid: {cfg.decoder_output_grid_size}x{cfg.decoder_output_grid_size}")
    
    # 1. Voxel Encoder
    voxel_enc = count_linear(4, 16) + count_layernorm(16) + count_linear(16, 16) + count_layernorm(16)
    print(f"\n1. Voxel Encoder:           {voxel_enc:>12,}")
    
    # 2. BEV Pooling
    bev_pool = (count_embedding(cfg.voxel_grid_z, cfg.voxel_feat_dim) + 
                count_linear(cfg.voxel_feat_dim * 2, cfg.bev_feat_dim) + 
                count_layernorm(cfg.bev_feat_dim))
    print(f"2. BEV Pooling:             {bev_pool:>12,}")
    
    # 3. Swin Encoder
    patch_res = cfg.voxel_grid_xy // cfg.patch_size  # 256
    num_patches = patch_res * patch_res  # 65536
    
    patch_embed = count_conv2d(cfg.bev_feat_dim, cfg.enc_embed_dim, cfg.patch_size) + count_layernorm(cfg.enc_embed_dim)
    pos_embed = num_patches * cfg.enc_embed_dim
    
    stage1 = sum(count_swin_block(cfg.enc_embed_dim, cfg.enc_stage1_heads, cfg.window_size, cfg.mlp_ratio) 
                 for _ in range(cfg.enc_stage1_depth))
    merging = count_patch_merging(cfg.enc_embed_dim)
    stage2 = sum(count_swin_block(cfg.enc_stage2_dim, cfg.enc_stage2_heads, cfg.window_size, cfg.mlp_ratio) 
                 for _ in range(cfg.enc_stage2_depth))
    
    swin_enc = patch_embed + pos_embed + stage1 + merging + stage2
    print(f"\n3. Swin Encoder:")
    print(f"   - PatchEmbed:            {patch_embed:>12,}")
    print(f"   - Positional embed:      {pos_embed:>12,}")
    print(f"   - Stage 1 (2 blocks):    {stage1:>12,}")
    print(f"   - PatchMerging:          {merging:>12,}")
    print(f"   - Stage 2 (6 blocks):    {stage2:>12,}")
    print(f"   Subtotal:                {swin_enc:>12,}")
    
    # 4. VQ
    vq = (count_layernorm(cfg.vq_dim) + 
          count_linear(cfg.vq_dim, cfg.vq_codebook_dim) + 
          count_linear(cfg.vq_codebook_dim, cfg.vq_dim))
    codebook = cfg.vq_codebook_size * cfg.vq_codebook_dim  # Buffer, not param
    print(f"\n4. Vector Quantizer:        {vq:>12,}")
    print(f"   (Codebook as buffer:     {codebook:>12,})")
    
    # 5. Swin Decoder
    dec_stage1 = sum(count_swin_block(cfg.enc_stage2_dim, cfg.dec_stage1_heads, cfg.window_size, cfg.mlp_ratio) 
                     for _ in range(cfg.dec_stage1_depth))
    upsample = count_patch_upsample(cfg.enc_stage2_dim)
    dec_stage2 = sum(count_swin_block(cfg.dec_output_dim, cfg.dec_stage2_heads, cfg.window_size, cfg.mlp_ratio) 
                     for _ in range(cfg.dec_stage2_depth))
    dec_norm = count_layernorm(cfg.dec_output_dim)
    
    swin_dec = dec_stage1 + upsample + dec_stage2 + dec_norm
    print(f"\n5. Swin Decoder:")
    print(f"   - Stage 1 (6 blocks):    {dec_stage1:>12,}")
    print(f"   - PatchUpsample:         {upsample:>12,}")
    print(f"   - Stage 2 (2 blocks):    {dec_stage2:>12,}")
    print(f"   - Final norm:            {dec_norm:>12,}")
    print(f"   Subtotal:                {swin_dec:>12,}")
    
    # 6. NFG
    nfg_head_out = cfg.nfg_upsample_factor ** 2 * cfg.nfg_z_bins * cfg.nfg_feat_dim
    nfg = (count_layernorm(cfg.dec_output_dim) + 
           count_linear(cfg.dec_output_dim, nfg_head_out) + 
           count_linear(cfg.nfg_feat_dim, cfg.nfg_mlp_hidden) + 
           count_linear(cfg.nfg_mlp_hidden, 1))
    print(f"\n6. Neural Feature Grid:     {nfg:>12,}")
    
    # 7. Spatial Skip
    skip_out = cfg.skip_upsample_factor ** 2 * cfg.voxel_grid_z
    spatial_skip = count_layernorm(cfg.dec_output_dim) + count_linear(cfg.dec_output_dim, skip_out)
    print(f"7. Spatial Skip Branch:     {spatial_skip:>12,}")
    
    # Totals
    total_trainable = voxel_enc + bev_pool + swin_enc + vq + swin_dec + nfg + spatial_skip
    total_with_codebook = total_trainable + codebook
    
    print(f"\n{'='*70}")
    print(f"TOTAL TRAINABLE PARAMS:     {total_trainable:>12,} = {total_trainable/1e6:.2f}M")
    print(f"With codebook (if trainable):{total_with_codebook:>12,} = {total_with_codebook/1e6:.2f}M")
    print(f"{'='*70}")
    
    # What if we don't count positional embedding?
    without_pos_embed = total_trainable - pos_embed
    print(f"\nWithout pos_embed:          {without_pos_embed:>12,} = {without_pos_embed/1e6:.2f}M")
    
    # What config gives 13M?
    print(f"\n{'='*70}")
    print("Analysis: What would give ~13M parameters?")
    print(f"{'='*70}")
    print(f"Current with default config: ~{total_trainable/1e6:.1f}M")
    print(f"\nMajor components:")
    print(f"  - Positional embedding:   {pos_embed/1e6:.1f}M ({pos_embed/total_trainable*100:.1f}%)")
    print(f"  - SwinEncoder blocks:     {(stage1+stage2+merging)/1e6:.1f}M")
    print(f"  - SwinDecoder blocks:     {(dec_stage1+dec_stage2)/1e6:.1f}M")
    print(f"\nTo get 13M, possible changes:")
    print(f"  - Remove pos_embed:       {without_pos_embed/1e6:.1f}M")
    print(f"  - Or use 512x512 grid:    (would reduce pos_embed by 4x)")


if __name__ == "__main__":
    main()
