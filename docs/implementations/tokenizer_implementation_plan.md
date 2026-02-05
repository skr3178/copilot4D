# CoPilot4D Tokenizer Implementation Plan

Implement the full VQVAE tokenizer: LiDAR point clouds -> discrete BEV tokens -> differentiable depth rendering.

## Files to Create (in dependency order)

### Layer 0: Config & Package Init
1. **`copilot4d/__init__.py`** -- empty
2. **`copilot4d/utils/__init__.py`** -- re-export TokenizerConfig
3. **`copilot4d/utils/config.py`** -- `TokenizerConfig` dataclass with all spatial dims as properties
4. **`copilot4d/data/__init__.py`** -- re-export KITTITokenizerDataset
5. **`copilot4d/tokenizer/__init__.py`** -- re-export CoPilot4DTokenizer

### Layer 1: Data Pipeline
6. **`copilot4d/data/point_cloud_utils.py`** -- `filter_roi()`, `voxelize_points()`, `generate_rays()`, `sample_training_rays()`
7. **`copilot4d/data/kitti_dataset.py`** -- `KITTITokenizerDataset(Dataset)`, `tokenizer_collate_fn()`

### Layer 2: Tokenizer Modules
8. **`copilot4d/tokenizer/voxel_encoder.py`** -- `VoxelPointNet`: MLP(4->16->16) + Sum + LN
9. **`copilot4d/tokenizer/bev_pooling.py`** -- `BEVPillarPooling`: z-embed + project + scatter_add
10. **`copilot4d/tokenizer/swin_transformer.py`** -- All Swin blocks adapted from reference: `WindowAttention`, `SwinTransformerBlock`, `PatchEmbed`, `PatchMerging`, `PatchUpsample` (new), `BasicLayer`, `SwinEncoder`, `SwinDecoder`
11. **`copilot4d/tokenizer/vector_quantizer.py`** -- `VectorQuantizer`: straight-through, EMA, K-Means reinit, FP32-forced
12. **`copilot4d/tokenizer/neural_feature_grid.py`** -- `NeuralFeatureGrid`: build NFG from decoder output, trilinear query via `F.grid_sample`, volume rendering with ray chunking
13. **`copilot4d/tokenizer/spatial_skipping.py`** -- `SpatialSkipBranch`: binary occupancy logits + GT computation
14. **`copilot4d/tokenizer/tokenizer_losses.py`** -- `l1_depth_loss`, `surface_concentration_loss`, `spatial_skip_bce_loss`, `tokenizer_total_loss`

### Layer 3: Assembly
15. **`copilot4d/tokenizer/tokenizer_model.py`** -- `CoPilot4DTokenizer`: full encode->VQ->decode pipeline with `forward()`, `encode()`, `decode()`

### Layer 4: Configs & Training
16. **`configs/tokenizer_debug.yaml`** -- 256x256x16 grid, 32x32 tokens
17. **`configs/tokenizer_prototype.yaml`** -- 512x512x32 grid, 64x64 tokens
18. **`scripts/train_tokenizer.py`** -- AdamW + cosine LR + AMP + grad accumulation + checkpointing

### Layer 5: Tests
19. **`tests/test_tokenizer_shapes.py`** -- shape verification for every module at debug & prototype configs

---

## Architecture Flow (shapes for prototype 512x512x32)

```
Input: KITTI scan (N, 4) [x,y,z,reflectance]
  |
  v  filter_roi + voxelize_points
Sparse voxels: coords (V, 3), features (V, 35, 4), num_pts (V,)
  |
  v  VoxelPointNet: MLP(4->16->16) + mask + sum + LN
Voxel features: (V, 16)
  |
  v  BEVPillarPooling: z_embed + project(32->64) + scatter_add + LN
BEV features: (B, 512, 512, 64)
  |
  v  permute to (B, 64, 512, 512)
  v  PatchEmbed(patch_size=4, 64->128)
  v  2 Swin blocks (dim=128, heads=8, win=8)
  v  PatchMerging (128->256, spatial /2)
  v  6 Swin blocks (dim=256, heads=16, win=8)
  v  LN
Encoder output: (B, 4096, 256)  [64x64 tokens]
  |
  v  LN -> GELU -> Linear(256, 1024)
  v  VectorQuantizer (1024 codes x 1024-dim, EMA, K-Means reinit)
  v  Linear(1024, 256)
Quantized: (B, 4096, 256), indices: (B, 64, 64)
  |
  v  6 Swin blocks (dim=256, heads=16, win=8)
  v  PatchUpsample (256->128, spatial x2)
  v  2 Swin blocks (dim=128, heads=8, win=8)
  v  LN
Decoder output: (B, 16384, 128)  [128x128 positions]
  |
  +---> NFG head: LN -> Linear(128, 2*2*32*16=2048) -> reshape
  |     NFG: (B, 256, 256, 32, 16)
  |       |
  |       v  trilinear query along rays (F.grid_sample 5D)
  |       v  MLP(16->32->1) -> sigmoid -> occupancy alpha
  |       v  volume rendering: w_i = alpha_i * cumprod(1-alpha_j)
  |       v  expected depth: D = sum(w_i * d_i)
  |     pred_depths: (B, R)
  |
  +---> Spatial skip: LN -> Linear(128, 4*4*32=512) -> reshape
        skip_logits: (B, 512, 512, 32)
```

## Key Config Dimensions

| Property | Debug | Prototype | Paper |
|----------|-------|-----------|-------|
| voxel_grid_xy | 256 | 512 | 1024 |
| voxel_grid_z | 16 | 32 | 64 |
| token grid | 32x32 | 64x64 | 128x128 |
| decoder output | 64x64 | 128x128 | 256x256 |
| NFG (H,W,Z,F) | 128x128x16x16 | 256x256x32x16 | 512x512x64x16 |
| rays/frame | 1024 | 2048 | 4096 |
| ray chunk | 256 | 256 | 1024 |
| batch | 8 | 4+accum4 | 16 |

## Critical Implementation Details

1. **Sparse BEV pooling**: scatter_add with flat index `batch*H*W + ix*W + iy`. Never materialize (H,W,Z) tensor.
2. **VQ in FP32**: Use `@torch.cuda.amp.custom_fwd(cast_to=torch.float32)`. Distance computation and EMA are numerically sensitive.
3. **Swin adapted from reference**: Port `WindowAttention`, `SwinTransformerBlock`, `PatchEmbed`, `PatchMerging`, `BasicLayer` from `reference_code/Swin-Transformer/models/swin_transformer.py`. Add new `PatchUpsample` (inverse of PatchMerging).
4. **NFG trilinear interp**: `F.grid_sample(nfg_5d, grid, mode='bilinear')` -- 'bilinear' on 5D = trilinear. NFG permuted to (B,C,D,H,W). Grid coords: (x->W, y->H, z->D) all in [-1,1].
5. **Ray chunking**: Process `ray_chunk_size` rays at a time through NFG query + volume rendering. Concat results.
6. **Volume rendering**: `weights = alpha * cumprod(1-alpha, exclusive)`, `depth = sum(weights * sample_depths)`.
7. **Gradient checkpointing**: Every Swin block via `use_checkpoint` flag.
8. **KITTI data**: pykitti at `data/kitti/pykitti/`. `get_velo(idx)` returns (N,4). Sequences 00-08 train, 09 val, 10 test. Sensor origin = (0,0,0) in Velodyne frame.

## Reference Code Locations
- Swin: `reference_code/Swin-Transformer/models/swin_transformer.py`
- VQ: `reference_code/vector-quantize-pytorch/vector_quantize_pytorch/vector_quantize_pytorch.py`
- NeRF volume rendering: `reference_code/nerf-pytorch/run_nerf.py`
- pykitti: `data/kitti/pykitti/pykitti/odometry.py`

## Loss Function
```
L = L_depth_l1 + L_surface_concentration + 1.0 * L_vq + L_skip_bce
```
- L_depth_l1 = |pred_depth - gt_depth|.mean()
- L_conc = (w_i^2 * [|d_i - d_gt| > epsilon]).sum() / (B*R)
- L_vq = commitment_cost * ||z_e - sg(z_q)||^2 (from VQ module)
- L_skip_bce = BCE_with_logits(skip_logits, gt_occupancy)

## Verification
1. Run `tests/test_tokenizer_shapes.py` -- all shape assertions pass for debug and prototype configs
2. Smoke test: `python scripts/train_tokenizer.py --config configs/tokenizer_debug.yaml` -- loss decreases for 100 iterations
3. Memory check: peak GPU < 4GB on debug, < 11.5GB on prototype
