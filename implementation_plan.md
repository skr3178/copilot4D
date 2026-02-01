# CoPilot4D Implementation Plan

## Overview

Full PyTorch reproduction of CoPilot4D (ICLR 2024): an unsupervised world model that tokenizes LiDAR point clouds into discrete BEV tokens via VQVAE, then predicts future frames via discrete diffusion.

**Two-stage pipeline:**
1. **Tokenizer** (13M params): VQVAE with differentiable depth rendering
2. **World Model** (39M params): Spatio-temporal Transformer with discrete diffusion

## GPU Adaptation (RTX 3060 12GB)

The paper's full 1024x1024 voxel grid requires ~8GB just for the 3D feature volume. We must use a reduced resolution for prototyping:

| Parameter | Paper | Prototype |
|-----------|-------|-----------|
| Spatial range | [-80m, 80m]^2 | [-40m, 40m]^2 |
| Voxel grid (XY) | 1024x1024 | 512x512 |
| Voxel grid (Z) | 64 | 32 |
| BEV tokens | 128x128 (16384) | 64x64 (4096) |
| NFG resolution | 512x512x64 | 256x256x32 |
| Tokenizer batch | 16 | 4 + grad accum |
| World model batch | 8 | 1-2 + grad accum |

All spatial dimensions will be configurable so we can scale up on a larger GPU. Mixed precision (FP16) and gradient checkpointing are mandatory.

## Project Structure

```
CoPilot4D/
├── configs/
│   ├── default.yaml
│   ├── tokenizer_nuscenes.yaml
│   ├── tokenizer_kitti.yaml
│   ├── worldmodel_nuscenes_1s.yaml
│   └── prototype.yaml            # Reduced resolution for 12GB
├── copilot4d/
│   ├── data/
│   │   ├── nuscenes_dataset.py   # NuScenes data loading & sequences
│   │   ├── kitti_dataset.py      # KITTI Odometry data loading
│   │   └── point_cloud_utils.py  # Voxelization, transforms, ray generation
│   ├── tokenizer/
│   │   ├── voxel_encoder.py      # PointNet-like voxel encoder
│   │   ├── bev_pooling.py        # 3D-to-BEV pillar pooling (sparse scatter)
│   │   ├── swin_transformer.py   # Swin blocks, PatchEmbed/Merge/Upsample
│   │   ├── vector_quantizer.py   # VQ with K-Means codebook reinit
│   │   ├── neural_feature_grid.py # NFG + differentiable depth rendering
│   │   ├── spatial_skipping.py   # Coarse binary voxel branch
│   │   ├── tokenizer_model.py    # Full encoder-VQ-decoder assembly
│   │   └── tokenizer_losses.py   # L_vq + L_render + BCE
│   ├── world_model/
│   │   ├── gpt2_block.py         # Temporal GPT-2 attention block
│   │   ├── spatio_temporal_block.py  # 2 Swin + 1 GPT2 combined
│   │   ├── unet_transformer.py   # 3-level U-Net with ST blocks
│   │   ├── embeddings.py         # Token, action, spatial/temporal pos enc
│   │   ├── discrete_diffusion.py # Algorithms 1 & 2, CFG, mask schedule
│   │   └── world_model.py        # Full world model with training objectives
│   ├── evaluation/
│   │   ├── metrics.py            # Chamfer distance, L1 depth, AbsRel
│   │   └── visualize.py          # BEV point cloud rendering
│   └── utils/
│       ├── config.py, checkpoint.py, lr_scheduler.py
├── scripts/
│   ├── preprocess_nuscenes.py    # Cache scene metadata, poses, paths
│   ├── train_tokenizer.py
│   ├── tokenize_dataset.py       # Pre-tokenize all frames
│   ├── train_world_model.py
│   └── evaluate.py
├── tests/                        # Unit + integration tests per module
└── requirements.txt
```

## Implementation Phases

### Phase 0: Environment & Data Pipeline

**Files:** `requirements.txt`, `configs/`, `copilot4d/data/`, `scripts/preprocess_nuscenes.py`

1. Set up environment: `torch>=2.0`, `einops`, `timm` (reference), `nuscenes-devkit`, `pykitti`, `open3d`
2. Implement `point_cloud_utils.py`:
   - `points_to_voxels()` - sparse voxelization (use floor + dictionary grouping, or torch_scatter)
   - `transform_points()` - apply SE(3) transforms
   - `compute_relative_pose()` - relative poses between frames
   - `generate_rays()` - ray origins/directions/depths from point clouds
   - **Critical**: never materialize a dense (1024,1024,64) tensor. Use sparse coords + scatter.
3. Implement `nuscenes_dataset.py`:
   - `NuScenesTokenizerDataset` - single frames: voxels, coords, num_points, rays, depths, sensor_origin
   - `NuScenesWorldModelDataset` - temporal sequences: token indices (pre-tokenized), relative poses
   - Load via nuscenes-devkit API: scene -> sample chain -> sample_data[LIDAR_TOP] -> ego_pose + calibrated_sensor
   - Point clouds are `.pcd.bin` files with (x, y, z, intensity, ring_index), use only xyz
4. Implement `kitti_dataset.py` (similar but simpler, for when download completes)
5. `preprocess_nuscenes.py` - cache scene lists, ego pose matrices, file paths to avoid repeated JSON parsing

**Key files to reference:**
- `/media/skr/storage/self_driving/CoPilot4D/nuscenes-devkit/python-sdk/nuscenes/nuscenes.py`
- `/media/skr/storage/self_driving/CoPilot4D/Data/NuScenes/v1.0-mini/` (metadata JSONs)
- `/media/skr/storage/self_driving/CoPilot4D/Data/NuScenes/samples/LIDAR_TOP/` (404 scans)

### Phase 1: Tokenizer (13M params)

**Architecture (paper Fig 6, Appendix A.2.1):**

**Encoder:**
1. `VoxelPointNet` - per-voxel: point offsets -> MLP(3->16->16) -> Sum+LN -> (M, 16)
2. `BEVPillarPooling` - concat z-embed, MLP to 64-dim, scatter-add across Z, LN -> (H, W, 64)
3. `PatchEmbed(4)` -> 2 Swin blocks (dim=128, heads=8, win=8) -> `PatchMerging` -> 6 Swin blocks (dim=256, heads=16, win=8)
4. `LN -> GELU -> Linear(256, 1024)` -> `VectorQuantizer(1024 codes, dim=1024)`

**VQ details:** Straight-through estimator. K-Means reinit: memory bank = 10x codebook size, dead code = unused for 256 iters, reinit if >3% dead, min 200 iters between reinits. Loss: lambda1=0.25, lambda2=1.0.

**Decoder:**
1. 6 Swin blocks (dim=256) -> `PatchUpsample` -> 2 Swin blocks (dim=128) -> (256, 256, 128) output
2. **Branch 1 (NFG):** `LN -> Linear(128, 4*64*16)` -> reshape (512, 512, 64, 16). Query via trilinear interp + MLP(16->32->1) + sigmoid -> occupancy. Depth rendering: w_i = alpha_i * prod(1-alpha_j), D = sum(w_i * h_i). Surface concentration loss with epsilon=0.4.
3. **Branch 2 (Spatial skip):** `LN -> Linear(128, 16*64)` -> (1024, 1024, 64) binary logits. Bias init=-5.0. BCE loss. Max-pool for inference skipping.

**Training:** AdamW (beta2=0.95, wd=0.0001), LR=0.001, warmup=4000, cosine decay 400K iters, grad clip=0.1, batch=16 (effective via accumulation). Sample ~4096 rays/frame, ~64 points/ray. Process rays in chunks for memory.

### Phase 1.5: Pre-tokenize Dataset

Run trained tokenizer on all frames, save 64x64 (or 128x128) integer token grids. NuScenes mini: 404 frames * 8KB = ~3MB total.

### Phase 2: World Model (39M params)

**Architecture (paper Fig 7, Appendix A.2.2):**

`SpatioTemporalBlock` = 2 Swin spatial blocks + 1 GPT-2 temporal block

**U-Net layout:**
- Level 1 down (128x128, dim=256, heads=8, win=8): 2 ST blocks
- Level 2 down (64x64, dim=384, heads=12, win=8): 2 ST blocks
- Level 3 down (32x32, dim=512, heads=16, win=16): 1 ST block (2 Swin + 1 temporal)
- Level Merging up to Level 2: 1 ST block
- Level Merging up to Level 1: 2 ST blocks
- Final: LN -> weight-tied Linear -> (B, T, H, W, 1025) logits

**Inputs (all summed at network entry):**
- Token embeddings (dim=256, codebook=1024 + 1 mask token), weight-tied with output
- Action embeddings: flatten 4x4 pose -> Linear(16,D) -> LN -> Linear(D,D)
- ViT-style spatial positional encoding (shared across time)
- Learnable temporal positional encoding (shared across space)
- Post-embedding: Linear -> LN -> Linear

**Initialization (A.3):** Fan-in Normal(0, sqrt(1/(3H))). Residual scaling: sqrt(1/L) per level. No bias in Linear (except Swin QKV).

**Discrete Diffusion (Algorithms 1 & 2):**

Training (Alg 1): mask cos(u*pi/2)*N tokens, noise up to 20% of remaining with random codebook indices, cross-entropy to reconstruct x0.

Sampling (Alg 2): K=10 steps. Start all-mask. Each step: predict x0 with top-3 sampling, compute confidence = log_prob + Gumbel*k/K, keep non-mask indices from previous step (+inf confidence), unmask top-M by confidence where M=ceil(cos(k/K*pi/2)*N).

**Training Objectives (mixed per iteration):**
1. 50%: past=ground truth, future=masked+noised, causal temporal mask
2. 40%: all frames masked+noised, causal temporal mask
3. 10%: each frame independently masked+noised, identity temporal mask (for CFG)

**CFG at inference (Fig 10, Eq 2):** Efficient single forward pass: append duplicate target frame with identity temporal mask. logits_cfg = logits_cond + w*(logits_cond - logits_uncond). w=1.0 or 2.0.

**Training:** AdamW (beta2=0.95, wd=0.0001), LR=0.001, warmup=2000, cosine decay 750K iters, grad clip=5.0, batch=8 (effective), label smoothing=0.1.

### Phase 3: Evaluation

**Metrics:** Chamfer distance, L1 depth (mean/median), AbsRel (mean/median), all within ROI [-70m,70m]^2 x [-4.5m,4.5m].

**Protocol:**
- NuScenes 2Hz: 1s = 2 past -> 2 future; 3s = 6 past -> 6 future
- KITTI 10Hz: 5 past -> 5 future (subsample for 3s)
- Autoregressively predict each future frame with K=10 diffusion steps + CFG

## Key Design Decisions

1. **Sparse voxelization**: Never materialize dense 3D volumes. Use sparse coords + scatter_add for BEV pooling.
2. **Ray chunking**: Process rays in batches of 1024 during training to manage memory.
3. **Gradient checkpointing**: Apply to every Swin and GPT-2 block.
4. **VQ in FP32**: Keep codebook operations in full precision for stability.
5. **PointNet dim**: MLP outputs dim=16, BEV pooling outputs dim=64 per pixel.
6. **NFG interpolation**: Use trilinear (PyTorch grid_sample on 5D input with bilinear mode).
7. **Rays per frame**: 4096 training rays, 64 sample points per ray (configurable).
8. **Top-K sampling**: Per-location top-3 logit sampling during diffusion inference.
9. **All dimensions configurable** via YAML config to allow scaling between prototype and full resolution.

## Verification Plan

1. **Unit tests per module**: Shape checks, gradient flow, known-input/output for voxelization, VQ, depth rendering, diffusion masking
2. **Smoke test**: Train tokenizer 100 iters on NuScenes mini -> loss decreases
3. **Reconstruction check**: Encode + decode a point cloud -> visually inspect BEV rendering
4. **World model smoke test**: Train 100 iters on pre-tokenized mini data -> loss decreases
5. **End-to-end**: Tokenize past frames -> predict 1 future -> decode -> compute Chamfer distance
6. **Memory profiling**: Verify peak GPU < 11GB on prototype resolution before long runs
7. **Full training on KITTI** (once download completes) for meaningful quantitative evaluation




```markdown
# CoPilot4D Implementation Plan (KITTI-Optimized)

## Overview
Full PyTorch reproduction of CoPilot4D (ICLR 2024): unsupervised world model that tokenizes LiDAR point clouds into discrete BEV tokens via VQVAE, then predicts future frames via discrete diffusion.

**Two-stage pipeline:**
- Tokenizer (13M params): VQVAE with differentiable depth rendering
- World Model (39M params): Spatio-temporal Transformer with discrete diffusion

**Primary dataset:** KITTI Odometry (80GB, already downloaded)  
*Advantages over NuScenes Mini: 23k frames (vs 404), continuous sequences, simpler pose format, no download wait*

---

## GPU Adaptation (RTX 3060 12GB)

| Parameter               | Paper (Full) | Prototype (KITTI) | Debug (Initial Validation) |
|-------------------------|--------------|-------------------|----------------------------|
| Spatial range (XY)      | [-80m, 80m]² | **[-40m, 40m]²**  | [-40m, 40m]²               |
| Voxel grid (XY)         | 1024×1024    | **512×512**       | 256×256                    |
| Voxel grid (Z)          | 64           | **32**            | 16                         |
| BEV tokens              | 128×128      | **64×64**         | 32×32                      |
| Effective resolution    | 0.156m/voxel | **0.156m/voxel**  | 0.312m/voxel               |
| NFG resolution          | 512×512×64   | **256×256×32**    | 128×128×16                 |
| Tokenizer batch         | 16           | **4 + grad accum**| 8                          |
| World model batch       | 8            | **1-2 + grad accum**| 4                        |
| Ray chunks              | 4096         | **256**           | 512                        |

> 💡 **Critical insight:** KITTI scenes are narrower (±40m typical) → same 0.156m resolution as paper but with 8× fewer voxels. Z-resolution halved (64→32) → acceptable tradeoff for driving tasks (lateral precision > vertical).

**Mandatory optimizations:**
- Gradient checkpointing on **every Swin/GPT-2 block**
- FP16 mixed precision + VQ operations in FP32
- Sparse voxelization (never materialize dense 3D tensors)
- Ray chunking ≤256 rays during NFG training

---

## Project Structure
```
CoPilot4D/
├── configs/
│   ├── default.yaml
│   ├── tokenizer_kitti.yaml        # PRIMARY CONFIG (512×512×32)
│   ├── tokenizer_kitti_debug.yaml  # Debug config (256×256×16)
│   ├── worldmodel_kitti_0.5s.yaml  # 5 past → 5 future @10Hz
│   └── prototype_kitti.yaml        # Unified prototype settings
├── copilot4d/
│   ├── data/
│   │   ├── kitti_dataset.py        # IMPLEMENT FIRST (simpler than NuScenes)
│   │   ├── nuscenes_dataset.py     # Optional later
│   │   └── point_cloud_utils.py    # Voxelization + KITTI-specific augmentations
│   ├── tokenizer/
│   │   ├── ... (as original)
│   │   └── tokenizer_losses.py     # Add augmentation hooks
│   ├── world_model/
│   │   └── ... (as original)
│   ├── evaluation/
│   │   └── metrics.py              # KITTI ROI: [-35,35]² × [-3.0,1.0]
│   └── utils/
│       └── wandb_logger.py         # NEW: W&B integration
├── scripts/
│   ├── preprocess_kitti.py         # RUN FIRST: cache poses/frame indices
│   ├── train_tokenizer.py          # --config configs/tokenizer_kitti_debug.yaml
│   ├── tokenize_dataset.py
│   ├── train_world_model.py
│   └── evaluate.py
├── tests/
└── requirements.txt                # Add: wandb, pykitti
```

---

## Implementation Phases (Revised Timeline: 5–6 Weeks)

### Phase 0: Environment & KITTI Data Pipeline (4–5 days)
**Priority:** Implement KITTI loader first (no download wait)

**Critical additions vs original plan:**
```python
# copilot4d/data/kitti_dataset.py
class KITTITokenizerDataset(Dataset):
    def __init__(self, root="/media/skr/storage/self_driving/CoPilot4D/Data/KITTI",
                 augment=True, val_split=0.05):
        # Load sequences 00-10 (train), 11-21 (val/test)
        # Augmentations: random_rotation_2d(±π), random_translation(±2m), 
        #                random_dropout(0.05-0.2), intensity jitter
    
    def __getitem__(self, idx):
        # Returns: points (M,3) in sensor frame, global_pose (4×4), seq_id, frame_id

# copilot4d/data/point_cloud_utils.py
def kitti_points_to_voxels(points, range_xy=40.0, range_z=3.0, voxel_size=0.156):
    # Clip Z to [-3.0m, 1.0m] (road-focused)
    # Apply vertical angle weighting: weights = cos(atan2(z, sqrt(x²+y²)))²
    # Return sparse coords + features (offsets + reflectance + weights)
```

**Tasks:**
1. Run `preprocess_kitti.py` to cache poses/frame counts (1 hour)
2. Implement LiDAR augmentation pipeline (critical for VQ stability)
3. Integrate W&B logging: log VQ usage histograms, BEV reconstructions per 100 iters
4. Verify: `seq 00 has ~4541 frames` (sanity check on existing download)

> ⚠️ **Do not proceed to Phase 1 until:**  
> - Augmentations validated visually  
> - Validation split (5%) implemented  
> - W&B logging shows decreasing loss on debug config

---

### Phase 1a: Baseline Tokenizer (10–14 days)
**Goal:** Stable VQ reconstruction WITHOUT NFG complexity

**Simplified architecture:**
- Encoder → VQ → Decoder → **Spatial skip branch ONLY** (binary occupancy)
- Loss: L1 voxel reconstruction + VQ commitment loss (λ=0.25)
- **NO NFG rendering loss** (defer to Phase 1b)

**Success criteria:**
- VQ codebook usage >95% after 5k iterations
- Chamfer distance <0.4m on validation set
- Peak GPU memory <10GB (debug config) → <11.5GB (prototype config)

**Debugging protocol:**
1. Start with **debug config** (256×256×16 → 32×32 tokens)
2. Verify loss decreases for 1k iters → scale to prototype config
3. Only after stability: proceed to Phase 1b

---

### Phase 1b: NFG Integration (7 days)
**Goal:** Add differentiable depth rendering

**Incremental integration:**
1. Implement NFG branch (trilinear interp + MLP occupancy)
2. Add depth rendering loss (surface concentration ε=0.4)
3. Validate: predicted depth vs ray depth scatter plot shows correlation
4. **Critical check:** Gradient norm through NFG <1.0 (clip if unstable)

**Success criteria:**
- Depth AbsRel <0.15 on validation rays
- No VQ collapse after NFG integration (usage remains >90%)

---

### Phase 1.5: Pre-tokenize Dataset (1 day)
- Tokenize KITTI sequences 00-10 (train) → save 64×64 integer grids
- Total size: ~15k frames × 8KB = **120MB** (fits in RAM)

---

### Phase 2: World Model (3 weeks)
**Curriculum training schedule:**
| Steps   | Objective Mix                          | Purpose                     |
|---------|----------------------------------------|-----------------------------|
| 0–5k    | 100% past=GT, future=masked            | Stabilize temporal modeling |
| 5k–20k  | 50% GT past + 40% full mask + 10% CFG  | Introduce diffusion noise   |
| 20k+    | Full mixed objectives                  | Final training              |

**Critical mitigations:**
- Gradient norm clipping = 1.0 (not 5.0) initially
- CFG weight w=1.0 during training (w=2.0 only at inference)
- Monitor token prediction accuracy per timestep (should increase monotonically)

**Success criteria:**
- Predict 5 future frames (0.5s @10Hz) with Chamfer <0.8m
- Peak memory <11.8GB during training

---

### Phase 3: Evaluation (3 days)
**KITTI-specific protocol:**
```yaml
# configs/worldmodel_kitti_0.5s.yaml
prediction_horizon: 5          # 5 future frames @10Hz = 0.5s
past_frames: 5
sequences_val: [09, 10]        # Standard KITTI val split
sequences_test: [11, 12, ..., 21]
metrics_roi: [-35, 35, -35, 35, -3.0, 1.0]  # Tighter than NuScenes
```

**Metrics:** Chamfer distance, L1 depth (mean/median), AbsRel within ROI

---

## Key Design Decisions (Updated)

| Decision | Rationale |
|----------|-----------|
| **KITTI as primary dataset** | 23k frames enables VQ stability; already downloaded |
| **Spatial range = [-40m, 40m]²** | Matches KITTI scene width; maintains 0.156m resolution |
| **Debug config first** | 256×256×16 grid validates pipeline before memory pressure |
| **Phase 1a/1b split** | Isolates VQ stability issues from NFG rendering complexity |
| **LiDAR augmentations mandatory** | Prevents VQ collapse with limited data (critical for 12GB GPU) |
| **W&B logging from Phase 0** | Early detection of training collapse (VQ usage <90% = failure) |
| **Ray chunks = 256** | Prevents OOM during NFG training (4096 rays → 16× memory reduction) |
| **Z-resolution = 32** | Acceptable tradeoff: lateral precision > vertical for driving tasks |

---

## Verification Plan (Enhanced)

| Milestone | Validation Check | Pass Criteria |
|-----------|------------------|---------------|
| Phase 0   | W&B logs per 100 iters | Loss curve visible; VQ usage histogram logged |
| Phase 1a  | Debug config training | Loss ↓ for 1k iters; GPU mem <8GB |
| Phase 1a  | Prototype config | VQ usage >95%; Chamfer <0.4m on val |
| Phase 1b  | Depth rendering | Predicted vs actual depth R² >0.7 |
| Phase 2   | World model smoke test | Loss ↓ for 500 iters on debug config |
| Phase 2   | Full training | Prediction Chamfer <0.8m @0.5s horizon |
| All phases| Memory profiling | Peak GPU <11.8GB (measured via torch.cuda.max_memory_allocated) |

> ✅ **First milestone definition:** "Tokenizer reconstructs 50% of input points within 0.5m Chamfer distance on debug config" — not full world modeling.

---

## Risk Mitigation Table

| Risk | Probability | Mitigation |
|------|-------------|------------|
| VQ codebook collapse | High | Augmentations + K-Means reinit (dead code threshold=3%) + W&B monitoring |
| NFG gradient explosion | Medium | Gradient clipping=0.5 initially; validate depth scatter plots before full training |
| OOM during world model | Medium | Start debug config (32×32 tokens); scale only after stability |
| KITTI pose drift | Low | Use relative pose deltas between consecutive frames (not absolute poses) |
| Insufficient sequences | Low | KITTI seq 00-10 = 15k frames → 500+ sequences of 10 frames = sufficient |

---

## Realistic Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 0 | 4–5 days | KITTI loader + augmentations + W&B logging |
| Phase 1a | 10–14 days | Stable tokenizer (Chamfer <0.4m) |
| Phase 1b | 7 days | NFG depth rendering validated |
| Phase 2 | 3 weeks | World model predicts 0.5s with Chamfer <0.8m |
| Phase 3 | 3 days | Full evaluation on seq 11–21 |
| **Total** | **5–6 weeks** | Working prototype (not SOTA results) |

> 💡 **Recommendation:** Budget 2 extra days between phases for debugging. First working prototype expected at **~3 weeks** (after Phase 1a).
```


---

## Plan Review: Bugs & Fixes

*Review date: 2026-02-01. Verified against actual filesystem and KITTI/NuScenes data on disk.*

### Critical Bugs

#### Bug 1: KITTI sequences 11-21 have NO ground truth poses
**Location:** Lines 279, 373, 420, 432

The plan uses sequences 11-21 for testing/evaluation:
```yaml
sequences_test: [11, 12, ..., 21]
```
and says "Full evaluation on seq 11-21".

**Problem:** Confirmed on disk -- only `dataset/poses/00.txt` through `10.txt` exist. Sequences 11-21 have velodyne + calibration + timestamps but **no poses**. CoPilot4D requires ego-motion poses for action conditioning in the world model. You cannot evaluate future prediction without knowing the ego-motion between frames.

**Fix:** Use sequences 00-08 for training, 09-10 for validation/test (standard KITTI odometry split). Remove all references to sequences 11-21 for evaluation.
- Line 279: Change `# Load sequences 00-10 (train), 11-21 (val/test)` to `# Load sequences 00-08 (train), 09-10 (val/test)`
- Line 373: Change `sequences_test: [11, 12, ..., 21]` to `sequences_test: [09, 10]` (or hold out 10 as test, 09 as val)
- Line 420: Update frame count accordingly
- Line 432: Change "Full evaluation on seq 11-21" to "Full evaluation on seq 09-10"

---

#### Bug 2: KITTI data path has wrong case
**Location:** Line 277

```python
root="/media/skr/storage/self_driving/CoPilot4D/Data/KITTI"
```

**Problem:** The actual directory on disk is `Data/Kitti` (lowercase `itti`), not `Data/KITTI`. This will cause a `FileNotFoundError` immediately on Linux (case-sensitive filesystem).

**Fix:** Change to `root="/media/skr/storage/self_driving/CoPilot4D/Data/Kitti"`

---

#### Bug 3: World Model U-Net dimensions don't match prototype BEV token grid
**Location:** Lines 128-133

The U-Net layout is specified as:
```
Level 1 down (128x128, dim=256) -> Level 2 down (64x64, dim=384) -> Level 3 down (32x32, dim=512)
```

**Problem:** The prototype uses 64x64 BEV tokens (lines 20, 216), not 128x128. If you feed 64x64 tokens into this U-Net, Level 1 would be 64x64, Level 2 would be 32x32, Level 3 would be 16x16. The plan describes paper-resolution dimensions but the prototype halves them. An implementer following these dimensions literally will get shape mismatches.

**Fix:** Add a note that lines 128-133 describe paper-resolution dimensions. For prototype (64x64 tokens): Level 1 = 64x64, Level 2 = 32x32, Level 3 = 16x16. Better yet, express levels as relative: Level 1 = full resolution, Level 2 = 1/2, Level 3 = 1/4. Make patch merging/expanding configurable.

---

#### Bug 4: Decoder/NFG/Spatial-skip dimensions hardcoded to paper resolution
**Location:** Lines 112-114

```
Decoder output: (256, 256, 128)
NFG: Linear(128, 4*64*16) -> reshape (512, 512, 64, 16)
Spatial skip: Linear(128, 16*64) -> (1024, 1024, 64)
```

**Problem:** These are all paper-resolution (1024x1024 voxel grid) dimensions. For the prototype (512x512 voxel grid, 64x64 tokens):
- Decoder output would be **(128, 128, 128)** -- not (256, 256, 128)
- NFG should reshape to **(256, 256, 32, 16)** -- not (512, 512, 64, 16)
- Spatial skip should reshape to **(512, 512, 32)** -- not (1024, 1024, 64)

The hardcoded Linear layer parameters (`4*64*16`, `16*64`) embed spatial upsampling factors and Z-resolution that change with prototype config. Using these values directly will produce wrong output shapes.

**Fix:** Express these as formulas parameterized by config:
- NFG: `Linear(C, upsample_xy^2 * Z_dim * feat_dim)` -> reshape `(H*upsample_xy, W*upsample_xy, Z_dim, feat_dim)`
- Spatial skip: `Linear(C, skip_upsample^2 * Z_dim)` -> reshape `(H*skip_up, W*skip_up, Z_dim)`
- For paper: upsample_xy=2, Z_dim=64, skip_upsample=4. For prototype: upsample_xy=2, Z_dim=32, skip_upsample=4.

---

### Medium Bugs

#### Bug 5: Document structure -- two overlapping contradictory plans in one file
**Location:** Lines 1-190 (original NuScenes-first plan) vs. Lines 194-436 (KITTI-optimized plan)

The KITTI-optimized plan is embedded inside a `` ```markdown ``` `` code block, so it renders as a code snippet rather than actual content. The two plans contradict each other on: primary dataset, gradient clip values, ray counts, evaluation protocol. An implementer won't know which to follow.

**Fix:** Remove the original plan (lines 1-190) or merge both into a single coherent document. The KITTI-optimized plan supersedes the original.

---

#### Bug 6: Frame count contradiction (23k vs 15k)
**Location:** Lines 205, 385 vs. Line 420

Lines 205/385 say "23k frames" for KITTI. Line 420 says "KITTI seq 00-10 = 15k frames". Actual count from disk for sequences 00-10: 4541+1101+4661+801+271+2761+1101+1101+4071+1591+1201 = **~23,201 frames**.

**Fix:** Line 420 is wrong. Change "15k frames" to "~23k frames" and update the derived sequence count: "23k frames -> 2300+ sequences of 10 frames".

---

#### Bug 7: Ray chunks table is ambiguous and possibly inverted
**Location:** Lines 221-222

| Config | Ray chunks |
|--------|-----------|
| Paper | 4096 |
| Prototype | 256 |
| Debug | 512 |

**Problem:** "Ray chunks" conflates two concepts: total rays per frame and chunk size for processing. Also, debug (512) > prototype (256) is backwards if both refer to the same thing -- debug should be cheaper. If 256 means chunk size, the total rays/frame is unspecified for prototype.

**Fix:** Split into two rows:
- "Rays per frame" (total rays sampled): Paper=4096, Prototype=2048, Debug=1024
- "Ray chunk size" (processed per batch): Paper=1024, Prototype=256, Debug=256

---

#### Bug 8: KITTI 3s evaluation is undefined
**Location:** Line 166

```
KITTI 10Hz: 5 past -> 5 future (subsample for 3s)
```

**Problem:** 5 frames at 10Hz = 0.5s, not 3s. "Subsample for 3s" is vague -- does it mean use every 6th frame to span 3s? That would create huge temporal gaps not seen during training. The KITTI section (line 370) correctly says 0.5s, contradicting this.

**Fix:** Change to: `KITTI 10Hz: 5 past -> 5 future = 0.5s prediction horizon. For longer horizons (e.g., 3s), would require 30 past -> 30 future frames (out of scope for prototype).`

---

#### Bug 9: Missing U-Net skip connections description
**Location:** Lines 128-134

The world model is described as a "3-level U-Net" but skip connections between downsample and upsample paths are never mentioned. Without skip connections it is just an encoder-decoder, not a U-Net.

**Fix:** Add after line 133: "Skip connections: features from each downsample level are concatenated (channel-wise) with the corresponding upsample level before the ST blocks. Requires projection Linear layers to match channel dims after concatenation."

---

#### Bug 10: Gradient clip values are inconsistent across the two plans

| Component | Original plan | KITTI plan |
|-----------|--------------|------------|
| Tokenizer | 0.1 (line 116) | Not restated |
| World model | 5.0 (line 158) | 1.0 initially (line 356) |

**Problem:** The KITTI plan silently changes world model grad clip from 5.0 to 1.0 without justification. These are very different values (5x). The tokenizer clip of 0.1 is also very aggressive and may cause slow convergence.

**Fix:** Pick one value per component, document it in both plan sections, and justify any deviations. If the KITTI plan's 1.0 is intentional for stability, say so explicitly and note it differs from the paper's 5.0.

---

#### Bug 11: Random val_split causes data leakage on sequential data
**Location:** Line 278

```python
val_split=0.05  # Random 5% split
```

**Problem:** Random frame-level splitting on sequential driving data causes data leakage -- validation frames will be temporally adjacent (sometimes <0.1s apart) to training frames from the same sequence. The world model uses temporal sequences, so a val frame's neighbors could appear in training data.

**Fix:** Use sequence-level splitting: train on seq 00-08, validate on seq 09, test on seq 10. The evaluation config on line 372 partially does this (`sequences_val: [09, 10]`) but contradicts the dataset constructor's `val_split=0.05`. Remove `val_split` and use `train_sequences=[0..8], val_sequences=[9], test_sequences=[10]`.

---

### Minor Bugs

#### Bug 12: `grid_sample` mode naming will confuse implementers
**Location:** Line 176

> "Use trilinear (PyTorch grid_sample on 5D input with bilinear mode)"

PyTorch `F.grid_sample` uses `mode='bilinear'` for both 2D bilinear and 3D trilinear -- there is no `mode='trilinear'` parameter.

**Fix:** Reword to: "Use `F.grid_sample(input_5d, grid, mode='bilinear', ...)` -- PyTorch performs trilinear interpolation automatically for 3D volumes despite the 'bilinear' parameter name."

---

#### Bug 13: Depth rendering variable name is wrong
**Location:** Line 113

> `D = sum(w_i * h_i)`

The variable `h_i` typically denotes a hidden state. For depth rendering it should be `d_i` (depth of the i-th sample point along the ray).

**Fix:** Change to `D = sum(w_i * d_i)` where `d_i` is the depth of sample point i along the ray.

---

#### Bug 14: Vertical angle weighting is a custom deviation from the paper
**Location:** Line 289

```python
# Apply vertical angle weighting: weights = cos(atan2(z, sqrt(x^2+y^2)))^2
```

This is not from the CoPilot4D paper. Custom modifications to a reproduction should be flagged as deviations to avoid confusion when debugging discrepancies with paper results.

**Fix:** Add a comment: `# NOTE: Custom addition, not in CoPilot4D paper. May help with KITTI's Velodyne-64 vertical distribution.`

---

### Bug Summary Table

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| 1 | **CRITICAL** | KITTI seq 11-21 have no poses -- can't evaluate | L279, 373, 432 |
| 2 | **CRITICAL** | KITTI path case mismatch (`KITTI` vs `Kitti`) | L277 |
| 3 | **CRITICAL** | U-Net dimensions wrong for prototype (128^2 vs 64^2 tokens) | L128-133 |
| 4 | **CRITICAL** | Decoder/NFG/Skip dims hardcoded to paper resolution | L112-114 |
| 5 | Medium | Two overlapping contradictory plans in one file | L1-436 |
| 6 | Medium | Frame count contradiction (23k vs 15k) | L205 vs 420 |
| 7 | Medium | Ray chunks table ambiguous and possibly inverted | L221 |
| 8 | Medium | "Subsample for 3s" KITTI eval undefined | L166 |
| 9 | Medium | U-Net skip connections never described | L128-134 |
| 10 | Medium | Gradient clip values contradictory between plans | L116, 158, 356 |
| 11 | Medium | Random val split causes data leakage on sequences | L278 |
| 12 | Minor | grid_sample mode naming confusing | L176 |
| 13 | Minor | Depth rendering variable name wrong (h_i -> d_i) | L113 |
| 14 | Minor | Vertical angle weighting not flagged as paper deviation | L289 |

The most urgent fix is **#1** -- the entire evaluation strategy for the world model is based on sequences that lack the pose data CoPilot4D requires. Followed by **#2** which will cause an immediate crash, and **#3/#4** which will produce shape mismatches the moment someone implements the prototype config.