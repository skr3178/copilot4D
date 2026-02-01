## Plan

cat ~/.claude/plans/giggly-hugging-cat.md 

  6 tasks (2 done, 1 in progress, 3 open) · ctrl+t to hide tasks                                                                              
  ◼ Implement tokenizer modules (Layer 2)                                                                                                     
  ◻ Assemble full tokenizer model (Layer 3)                                                                                                   
  ◻ Create configs and training script (Layer 4)                                                                                              
  ◻ Create shape verification tests (Layer 5)                                                                                                 
  ✔ Create package structure and config (Layer 0)                                                                                             
  ✔ Implement data pipeline (Layer 1)    
# CoPilot4D Tokenizer Implementation Plan Review

## Overview
This document provides corrections and detailed specifications for implementing the CoPilot4D tokenizer based on the actual paper (ICLR 2024), addressing deviations from the original proposal.

---

## 1. Encoder Architecture (Critical Corrections)

**❌ Original Plan:** Pure Conv2d layers  
**✅ Paper Implementation:** PointNet + Swin Transformer

### Correct Architecture (Section A.2.1)

```python
class TokenizerEncoder(nn.Module):
    def __init__(self):
        # 1. Voxel-wise PointNet (similar to VoxelNet)
        self.pointnet = PointNet( 
            input_dim=4,  # [x,y,z] + offset features
            output_dim=64,
            voxel_size=(0.15625, 0.15625, 0.140625)  # meters
        )
        # Processes each voxel: MaxPool/Sum+LN of point features

        # 2. 3D → 2D BEV Pooling
        self.bev_pool = BEVPillarPooling(
            z_range=(-4.5, 4.5),
            aggregation='sum+layernorm'  # Critical: not maxpool
        )

        # 3. Swin Transformer backbone (not CNN!)
        # Two-stage process: PatchEmbed(4x) + PatchMerge(2x) = 8x total downsample
        # For prototype (512 voxel grid): 512 / 8 = 64x64 tokens
        self.patch_embed = PatchEmbed(patch_size=4, in_dim=64, embed_dim=128)  # 4x downsample
        self.swin_stage1 = SwinStage(dim=128, num_heads=8, num_blocks=2, window_size=8)
        self.patch_merge = PatchMerging(dim=128)  # 2x downsample
        self.swin_stage2 = SwinStage(dim=256, num_heads=16, num_blocks=6, window_size=8)
        # Total: 4x * 2x = 8x downsample. 1024 -> 128 tokens (prototype: 512 -> 64)

        # 4. Vector Quantization
        self.vq = VectorQuantizer(
            codebook_size=1024,
            dim=1024,  # Paper increases dim before VQ
            commitment_cost=0.25
        )
```

**Key Details:**
- Uses **Sum + LayerNorm** aggregation (not max pooling) for permutation invariance
- Swin Transformer processes BEV features hierarchically via 2 stages
- Downsampling: PatchEmbed(4x) + PatchMerging(2x) = **8x total**
- Output: **128×128 tokens** at paper resolution (1024 grid), **64×64** at prototype (512 grid)

---

## 2. Decoder: Neural Feature Grid + Implicit Representation

**❌ Original Plan:** ConvTranspose2d outputs to BEV features  
**✅ Paper Implementation:** 3D Neural Feature Grid with differentiable depth rendering

### Correct Decoder Architecture

```python
class TokenizerDecoder(nn.Module):
    def __init__(self, upsample_xy=2, z_dim=64, feat_dim=16):
        # Step 1: Swin Decoder upsamples tokens back to intermediate resolution
        # 6 Swin blocks (dim=256) -> PatchUpsample (2x) -> 2 Swin blocks (dim=128)
        # 128x128 tokens -> 256x256 feature map (paper)
        # 64x64 tokens -> 128x128 feature map (prototype)
        self.swin_decoder = SwinDecoder(...)
        # Output: (H_dec, W_dec, 128) where H_dec = token_grid * 2

        # Two branches (Section 4.1):

        # Branch 1: 3D Neural Feature Grid (NFG) - Primary
        # Step 2: Per-position linear produces a local 3D sub-volume
        # Each of the H_dec x W_dec positions outputs a (upsample_xy, upsample_xy, Z, 16) block
        self.nfg_head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, upsample_xy**2 * z_dim * feat_dim)
            # Paper:     Linear(128, 2*2*64*16 = 4096)  -> NFG 512x512x64x16
            # Prototype: Linear(128, 2*2*32*16 = 2048)  -> NFG 256x256x32x16
            # Debug:     Linear(128, 2*2*16*16 = 1024)  -> NFG 128x128x16x16
        )
        # Reshape: (H_dec, W_dec, upsample_xy, upsample_xy, Z, 16)
        #       -> (H_dec*up, W_dec*up, Z, 16) = full NFG

        self.implicit_mlp = nn.Sequential(
            nn.Linear(feat_dim, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()  # Occupancy α
        )

        # Branch 2: Coarse Spatial Skipping (binary)
        self.spatial_skip_head = nn.Sequential(
            nn.Linear(128, 1024*1024*64),  # Binary voxel grid
            nn.Sigmoid()
        )

    def render_depth(self, nfg, rays):
        """Differentiable depth rendering (Equation 4)"""
        # Sample points along rays, query NFG via trilinear interp
        # Return expected depth D(r) = Σ w_i * d_i
        # where w_i = α_i * Π(1-α_j) (volume rendering weights)
        # d_i = depth of sample point i along the ray
```

**NFG memory per config:**

| Config | Decoder output | Linear output/position | NFG shape | Memory (FP32) |
|--------|---------------|------------------------|-----------|---------------|
| Paper | 256x256x128 | `2*2*64*16 = 4096` | 512x512x64x16 | **1.07 GB** |
| Prototype | 128x128x128 | `2*2*32*16 = 2048` | 256x256x32x16 | **134 MB** |
| Debug | 64x64x128 | `2*2*16*16 = 1024` | 128x128x16x16 | **17 MB** |

**Critical Innovation:** Instead of reconstructing voxels directly, CoPilot4D renders point clouds from BEV tokens using volume rendering (similar to NeRF but with explicit feature grids).

---

## 3. Loss Functions

**❌ Original Plan:** Simple Chamfer distance  
**✅ Paper Implementation:** Multi-component loss with surface concentration

```python
def compute_loss(self, pred_depth, gt_depth, weights, epsilon=0.4):
    # L1 depth loss
    l1_loss = torch.abs(pred_depth - gt_depth).mean()

    # Surface concentration term (Equation 5)
    # Encourage weights w_i to concentrate within ε of surface
    surface_mask = torch.abs(sampled_heights - gt_depth.unsqueeze(-1)) > epsilon
    concentration_loss = torch.sum(weights ** 2 * surface_mask.float())

    # Binary cross entropy for spatial skipping branch
    skip_loss = F.binary_cross_entropy(pred_binary, gt_binary)

    # VQ loss with straight-through estimator
    vq_loss = commitment_loss + codebook_loss

    return l1_loss + concentration_loss + skip_loss + 0.25*vq_loss
```

---

## 4. Data Preprocessing Corrections

| Parameter | Original Plan | Paper (Section A.2.1) |
|-----------|---------------|----------------------|
| **ROI** | ±50m | **±80m** in x,y; ±4.5m in z |
| **Voxel size** | 0.2m | **0.15625m** (x,y), 0.140625m (z) |
| **Initial grid** | 512×512 | **1024×1024×64** |
| **Token grid** | 32×32 | **128×128** (8× downsampled) |
| **Tokens/frame** | 1,024 | **16,384** |

```python
# Correct preprocessing
def preprocess_lidar(point_cloud):
    # Filter points within ROI
    roi_mask = (
        (point_cloud[:, 0] >= -80) & (point_cloud[:, 0] <= 80) &  # x
        (point_cloud[:, 1] >= -80) & (point_cloud[:, 1] <= 80) &  # y
        (point_cloud[:, 2] >= -4.5) & (point_cloud[:, 2] <= 4.5)   # z
    )
    points = point_cloud[roi_mask]

    # Voxelization
    voxel_size = (0.15625, 0.15625, 0.140625)
    grid_shape = (1024, 1024, 64)  # x, y, z

    return voxel_grid
```

---

## 5. Spatial Skipping Implementation (Section A.2.1)

During inference, use the coarse branch to accelerate rendering:

```python
def render_with_spatial_skipping(self, bev_tokens, rays):
    # 1. Predict binary occupancy from coarse branch
    binary_logits = self.spatial_skip_head(bev_tokens)
    binary_probs = torch.sigmoid(binary_logits)

    # 2. Add noise and threshold (during inference)
    noise = torch.distributions.Logistic(0, 1).sample(binary_probs.shape)
    binary_mask = (binary_probs + noise) > 0

    # 3. Max pooling to increase recall (factor 8 in BEV)
    coarse_mask = F.max_pool3d(
        binary_mask.float(), 
        kernel_size=(8, 8, 1), 
        stride=(8, 8, 1)
    )

    # 4. Only sample points within non-empty coarse voxels
    valid_rays = rays[coarse_mask[rays.coordinates]]

    # 5. Render depth only for valid rays
    depth = self.render_depth(nfg, valid_rays)
    return depth
```

**Purpose:** Speeds up inference by skipping empty space during raycasting.

---

## 6. Ray Construction from LiDAR Point Clouds

Rays are constructed directly from the measured point positions (standard for LiDAR depth rendering). No synthetic ray grid is needed.

```python
def generate_rays_from_pointcloud(points_xyz, sensor_origin=None):
    """
    Construct rays from a KITTI Velodyne scan.

    Args:
        points_xyz: (N, 3) point cloud in Velodyne frame
        sensor_origin: (3,) default [0, 0, 0] in Velodyne frame

    Returns:
        ray_origins: (N, 3) all identical (sensor position)
        ray_dirs: (N, 3) unit vectors from sensor to each point
        ray_depths: (N,) Euclidean distance from sensor to point
    """
    if sensor_origin is None:
        sensor_origin = np.zeros(3)

    directions = points_xyz - sensor_origin          # (N, 3)
    depths = np.linalg.norm(directions, axis=1)      # (N,)
    ray_dirs = directions / depths[:, None]           # (N, 3) unit vectors
    ray_origins = np.broadcast_to(sensor_origin, (len(points_xyz), 3))

    return ray_origins, ray_dirs, depths
```

**Training ray sampling:**
- Sample `num_rays` random rays per frame from the full ~120k points
  - Paper: 4096 rays/frame
  - Prototype: 2048 rays/frame
  - Debug: 1024 rays/frame
- Process in chunks (256 rays per chunk) for memory efficiency

**KITTI detail:** The sensor origin is `(0, 0, 0)` in the Velodyne frame, which is the coordinate frame returned by `pykitti.get_velo()`. No transform needed.

---

## 7. Codebook Management Strategy

**❌ Standard:** Random restart for dead codes  
**✅ Paper:** K-Means clustering restart

```python
class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, dim):
        self.codebook = nn.Embedding(codebook_size, dim)
        self.memory_bank = []  # Store recent encoder outputs
        self.memory_bank_size = 10 * codebook_size
        self.dead_code_threshold = 256  # iterations

    def update_codebook(self, z_e, usage_counts):
        # Check for dead codes
        dead_codes = (usage_counts == 0).sum()

        if dead_codes > 0.03 * self.codebook_size:  # >3% dead
            if len(self.memory_bank) >= self.memory_bank_size:
                # Run K-Means on memory bank
                data = torch.stack(self.memory_bank)
                new_codes = kmeans(data, self.codebook_size)
                self.codebook.weight.data = new_codes

        # Standard EMA or straight-through update
```

---

## 8. Implementation Sequence Recommendation

1. **Phase 1: 3D Encoder**
   - Implement PointNet with Sum+LayerNorm aggregation
   - BEV pillar pooling
   - Swin Transformer backbone

2. **Phase 2: Implicit Decoder**
   - Neural Feature Grid generation
   - Differentiable depth rendering (critical!)
   - Verify gradients flow through ray sampling

3. **Phase 3: Spatial Skipping**
   - Add binary classification branch
   - Implement max pooling mask
   - Optimize inference speed

4. **Phase 4: Training**
   - Straight-through estimator for VQ
   - K-Means codebook restart
   - Surface concentration loss

---

## 9. Validation Checklist

Before training the world model, verify:

- [ ] **Reconstruction Chamfer loss < 0.1m** on validation set
- [ ] **Codebook usage > 90%** (check for collapse)
- [ ] **128×128 token grid** output shape
- [ ] **Temporal coherence:** Adjacent frames have similar token distributions
- [ ] **Qualitative check:** Rendered point clouds preserve:
  - Vehicle shapes
  - Road boundaries/curbs
  - Overhead structures

---

## 10. Model Specifications

| Component | Paper Value |
|-----------|-------------|
| **Total Parameters** | 13M (tokenizer only) |
| **Codebook Size** | 1024 |
| **Token Dimension** | 1024 (pre-VQ), 512 (latent) |
| **Sequence Length** | 128×128 = 16,384 tokens |
| **Voxel Size** | (0.15625, 0.15625, 0.140625) m |
| **Rendering Samples** | N_r per ray (configurable, ~64-128) |
| **Surface Margin ε** | 0.4m |

### Memory Budget (FP32)

| Component | Paper | Prototype | Debug |
|-----------|-------|-----------|-------|
| Sparse voxels (encoder input) | ~120k x 16 x 4B = **8 MB** | Same | Same |
| BEV features (after pooling) | 1024x1024x64 x 4B = **256 MB** | 512x512x64 = **64 MB** | 256x256x64 = **16 MB** |
| Swin encoder activations | ~500 MB | ~125 MB | ~32 MB |
| VQ codebook | 1024x1024 x 4B = **4 MB** | Same | Same |
| NFG volume | 512x512x64x16 x 4B = **1.07 GB** | 256x256x32x16 = **134 MB** | 128x128x16x16 = **17 MB** |
| Ray samples (chunk of 256 rays x 64 pts) | 256x64x16 x 4B = **1 MB** | Same | Same |
| **Estimated peak** | **~3-4 GB** | **~500 MB** | **~100 MB** |

With FP16 mixed precision, activation memory is approximately halved. The prototype config fits comfortably on a 12GB RTX 3060.

**Critical:** BEV pooling must use sparse scatter, NOT dense tensors. Never materialize a dense (1024, 1024, 64) tensor -- use sparse coords + scatter_add.

---

## References

- Paper: "CoPilot4D: Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion" (ICLR 2024)
- Section A.2.1: Model Details (Tokenizer)
- Section 4.1: Tokenize the 3D World
- Section 4.2: MaskGIT as Discrete Diffusion
