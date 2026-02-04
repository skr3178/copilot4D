# Tokenizer Equations Review: Paper vs. Code Implementation

**Paper:** CoPilot4D: Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion (ICLR 2024)  
**Code Location:** `/media/skr/storage/self_driving/CoPilot4D/copilot4d/tokenizer/`  
**Date:** 2026-02-03

---

## Overview

This document reviews the mathematical equations presented in the CoPilot4D paper (Section 4.1: Tokenize the 3D World) and their corresponding implementations in the tokenizer codebase.

---

## 1. Volume Rendering for Depth Prediction (Equation 4)

### Paper Equation

For a ray $r(h) = p + hd$ with $N_r$ sampled points along the ray:

$$
\alpha_i = \sigma(\text{MLP}(\text{interp}(\text{NFG}(\hat{z}), (x_i, y_i, z_i))))
$$

$$
w_i = \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)
$$

$$
D(r, \hat{z}) = \sum_{i=1}^{N_r} w_i h_i
$$

Where:
- $\alpha_i$ = occupancy probability at sample point $i$
- $w_i$ = volume rendering weight (transmittance × density)
- $D(r, \hat{z})$ = expected depth along ray $r$

### Code Implementation

**File:** `neural_feature_grid.py` (lines 173-188)

```python
# Occupancy MLP -> alpha
alpha = self.occ_mlp(features).squeeze(-1)  # (B, C, S)
alpha = torch.sigmoid(alpha)                # σ activation

# Volume rendering weights: w_i = α_i * cumprod(1 - α_j, j < i)
one_minus_alpha = 1.0 - alpha + 1e-10
transmittance = torch.cumprod(
    torch.cat([torch.ones(B, C, 1, device=device), one_minus_alpha[..., :-1]], dim=-1),
    dim=-1,
)
weights = alpha * transmittance  # (B, C, S)

# Expected depth: D = Σ w_i * h_i
pred_depth = (weights * sample_depths).sum(dim=-1)  # (B, C)
```

### Verification

| Component | Paper | Code |
|-----------|-------|------|
| Occupancy MLP | $\sigma(\text{MLP}(\cdot))$ | `torch.sigmoid(self.occ_mlp(features))` |
| Weight computation | $\alpha_i \prod_{j<i}(1-\alpha_j)$ | `alpha * cumprod(1-alpha)` |
| Depth aggregation | $\sum w_i h_i$ | `(weights * sample_depths).sum()` |

**Status:** ✅ **Correctly implemented**

---

## 2. Vector Quantization Loss

### Paper Equation

From Section 4.1:

$$
\mathcal{L}_{vq} = \lambda_1 \|\text{sg}[E(o)] - \hat{z}\|_2^2 + \lambda_2 \|\text{sg}[\hat{z}] - E(o)\|_2^2
$$

Where:
- First term = codebook loss (updates codebook vectors)
- Second term = commitment loss (updates encoder)
- sg = stop_gradient operator

### Code Implementation

**File:** `vector_quantizer.py`

**Commitment Loss** (lines 153-154):
```python
# Commitment loss: ||z_e - sg(z_q)||^2
commitment_loss = self.commitment_cost * F.mse_loss(flat, z_q.detach())
```

**EMA Codebook Update** (lines 131-148):
```python
# EMA update of codebook (handles first term implicitly)
self.cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

# Laplace smoothing
cluster_size_smoothed = (
    (self.cluster_size + self.eps)
    / (n + self.codebook_size * self.eps)
    * n
)
self.embed.data.copy_(self.embed_avg / cluster_size_smoothed.unsqueeze(1))
```

### Verification

| Term | Paper | Code |
|------|-------|------|
| Codebook Loss | $\lambda_1 \|\text{sg}[E(o)] - \hat{z}\|^2$ | EMA update (implicit) |
| Commitment Loss | $\lambda_2 \|\text{sg}[\hat{z}] - E(o)\|^2$ | `F.mse_loss(flat, z_q.detach())` |

**Note:** The code uses EMA (Exponential Moving Average) to update codebook vectors instead of explicit gradient descent, which is a common practice in modern VQ implementations (VQVAE-2, VQGAN).

**Status:** ✅ **Correctly implemented** (EMA-based VQ pattern)

---

## 3. Tokenizer Rendering Loss (Equation 5)

### Paper Equation

$$
\mathcal{L}_{render} = \mathbb{E}_r\left[\|D(r, \hat{z}) - D_{gt}\|_1 + \sum_i \mathbb{1}(|h_i - D_{gt}| > \epsilon)\|w_i\|^2\right] + \text{BCE}(v, v_{gt})
$$

Where:
- Term 1: L1 depth loss
- Term 2: Surface concentration loss (penalizes weights far from surface)
- Term 3: Binary cross-entropy for spatial skip branch

### Code Implementation

**File:** `tokenizer_losses.py`

**L1 Depth Loss** (lines 8-18):
```python
def l1_depth_loss(pred_depths: torch.Tensor, gt_depths: torch.Tensor) -> torch.Tensor:
    """L1 depth loss between predicted and ground truth depths."""
    return torch.abs(pred_depths - gt_depths).mean()
```

**Surface Concentration Loss** (lines 21-53):
```python
def surface_concentration_loss(
    weights: torch.Tensor,
    sample_depths: torch.Tensor,
    gt_depths: torch.Tensor,
    epsilon: float = 1.0,
) -> torch.Tensor:
    """Surface concentration loss: penalize weights far from surface.
    
    L_conc = (w_i^2 * [|d_i - d_gt| > epsilon]).sum() / (B*R)
    """
    # Expand gt_depths to match sample_depths
    gt_depths_expanded = gt_depths.unsqueeze(-1)  # (B, R, 1)
    
    # Mask: 1 where |d_i - d_gt| > epsilon
    far_mask = (torch.abs(sample_depths - gt_depths_expanded) > epsilon).float()
    
    # Penalize squared weights that are far from surface
    loss = (weights ** 2 * far_mask).sum() / (B * R)
    return loss
```

**Spatial Skip BCE Loss** (lines 56-77):
```python
def spatial_skip_bce_loss(
    skip_logits: torch.Tensor,
    gt_occupancy: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy loss for spatial skip occupancy prediction."""
    logits_flat = skip_logits.reshape(-1)
    gt_flat = gt_occupancy.reshape(-1)
    loss = F.binary_cross_entropy_with_logits(logits_flat, gt_flat)
    return loss
```

**Total Loss** (lines 80-129):
```python
def tokenizer_total_loss(...):
    # L1 depth loss
    depth_l1 = l1_depth_loss(pred_depths, gt_depths)
    
    # Surface concentration loss
    surface_conc = surface_concentration_loss(
        weights, sample_depths, gt_depths, epsilon=surface_conc_eps
    )
    
    # Spatial skip BCE loss
    skip_bce = spatial_skip_bce_loss(skip_logits, gt_occupancy)
    
    # Total loss: L_depth + L_surface + λ*L_vq + L_skip
    total = depth_l1 + surface_conc + vq_weight * vq_loss + skip_bce
    
    return {
        "total": total,
        "depth_l1": depth_l1,
        "surface_conc": surface_conc,
        "vq": vq_loss,
        "skip_bce": skip_bce,
    }
```

### Verification

| Term | Paper | Code |
|------|-------|------|
| L1 Depth | $\|D - D_{gt}\|_1$ | `torch.abs(pred - gt).mean()` |
| Surface Concentration | $\sum \mathbb{1}(|h_i - D_{gt}| > \epsilon)\|w_i\|^2$ | `(weights^2 * far_mask).sum()` |
| Spatial Skip | $\text{BCE}(v, v_{gt})$ | `F.binary_cross_entropy_with_logits` |

**Status:** ✅ **Correctly implemented**

---

## 4. Distance Computation in VQ (Implicit)

### Equation

Efficient distance computation using expansion:

$$
\|z_e - e\|^2 = \|z_e\|^2 - 2 z_e \cdot e^T + \|e\|^2
$$

### Code Implementation

**File:** `vector_quantizer.py` (lines 122-125)

```python
# Compute distances: ||z_e - e||^2 = ||z_e||^2 - 2*z_e*e^T + ||e||^2
z_e_sq = (flat ** 2).sum(dim=1, keepdim=True)     # (B*N, 1)
e_sq = (self.embed ** 2).sum(dim=1, keepdim=True)  # (K, 1)
dist = z_e_sq - 2.0 * flat @ self.embed.t() + e_sq.t()  # (B*N, K)

# Find nearest code
indices = dist.argmin(dim=1)  # (B*N,)
```

**Status:** ✅ **Correctly implemented** - Efficient computation avoids materializing full pairwise distance matrix.

---

## 5. Straight-Through Gradient Estimator

### Equation

For backpropagating through the argmin quantization:

$$
z_q^{st} = z_e + \text{stop_gradient}(z_q - z_e)
$$

### Code Implementation

**File:** `vector_quantizer.py` (lines 156-157)

```python
# Straight-through estimator
z_q_st = flat + (z_q - flat).detach()  # (B*N, codebook_dim)
```

**Status:** ✅ **Correctly implemented** - Standard straight-through gradient estimator.

---

## 6. Neural Feature Grid Query (Implicit in Paper)

### Description

The paper describes querying the Neural Feature Grid (NFG) via bilinear interpolation at continuous coordinates $(x, y, z)$.

### Code Implementation

**File:** `neural_feature_grid.py` (lines 152-171)

```python
# Compute 3D sample positions: origin + t * direction
pts = origins.unsqueeze(2) + directions.unsqueeze(2) * sample_depths.unsqueeze(3)
# pts: (B, C, S, 3) -- xyz coordinates

# Normalize coordinates to [-1, 1] for grid_sample
x_norm = 2.0 * (pts[..., 0] - cfg.x_min) / (cfg.x_max - cfg.x_min) - 1.0
y_norm = 2.0 * (pts[..., 1] - cfg.y_min) / (cfg.y_max - cfg.y_min) - 1.0
z_norm = 2.0 * (pts[..., 2] - cfg.z_min) / (cfg.z_max - cfg.z_min) - 1.0

# grid_sample expects (x, y, z) = (W, H, D) order
grid = torch.stack([y_norm, x_norm, z_norm], dim=-1)  # (B, C, S, 3)
grid = grid.unsqueeze(3)  # (B, C, S, 1, 3)

# Trilinear interpolation via grid_sample
features = F.grid_sample(
    nfg, grid, mode="bilinear", padding_mode="zeros", align_corners=True
)
```

**Status:** ✅ **Correctly implemented** - Uses PyTorch's `F.grid_sample` for trilinear interpolation.

---

## Summary Table

| Equation/Concept | Paper Section | Code File | Lines | Status |
|------------------|---------------|-----------|-------|--------|
| Volume Rendering (Eq. 4) | 4.1 | `neural_feature_grid.py` | 173-188 | ✅ Correct |
| VQ Commitment Loss | 4.1 | `vector_quantizer.py` | 153-154 | ✅ Correct |
| VQ EMA Update | 4.1 (implicit) | `vector_quantizer.py` | 131-148 | ✅ Correct |
| L1 Depth Loss (Eq. 5) | 4.1 | `tokenizer_losses.py` | 8-18 | ✅ Correct |
| Surface Concentration (Eq. 5) | 4.1 | `tokenizer_losses.py` | 21-53 | ✅ Correct |
| Spatial Skip BCE (Eq. 5) | 4.1 | `tokenizer_losses.py` | 56-77 | ✅ Correct |
| Distance Computation | Implicit | `vector_quantizer.py` | 122-125 | ✅ Correct |
| Straight-Through | Implicit | `vector_quantizer.py` | 156-157 | ✅ Correct |
| NFG Query/Interpolation | 4.1 | `neural_feature_grid.py` | 152-171 | ✅ Correct |

---

## Architecture Components Mapping

| Component | Paper Description | Code Implementation |
|-----------|-------------------|---------------------|
| **Voxel Encoder** | PointNet with sum pooling | `voxel_encoder.py`: `VoxelPointNet` |
| **BEV Pooling** | Z-axis aggregation with learned embedding | `bev_pooling.py`: `BEVPillarPooling` |
| **Encoder Backbone** | Swin Transformer (2 stages) | `swin_transformer.py`: `SwinEncoder` |
| **Vector Quantizer** | EMA-based VQ with codebook | `vector_quantizer.py`: `VectorQuantizer` |
| **Decoder Backbone** | Swin Transformer (2 stages) | `swin_transformer.py`: `SwinDecoder` |
| **Neural Feature Grid** | 3D implicit representation | `neural_feature_grid.py`: `NeuralFeatureGrid` |
| **Spatial Skip Branch** | Binary occupancy prediction | `spatial_skipping.py`: `SpatialSkipBranch` |

---

## Notes

1. **EMA vs Explicit Codebook Loss:** The paper's VQ loss equation shows both terms explicitly, but the code uses EMA updates for the codebook (first term) and commitment loss for the encoder (second term). This is a common and valid approach that often provides better training stability.

2. **Surface Concentration Epsilon:** The paper uses $\epsilon$ (default 1.0m) as the threshold for surface concentration loss, matching the `surface_conc_eps` config parameter in the code.

3. **Volume Rendering Sample Count:** The paper mentions $N_r$ sampled points along each ray, which corresponds to `num_depth_samples` in the config (default typically 64-128).

4. **Loss Weights:** The paper mentions $\lambda_1$ and $\lambda_2$ for VQ loss weighting, which in the code are represented by `commitment_cost` and implicit in the EMA decay rate.

---

## Conclusion

All mathematical equations from the paper's tokenizer section (Section 4.1) are correctly implemented in the codebase. The implementation follows the paper's specifications while using standard deep learning patterns (EMA for VQ, straight-through estimators, efficient distance computation).



Looking at Equation 4 in the paper, $h_i$ is the depth value (distance along the ray) for the $i$-th sample point.

  From the paper text:

  ▌ "Given a ray $r(h) = p + hd$ starting at point $p$ and traveling in direction $d$, the expected depth $D$ can be calculated via differentiable depth rendering on $N_r$ sampled poi
  ▌ nts ${(x_i, y_i, z_i)}_{i=1}^{N_r}$ along the ray"

  And Equation 4: $$D(r, \hat{z}) = \sum_{i=1}^{N_r} w_i h_i$$

  In volume rendering terminology:

  • $h_i$ = the distance from the ray origin to sample point $i$ (i.e., the depth value at that sample)
  • $w_i$ = the volume rendering weight at sample point $i$
  • $D_{gt}$ = ground truth depth (where the actual surface is)

  In the surface concentration loss term: $$\sum_i \mathbb{1}(|h_i - D_{gt}| > \epsilon)|w_i|^2$$

  This means: "If sample point $i$ is farther than $\epsilon$ from the true surface, penalize having any weight there."

  This encourages the model to concentrate all the weight $w_i$ near the actual surface, rather than spreading it out along the ray.

  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  In the code (neural_feature_grid.py, lines 143-145):

  t_vals = torch.linspace(0, 1, S, device=device)
  sample_depths = depth_min + (depth_max - depth_min) * t_vals  # (S,)

  sample_depths is the array of $h_i$ values — the depths at which samples are taken along each ray.


  Great question! Yes, the equation handles exactly that scenario.

  The Physics: LiDAR vs. "Soft" Occluders

  Solid surface (wall, car):

  • $\alpha_i \approx 1$
  • All light is reflected/absorbed
  • Ray terminates completely

  Partial occlusion (dust, fog, rain):

  • $0 < \alpha_i < 1$
  • Some light passes through, some is scattered back
  • Ray continues with reduced intensity

  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  How the Math Handles Partial Occlusion

  Recall: $$w_i = \alpha_i \times \underbrace{\prod_{j=1}^{i-1}(1 - \alpha_j)}_{\text{transmittance}}$$

  Example: Dust cloud at 10m

   Sample   $\alpha_i$        Transmittance (reaching this point)   $w_i$
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1-9m     0 (empty)         100%                                  0
   10m      0.3 (dust)        100%                                  0.3 × 1.0 = 0.3
   11m      0 (behind dust)   70% (1-0.3)                           0 × 0.7 = 0
   15m      0.9 (wall)        70%                                   0.9 × 0.7 = 0.63

  Result: The model predicts:

  • Some return at 10m (dust reflection)
  • Stronger return at 15m (wall behind dust)

  The expected depth $D = \sum w_i h_i$ would be somewhere between 10m and 15m, weighted by the strength of each return.

  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Real-World LiDAR Behavior

  Multi-return LiDAR (common in autonomous driving):

  • Records multiple returns per pulse
  • 1st return: dust/fog
  • 2nd return: actual surface behind it

  The model mimics this:

  • Multiple points can have non-zero weights
  • The sum of all $w_i$ equals the probability the ray hits anything
  • If $\sum w_i < 1$, some light "escaped to infinity" (no hit)

  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Code Check

  In neural_feature_grid.py:

  alpha = torch.sigmoid(alpha)  # α_i ∈ (0, 1)

  # Transmittance = probability light reaches point i
  transmittance = torch.cumprod(
      torch.cat([ones, one_minus_alpha[..., :-1]], dim=-1),
      dim=-1,
  )

  weights = alpha * transmittance  # w_i

  Key insight: The cumprod of (1 - α) naturally handles the attenuation:

  • Each dust particle reduces the remaining light
  • By the time you reach the wall, intensity is reduced

  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Limitation in CoPilot4D

  The surface concentration loss assumes sharp surfaces: $$\text{Loss} = \sum_{i} \mathbb{1}(|h_i - D_{gt}| > \epsilon) \times w_i^2$$

  This penalizes spreading weight across multiple depths. It encourages:

  • All weight concentrated at ONE depth (like a solid wall)
  • Not distributed (like dust + wall)

  Trade-off: The model is optimized for clear surfaces, not for modeling volumetric scattering like heavy fog. For autonomous driving, this is usually desirable—you want to detect the a
  ctual obstacle, not the dust in front of it.

  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Summary

   Scenario                $\alpha_i$ pattern         What the model learns
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Clear air               0, then 1 at surface       Single sharp peak
   Dust in front of wall   0.2, 0.3, then 0.9         Multiple small peaks
   Thick fog               0.1, 0.1, 0.1... gradual   Broad distribution
   Glass/transparent       Low $\alpha$, then high    Passes through, hits behind