# CoPilot4D World Model: Steps and Insights

## Overview

CoPilot4D is an unsupervised world model for autonomous driving that learns to predict future LiDAR point cloud observations. The world model operates on **discrete tokens** in Bird's-Eye View (BEV) space, using a novel discrete diffusion approach that recasts MaskGIT as a discrete diffusion model with key improvements.

**Key Innovation**: Instead of working directly on raw point clouds, the world model predicts the future by denoising discrete tokens through a discrete diffusion process, conditioned on past observations and ego vehicle actions.

---

## High-Level Pipeline (Figure 2)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         COPILOT4D PIPELINE                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  INPUT POINT CLOUDS          TOKENIZATION           WORLD MODEL PREDICTION      │
│       o(t)                      ↓                         ↓                      │
│        │                   ┌─────────┐               ┌──────────┐               │
│        │                   │  VQVAE  │               │ Discrete │               │
│        ▼                   │Tokenizer│               │ Diffusion│               │
│   ┌─────────┐              └────┬────┘               └────┬─────┘               │
│   │Encoder +│                   │                         │                     │
│   │   VQ    │─────► BEV Tokens x(t) ─────► Future Tokens x(t+1)                │
│   │Decoder  │      (discrete)        (denoising via    (discrete)               │
│   │(NeRF)   │                         parallel decoding)                        │
│   └─────────┘                   │                         │                     │
│                                 ▼                         ▼                     │
│                          ┌──────────┐              ┌──────────┐                │
│                          │Neural    │              │Tokenizer │                │
│                          │Feature   │              │Decoder   │                │
│                          │Grid      │              │(Volume   │                │
│                          │(NFG)     │              │Rendering)│                │
│                          └────┬─────┘              └────┬─────┘                │
│                               │                         │                       │
│                               ▼                         ▼                       │
│                         RECONSTRUCTED             PREDICTED                     │
│                          POINT CLOUD              POINT CLOUD                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Tokenization (VQVAE)

**Reference Figure**: `docs/figures/page15_fig7-15.png` (top half - Figure 6)

Before the world model can operate, raw LiDAR point clouds must be converted into discrete tokens.

### Tokenization Steps:
1. **Voxelization**: Points are grouped into voxels (15.625cm × 15.625cm × 14.0625cm)
2. **PointNet Encoding**: Each voxel's points are encoded using a modified PointNet (sum + LayerNorm instead of max pooling)
3. **BEV Projection**: 3D feature volume (1024×1024×64×64) is collapsed to 2D BEV via z-axis pooling
4. **Swin Transformer**: Spatial features are processed with a Swin Transformer backbone
5. **Vector Quantization**: Encoder output z is quantized to ẑ using a learned codebook V
6. **Decoder with Neural Feature Grid (NFG)**: The decoder outputs a 3D neural feature grid
7. **Differentiable Depth Rendering**: Point clouds are reconstructed via NeRF-style volume rendering

**Output**: Tokenized observation x(t) ∈ {0, ..., |V|-1}^N where N is number of tokens (e.g., 128×128 = 16,384)

---

## Step 2: World Model Architecture (Figure 7)

**Reference Figure**: `docs/figures/page15_fig7-15.png` (bottom half - Figure 7)

The world model is a **spatio-temporal U-Net Transformer** that interleaves spatial and temporal attention blocks.

### Architecture Components:

**From the Paper Figure 7:**
- Shows U-Net based Transformer with 3 spatial resolution levels (32×32, 64×64, 128×128)
- Interleaves spatial (Swin) and temporal blocks
- Actions (ego poses) are injected at each level
- Outputs logits of shape (128, 128, 1024) over vocabulary

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    WORLD MODEL ARCHITECTURE (Figure 7)                           │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  INPUT: BEV Tokens from Past Observations + Actions                              │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    U-NET STRUCTURE (3 Levels)                           │    │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐                             │    │
│  │  │ Level 1 │───►│ Level 2 │───►│ Level 3 │  (Encoder - Downsampling)   │    │
│  │  │128×128  │    │ 64×64   │    │ 32×32   │                             │    │
│  │  └────┬────┘    └────┬────┘    └────┬────┘                             │    │
│  │       │              │              │                                   │    │
│  │       └──────────────┴──────────────┘                                   │    │
│  │                      │                                                  │    │
│  │                 Bottleneck                                             │    │
│  │                      │                                                  │    │
│  │       ┌──────────────┴──────────────┐                                   │    │
│  │       │              │              │                                   │    │
│  │  ┌────┴────┐    ┌────┴────┐    ┌────┴────┐  (Decoder - Upsampling)     │    │
│  │  │ Level 1 │◄───│ Level 2 │◄───│ Level 3 │                             │    │
│  │  │128×128  │    │ 64×64   │    │ 32×32   │                             │    │
│  │  └─────────┘    └─────────┘    └─────────┘                             │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  AT EACH LEVEL:                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐      │
│  │  1. Add Ego Action (pose) to each frame via Linear projection         │      │
│  │                                                                       │      │
│  │  2. SPATIAL ATTENTION (Swin Transformer Blocks)                       │      │
│  │     └─► Windowed self-attention within each frame                     │      │
│  │     └─► Shifted window mechanism for cross-window connections         │      │
│  │     └─► 2 Swin Blocks per level                                       │      │
│  │                                                                       │      │
│  │  3. TEMPORAL ATTENTION (GPT2 Blocks)                                  │      │
│  │     └─► Causal attention across time for same spatial location        │      │
│  │     └─► Each position attends to previous time steps                  │      │
│  │                                                                       │      │
│  │  4. Residual Connections between encoder and decoder levels           │      │
│  └───────────────────────────────────────────────────────────────────────┘      │
│                                                                                  │
│  OUTPUT: Logits over vocabulary (128×128×1024) for next frame prediction         │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Choices:
- **Spatial Attention**: Swin Transformer captures spatial relationships within each BEV frame
- **Temporal Attention**: GPT2-style causal attention models temporal dynamics
- **Action Conditioning**: Ego vehicle poses are injected at each feature level
- **U-Net Structure**: Multi-scale processing with skip connections for better gradient flow

---

## Step 3: Training Objectives (Figure 4)

The world model is trained with a **mixture of three objectives**:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    TRAINING OBJECTIVES (Figure 4)                                │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  OBJECTIVE 1: CONDITION ON PAST, PREDICT FUTURE (50% of training)                │
│  ┌────────────────────────────────────────────────────────────────────────┐     │
│  │  Input:  [x(1), x(2), x(3), x(4)] + [a(1), a(2), a(3)]                 │     │
│  │  Mask:   [  ✓  ,  ✓  ,  ✓  ,  ✗  ]  (future masked)                    │     │
│  │  Target:                    x(4) (predict future)                      │     │
│  │                                                                       │     │
│  │  Temporal Mask: Causal (past frames can attend to earlier past)       │     │
│  └────────────────────────────────────────────────────────────────────────┘     │
│                                                                                  │
│  OBJECTIVE 2: JOINT MODELING OF PAST AND FUTURE (40% of training)                │
│  ┌────────────────────────────────────────────────────────────────────────┐     │
│  │  Input:  [x(1), x(2), x(3), x(4)] + [a(1), a(2), a(3)]                 │     │
│  │  Mask:   [  ✗  ,  ✗  ,  ✗  ,  ✗  ]  (all frames masked)                │     │
│  │  Target: [x(1), x(2), x(3), x(4)] (reconstruct all)                    │     │
│  │                                                                       │     │
│  │  Purpose: Harder pretraining task; ensures accurate predictions        │     │
│  │           even with imperfect past conditioning                        │     │
│  └────────────────────────────────────────────────────────────────────────┘     │
│                                                                                  │
│  OBJECTIVE 3: MODEL EACH FRAME INDIVIDUALLY (10% of training)                    │
│  ┌────────────────────────────────────────────────────────────────────────┐     │
│  │  Input:  [x(1), x(2), x(3), x(4)]                                       │     │
│  │  Mask:   [  ✗  ,  ✗  ,  ✗  ,  ✗  ]                                     │     │
│  │  Target: [x(1), x(2), x(3), x(4)]                                       │     │
│  │                                                                       │     │
│  │  Temporal Mask: IDENTITY (each frame attends only to itself)           │     │
│  │  Purpose: Enables classifier-free diffusion guidance (unconditional)   │     │
│  └────────────────────────────────────────────────────────────────────────┘     │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Discrete Diffusion Training (Algorithm 1):

```
TRAINING PROCEDURE (per token sequence x₀):
─────────────────────────────────────────
1. Sample u₀ ~ Uniform(0, 1)
2. Mask ⌈γ(u₀)·N⌉ tokens (γ(u) = cos(uπ/2) is mask schedule)
3. Sample u₁ ~ Uniform(0, 1)
4. Add uniform noise to (u₁·η)% of remaining tokens (η = 20)
5. Get masked-and-noised xₖ
6. Train model to maximize: log p_θ(x₀ | xₖ) with cross-entropy
```

**Key Insight**: Unlike MaskGIT, this approach adds uniform noise to unmasked tokens, making it a proper discrete diffusion model with a well-defined forward process.

---

## Step 4: Discrete Diffusion Sampling (Algorithm 2 & Figure 11)

**Reference Figure**: `docs/figures/page20_fig10_11-20.png` (bottom half - Figure 11)

During inference, the world model generates future frames through iterative denoising:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│              DISCRETE DIFFUSION SAMPLING (Figure 11)                             │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  INITIAL STATE: x_K = [MASK, MASK, MASK, ..., MASK] (all N tokens masked)       │
│                                                                                  │
│  FOR k = K-1 down to 0 (iterative denoising):                                   │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                                                                           │  │
│  │  STEP 1: PREDICT logits p_θ(x̃₀ | x_{k+1})                                 │  │
│  │          └─► World model forward pass on current masked tokens            │  │
│  │                                                                           │  │
│  │  STEP 2: ADD GUMBEL NOISE to logits for stochastic sampling               │  │
│  │          l_k = log p_θ(x̃₀ | x_{k+1}) + Gumbel(0,1) · (k/K)                │  │
│  │          └─► Temperature decreases as k decreases (less noise later)      │  │
│  │                                                                           │  │
│  │  STEP 3: PRESERVE already-decoded tokens                                  │  │
│  │          Set l_k = +∞ at non-mask indices of x_{k+1}                      │  │
│  │          └─► Already-sampled tokens are locked in                         │  │
│  │                                                                           │  │
│  │  STEP 4: DETERMINE how many tokens to unmask at this step                 │  │
│  │          M = ⌈γ(k/K) · N⌉                                                 │  │
│  │          └─► Cosine schedule: more tokens early, fewer later              │  │
│  │                                                                           │  │
│  │  STEP 5: SAMPLE top-M tokens by l_k                                       │  │
│  │          x_k ← x̃₀ on top-M indices of l_k                                 │  │
│  │          └─► Highest confidence tokens get unmasked                       │  │
│  │                                                                           │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
│  OUTPUT: x₀ (fully decoded token sequence)                                       │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Key Algorithm Insights (Figure 11):

```
DIFFUSION STEP k:
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│   INPUT: x_{k+1} (partially masked from previous step)        │
│                                                                │
│        ┌─────────────┐                                         │
│        │  World      │                                         │
│        │  Model      │────► Logits p_θ(x̃₀ | x_{k+1})          │
│        │             │        (128×128×1024)                   │
│        └─────────────┘                                         │
│              │                                                 │
│              ▼                                                 │
│   ┌─────────────────────┐                                      │
│   │  + Gumbel Noise     │                                      │
│   │  × (k/K)            │────► Noised Logits                   │
│   └─────────────────────┘                                      │
│              │                                                 │
│              ▼                                                 │
│   ┌─────────────────────┐                                      │
│   │  Set non-mask locs  │                                      │
│   │  to -∞ (preserve)   │────► Modified Logits                 │
│   └─────────────────────┘                                      │
│              │                                                 │
│              ▼                                                 │
│   ┌─────────────────────┐                                      │
│   │  Sort by confidence │                                      │
│   │  + Top-M selection  │────► Which M tokens to unmask        │
│   └─────────────────────┘      (M = γ(k/K)·N)                  │
│              │                                                 │
│              ▼                                                 │
│   OUTPUT: x_k (fewer masks than x_{k+1})                       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Step 5: Classifier-Free Diffusion Guidance (CFG) - Figure 10

**Reference Figure**: `docs/figures/page20_fig10_11-20.png` (top half - Figure 10)

CFG significantly improves prediction quality by amplifying the influence of conditioning information:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│           CLASSIFIER-FREE DIFFUSION GUIDANCE (Figure 10)                         │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  CFG FORMULA:                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                         │    │
│  │  logit_cfg = logits_cond + w × (logits_cond - logits_uncond)           │    │
│  │                                                                         │    │
│  │  Where:                                                                 │    │
│  │  • logits_cond = p_θ(x₀ | x_{k+1}, c)   [conditioned on past]          │    │
│  │  • logits_uncond = p_θ(x₀ | x_{k+1})    [unconditioned]                │    │
│  │  • w = guidance scale (typically 2.0)                                   │    │
│  │                                                                         │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  INTUITION: Amplify the difference between conditional and unconditional        │
│             predictions to strengthen the effect of conditioning                │
│                                                                                  │
│  IMPLEMENTATION (Single Forward Pass):                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                         │    │
│  │  TIMESTEPS:    [t-3]    [t-2]    [t-1]    [t]     [t+1]                │    │
│  │  Tokens:       x(1)     x(2)     x(3)     x(4)    [MASK]               │    │
│  │                                                                         │    │
│  │  Sequence length increased by 1 for the frame being generated           │    │
│  │                                                                         │    │
│  │  ATTENTION MASK STRUCTURE:                                              │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │    │
│  │  │                 PAST (t-3 to t)      FUTURE (t+1)               │   │    │
│  │  │                                                                 │   │    │
│  │  │  Past frames    [CAUSAL MASK]        [CAN ATTEND]  ◄── conditional│   │    │
│  │  │  (can attend to earlier past)                                   │   │    │
│  │  │                                                                 │   │    │
│  │  │  Future frame   [IDENTITY MASK]      [SELF ONLY]   ◄── unconditional│  │    │
│  │  │  (can only attend to itself)        (no past conditioning)      │   │    │
│  │  │                                                                 │   │    │
│  │  └─────────────────────────────────────────────────────────────────┘   │    │
│  │                                                                         │    │
│  │  KEY: The last frame has IDENTITY mask → becomes unconditional          │    │
│  │       This allows computing both cond and uncond in ONE forward pass!   │    │
│  │                                                                         │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  CACHING OPTIMIZATION:                                                           │
│  • For all past timesteps, only need cached keys and values from temporal blocks │
│  • No need to recompute past activations at each diffusion step                 │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### CFG Impact (from paper Table 3):
- **Without CFG (w=0.0)**: Chamfer Distance = 1.40
- **With CFG (w=2.0)**: Chamfer Distance = 0.49
- **Improvement**: ~65% reduction in prediction error!

---

## Inference: Autoregressive Future Prediction

During inference, the world model predicts future frames autoregressively:

```
AUTOREGRESSIVE PREDICTION (e.g., 3 seconds into future):
────────────────────────────────────────────────────────

TIME:     t=0 (now)          t=1 (0.5s)        t=2 (1.0s)        t=3 (1.5s)...
          │                    │                  │                 │
          ▼                    ▼                  ▼                 ▼
┌─────────────┐          ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  x(0)       │          │  x(1)       │    │  x(2)       │    │  x(3)       │
│ (observed)  │          │ (predicted) │    │ (predicted) │    │ (predicted) │
└──────┬──────┘          └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                        │                  │                 │
       │         ┌──────────────┘                  │                 │
       │         │  ▲                              │                 │
       │         │  │ (use as past                 │                 │
       │         │   for next pred)               │                 │
       │         │                                 │                 │
       └────────►├─────────────────────────────────┘                 │
                 │    Discrete Diffusion with CFG                     │
                 │    (K steps, typically 12-32)                      │
                 │                                                   │
                 └───────────────────────────────────────────────────┘

INPUTS at each step:
  • Past observations: x(0), x(1), ..., x(t)
  • Past actions: a(0), a(1), ..., a(t)

OUTPUT:
  • Next observation tokens: x(t+1) (via discrete diffusion sampling)
  
DECODE:
  • Pass tokens through Tokenizer Decoder → Neural Feature Grid
  • Volume render → Predicted point cloud
```

---

## Key Insights Summary

### 1. Tokenization Enables Scalable World Modeling
- **Problem**: Raw point clouds are unstructured, variable-sized, and high-dimensional
- **Solution**: VQVAE tokenization compresses point clouds into compact discrete BEV tokens
- **Benefit**: World model operates on fixed-size discrete sequences (like GPT on text)

### 2. Discrete Diffusion > MaskGIT for World Modeling
- **Key Modification**: Add uniform noise (η=20%) to unmasked tokens during training
- **Benefit**: Creates well-defined forward diffusion process with proper posterior
- **Result**: Single model achieves what previously required two separate models

### 3. Spatio-Temporal Factorization is Crucial
- **Spatial (Swin)**: Captures within-frame relationships (objects, roads, etc.)
- **Temporal (GPT2)**: Captures across-time dynamics (motion, ego movement)
- **U-Net**: Multi-scale processing with residuals for stable gradients

### 4. Mixed Training Objectives Improve Generalization
- **50% Future Prediction**: Primary task of predicting next frame
- **40% Joint Modeling**: Harder task improves representation quality
- **10% Individual Frame**: Enables CFG for better conditional generation

### 5. Classifier-Free Guidance is Essential
- **Mechanism**: Amplify difference between conditional and unconditional predictions
- **Implementation**: Clever attention masking allows single forward pass
- **Impact**: ~65% improvement in prediction accuracy

### 6. Efficient Inference via Caching
- **Key Insight**: Past frame keys/values can be cached across diffusion steps
- **Benefit**: Only compute new activations for the frame being generated
- **Result**: Faster autoregressive generation

---

## Performance Results

| Dataset | Metric | Prior SOTA | CoPilot4D | Improvement |
|---------|--------|------------|-----------|-------------|
| NuScenes | Chamfer @ 1s | 1.40 | **0.49** | 65% ↓ |
| KITTI | Chamfer @ 1s | 1.23 | **0.41** | 67% ↓ |
| Argoverse2 | Chamfer @ 1s | 2.13 | **0.74** | 65% ↓ |
| NuScenes | Chamfer @ 3s | 2.58 | **1.29** | 50% ↓ |

---

## Architecture Parameters (Reference)

| Component | Configuration |
|-----------|---------------|
| Token Grid Size | 128 × 128 = 16,384 tokens |
| Vocabulary Size | 1,024 or 8,192 codebook entries |
| Spatial Backbone | Swin Transformer (Swin-T or Swin-B) |
| Temporal Backbone | GPT2 blocks |
| U-Net Levels | 3 (128→64→32→64→128) |
| Attention Heads | 4-8 per block |
| Training Steps | 200,000 - 750,000 |
| Diffusion Steps (K) | 12-32 during inference |
| CFG Scale (w) | 2.0 |

---

## Figure References

All figures are extracted from the CoPilot4D paper (ICLR 2024) and saved in `docs/figures/`:

| Figure | File | Description |
|--------|------|-------------|
| Figure 6 | `page15_fig7-15.png` (top) | Point Cloud Tokenizer Architecture |
| Figure 7 | `page15_fig7-15.png` (bottom) | U-Net Transformer World Model |
| Figure 10 | `page20_fig10_11-20.png` (top) | Classifier-Free Diffusion Guidance (CFG) |
| Figure 11 | `page20_fig10_11-20.png` (bottom) | Discrete Diffusion Sampling |

## References

- Paper: "CoPilot4D: Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion" (ICLR 2024)
- arXiv: 2311.01017v4 [cs.CV]
- Key Methods: VQVAE, Swin Transformer, GPT2, MaskGIT, Discrete Diffusion, Classifier-Free Guidance
- Datasets: NuScenes, KITTI Odometry, Argoverse2
- PDF Location: `papers/CoPilot4D.pdf`
