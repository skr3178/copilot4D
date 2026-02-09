# CoPilot4D World Model Architecture

**U-Net Spatio-Temporal Transformer** for discrete BEV token prediction with action conditioning.

## Overview

```
Input: Discrete BEV tokens (B, T, 128, 128) + Ego actions (B, T, 16)
                         ↓
Output: Logits (B, T, 16384, 1025) for discrete diffusion

Architecture: 3-level U-Net (128×128 → 64×64 → 32×32 → 64×64 → 128×128)
```

---

## 1. Input Embeddings

```
token_indices: (B, T, 16384)          actions: (B, T, 16)
       │                                        │
       ▼                                        ▼
┌─────────────────┐                    ┌─────────────────┐
│ Token Embedding │ 256-dim            │ Action Projection
│  (1025 → 256)   │                    │  (16 → 256)     │
└────────┬────────┘                    └────────┬────────┘
         │                                      │
         ▼                                      ▼
┌─────────────────┐                             │
│ Linear→LN→Linear│                             │
└────────┬────────┘                             │
         │                                      │
         ▼                                      ▼
┌─────────────────────────────────────────────────────────────┐
│  x = token_embed + spatial_pos + temporal_pos + action_bias  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Output: (B, T, 16384, 256)
```

| Component | Shape | Notes |
|-----------|-------|-------|
| Token embedding | (1025, 256) | Discrete BEV codebook + mask token |
| Spatial position | (1, 1, 16384, 256) | Learnable, shared across frames |
| Temporal position | (1, T_max, 1, 256) | Learnable, broadcast to spatial |
| Action projection | (16 → 256 → 256) | Linear→LN→Linear, no bias |

---

## 2. Spatio-Temporal Block (Inset Fig. 7)

```
Input: (B, T, N, C)    where N = H×W, temporal_mask: (T, T)
              │
              ▼
    ┌─────────────────┐
    │   Reshape to    │  (B×T, N, C) ──► per-frame spatial processing
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  Swin Block 1   │  Window attention, shift=0
    │  (local window) │  num_heads, window_size
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  Swin Block 2   │  Shifted window, shift=window_size//2
    │ (cross-window)  │  Enables global receptive field
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │   Reshape to    │  (B, T, N, C)
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  Temporal Block │  Causal attention across T at each spatial loc
    │   (B×N, T, C)   │  GPT-2 style: LN→MHA→LN→MLP
    └────────┬────────┘
             │
             ▼
    Output: (B, T, N, C)
```

**Per-level configurations:**

| Level | Resolution | Dim | Heads | Window Size | ST-Blocks |
|-------|------------|-----|-------|-------------|-----------|
| L1 | 128×128 | 256 | 8 | 8×8 | 2 (enc), 2 (dec) |
| L2 | 64×64 | 384 | 12 | 8×8 | 2 (enc), 1 (dec) |
| L3 | 32×32 | 512 | 16 | 16×16 | 1 (bottleneck) |

---

## 3. Full Architecture

### Encoder Path

```
Level 1 (128×128, 256-dim)              Level 2 (64×64, 384-dim)
┌─────────────────────────┐             ┌─────────────────────────┐
│ Input: (B,T,16384,256)  │             │ Input: (B,T,4096,384)   │
│                         │             │                         │
│ ┌─────────────────────┐ │             │ ┌─────────────────────┐ │
│ │ SpatioTemporalBlock │ │             │ │ SpatioTemporalBlock │ │
│ │ heads=8, ws=8       │ │             │ │ heads=12, ws=8      │ │
│ └─────────────────────┘ │             │ └─────────────────────┘ │
│ ┌─────────────────────┐ │             │ ┌─────────────────────┐ │
│ │ SpatioTemporalBlock │ │             │ │ SpatioTemporalBlock │ │
│ │ heads=8, ws=8       │ │             │ │ heads=12, ws=8      │ │
│ └─────────────────────┘ │             │ └─────────────────────┘ │
│           │             │             │           │             │
│      skip_l1            │             │      skip_l2            │
│           │             │             │           │             │
│           ▼             │             │           ▼             │
│ ┌─────────────────────┐ │             │ ┌─────────────────────┐ │
│ │   Patch Merging     │ │             │ │   Patch Merging     │ │
│ │  2×2 grouping       │ │             │ │  2×2 grouping       │ │
│ │  4×256 → 384        │ │             │ │  4×384 → 512        │ │
│ │  128² → 64²         │ │             │ │  64² → 32²          │ │
│ └─────────────────────┘ │             │ └─────────────────────┘ │
└─────────────────────────┘             └─────────────────────────┘
          │                                       │
          └─────────────────┬─────────────────────┘
                            ▼
              Level 3 / Bottleneck (32×32, 512-dim)
              ┌─────────────────────────┐
              │ Input: (B,T,1024,512)   │
              │                         │
              │ ┌─────────────────────┐ │
              │ │ SpatioTemporalBlock │ │
              │ │ heads=16, ws=16     │ │
              │ └─────────────────────┘ │
              └─────────────────────────┘
```

### Decoder Path

```
Bottleneck Output: (B,T,1024,512)
              │
              ▼
┌─────────────────────────────────────────┐
│           Level Merging 2               │
│  ┌─────────────────────────────────┐    │
│  │  Upsample: ConvTranspose2d(2,2) │    │
│  │  (B×T,512,32,32) → (B×T,384,64,64)  │
│  └─────────────────────────────────┘    │
│              │                          │
│  Concat[upsampled, skip_l2] ──► (2×384) │
│              │                          │
│  LN → Linear(2×384 → 384) + residual    │
└─────────────────────────────────────────┘
              │
              ▼ (B,T,4096,384)
┌─────────────────────────────────────────┐
│  SpatioTemporalBlock (heads=12, ws=8)   │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│           Level Merging 1               │
│  ┌─────────────────────────────────┐    │
│  │  Upsample: ConvTranspose2d(2,2) │    │
│  │  (B×T,384,64,64) → (B×T,256,128,128) │
│  └─────────────────────────────────┘    │
│              │                          │
│  Concat[upsampled, skip_l1] ──► (2×256) │
│              │                          │
│  LN → Linear(2×256 → 256) + residual    │
└─────────────────────────────────────────┘
              │
              ▼ (B,T,16384,256)
┌─────────────────────────────────────────┐
│  SpatioTemporalBlock (heads=8, ws=8)    │
│  SpatioTemporalBlock (heads=8, ws=8)    │
└─────────────────────────────────────────┘
              │
              ▼
        LayerNorm(256)
              │
              ▼
┌─────────────────────────────────────────┐
│   Weight-Tied Linear Layer              │
│   Uses token_embedding.weight: (256,1025)│
│   + learned bias: (1025,)               │
└─────────────────────────────────────────┘
              │
              ▼
    Output: (B, T, 16384, 1025)
```

---

## 4. Patch Merging (Encoder Downsampling)

```
Input: (B, H×W, C)     H = W = input_resolution
           │
           ▼
┌──────────────────────┐
│ Reshape: (B, H, W, C)│
└──────────┬───────────┘
           │
    2×2 patch grouping
    ┌──────┴──────┐
    ▼             ▼
x[:,0::2,0::2,:] x[:,1::2,0::2,:]   top-left, bottom-left
x[:,0::2,1::2,:] x[:,1::2,1::2,:]   top-right, bottom-right
    │             │
    └──────┬──────┘
           ▼
  Concatenate: (B, H/2, W/2, 4×C)
           │
           ▼
    LayerNorm(4×C)
           │
           ▼
    Linear(4×C → target_dim)
           │
           ▼
  Reshape: (B, H/2 × W/2, target_dim)
```

| Stage | Input Res | Input Dim | Output Res | Output Dim |
|-------|-----------|-----------|------------|------------|
| Merge 1 | 128×128 | 256 | 64×64 | 384 |
| Merge 2 | 64×64 | 384 | 32×32 | 512 |

---

## 5. Level Merging (Decoder Upsampling)

```
x_up: (B, T, N_up, C_up)     x_skip: (B, T, N_skip, C_skip)
         │                              │
         │    N_skip = 4×N_up (2×2 upsampling)
         │    C_skip = C_out
         │
         ▼
┌──────────────────────────────────────┐
│ Reshape per frame: (B×T, Hin, Win, Cin) │
│ Permute: (B×T, Cin, Hin, Win)        │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ ConvTranspose2d(Cin, Cout, k=2, s=2) │
│ Output: (B×T, Cout, Hout, Wout)      │
│ where Hout = 2×Hin, Wout = 2×Win     │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ Permute → Reshape: (B, T, N_skip, Cout) │
└──────────────┬───────────────────────┘
               │
               ▼
        Concat with x_skip
               │
    (B, T, N_skip, Cout + C_skip)
               │
               ▼
    LayerNorm(Cout + C_skip)
               │
               ▼
    Linear(Cout + C_skip → Cout)
               │
               ▼
         + x_skip (residual)
               │
               ▼
        Output: (B, T, N_skip, Cout)
```

| Stage | x_up Shape | x_skip Shape | Output Shape |
|-------|------------|--------------|--------------|
| Merge 2 | (B,T,1024,512) | (B,T,4096,384) | (B,T,4096,384) |
| Merge 1 | (B,T,4096,384) | (B,T,16384,256) | (B,T,16384,256) |

---

## 6. Action Interleaving

```
Raw actions: SE(3) transforms between consecutive frames
             Stored as flattened (16,) vectors per frame

Processing:
┌─────────────────────────────────────────────────────────────┐
│  actions: (B, T, 16)                                        │
│      │                                                      │
│      ▼                                                      │
│  Linear(16 → 256) → LayerNorm → Linear(256 → 256)          │
│      │                                                      │
│      ▼                                                      │
│  action_embed: (B, T, 256)                                  │
│      │                                                      │
│      ▼ broadcast to all spatial positions                   │
│  unsqueeze(2): (B, T, 1, 256)                               │
│      │                                                      │
│      ▼                                                      │
│  Added to token embeddings at every spatial location        │
│  x = x_token + x_spatial + x_temporal + x_action            │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Summary Table

| Component | Config | Input Shape | Output Shape |
|-----------|--------|-------------|--------------|
| **Embedding** | dim=256 | tokens (B,T,16384), actions (B,T,16) | (B,T,16384,256) |
| **Encoder L1** | dim=256, heads=8, ws=8, 2 blocks | (B,T,16384,256) | (B,T,16384,256) |
| **Patch Merge 1** | 4×256→384 | (B,T,16384,256) | (B,T,4096,384) |
| **Encoder L2** | dim=384, heads=12, ws=8, 2 blocks | (B,T,4096,384) | (B,T,4096,384) |
| **Patch Merge 2** | 4×384→512 | (B,T,4096,384) | (B,T,1024,512) |
| **Bottleneck L3** | dim=512, heads=16, ws=16, 1 block | (B,T,1024,512) | (B,T,1024,512) |
| **Level Merge 2** | upsample 512→384 | x_up (B,T,1024,512), skip (B,T,4096,384) | (B,T,4096,384) |
| **Decoder L2** | dim=384, heads=12, ws=8, 1 block | (B,T,4096,384) | (B,T,4096,384) |
| **Level Merge 1** | upsample 384→256 | x_up (B,T,4096,384), skip (B,T,16384,256) | (B,T,16384,256) |
| **Decoder L1** | dim=256, heads=8, ws=8, 2 blocks | (B,T,16384,256) | (B,T,16384,256) |
| **Output Head** | weight-tied | (B,T,16384,256) | (B,T,16384,1025) |

**Total ST-blocks:** 2+2+1 (encoder) + 1+2 (decoder) = **8 blocks**

---

## 8. Initialization

```python
# Appendix A.3: Residual scaling by √(1/L)
# where L = total number of sub-layers (8 ST-blocks × 3 sub-layers each = 24)
scale = sqrt(1.0 / 24)

# Applied to residual branches in all Linear layers
```
