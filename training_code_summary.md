# Training Code Summary - CoPilot4D on Moving MNIST

## 1. MODEL ARCHITECTURE (simple_model.py)

```python
SimpleVideoTransformer(
    vocab_size=17,          # 16 token levels + 1 mask token
    mask_token_id=16,       # ID for masked tokens
    num_frames=20,          # Sequence length
    height=32, width=32,    # Frame size (1024 tokens)
    embed_dim=128,          # Embedding dimension
    num_layers=2,           # Transformer layers
    num_heads=4,            # Attention heads
    action_dim=2,           # [dx, dy] ego-centric actions
)
```

**Key Components:**
- Token embedding (vocab_size → embed_dim)
- Action projection (2 → embed_dim) 
- Spatial-Temporal Transformer layers
- Each layer: Spatial Attention → Temporal Attention → FFN
- Output projection (embed_dim → vocab_size)

## 2. DATASET (moving_mnist_cached.py)

```python
MovingMNISTCached(
    data_path='mnist_test_seq.1.npy',  # (20, 10000, 64, 64)
    seq_len=20,
    num_sequences=2000,     # Training subset
    frame_size=32,          # Downsampled from 64
    num_token_levels=16,    # Quantization levels
    use_ego_centric=True,   # Track single digit
    ego_digit_id=0,         # Track largest digit
    cache_dir='data/mnist_cache',  # Pre-computed actions
)
```

**Action Generation (Ego-centric):**
```python
# Track center of single digit between frames
dx = (center_x[t] - center_x[t-1]) / (W/2)  # Normalize to [-1, 1]
dy = (center_y[t] - center_y[t-1]) / (H/2)
action = [dx, dy]  # Continuous motion vector
```

## 3. TRAINING LOOP (train_mnist_fast.py)

### Training Configuration
```python
epochs = 50
batch_size = 4
learning_rate = 3e-4
optimizer = AdamW(lr=3e-4, weight_decay=0.01)
scheduler = CosineAnnealingLR(eta_min=1e-6)
num_past_frames = 10  # Conditioning frames
```

### Forward Pass (Future Prediction)
```python
# Input: tokens (B, T, H, W), actions (B, T, 2)
num_past = 10
mask_ratio = random(0.5, 1.0)

# Mask FUTURE tokens only
masked_tokens = tokens.clone()
mask = rand(B, T-num_past, H, W) < mask_ratio
masked_tokens[:, num_past:][mask] = MASK_TOKEN

# Causal temporal mask (autoregressive)
causal_mask = torch.triu(-inf, diagonal=1)  # (T, T)

# Forward
logits = model(masked_tokens, actions, causal_mask)  # (B, T, N, V)

# Loss on FUTURE only
loss = CrossEntropy(logits[:, num_past:], tokens[:, num_past:])
```

### Key Features
- **Mixed precision** (torch.amp.autocast)
- **Gradient clipping** (max_norm=1.0)
- **Cosine LR schedule**
- **Checkpoint saving**: Every 10 epochs + best model

## 4. SAMPLING/GENERATION (generate_validation_results.py)

### Future Prediction (Conditional)
```python
# Given: past_frames (10 frames)
# Predict: future_frames (10 frames)

# Start with all-masked future
masked_tokens = [past_tokens] + [MASK] * 10

# Iterative denoising (cosine schedule)
for k in reversed(range(K)):  # K=20 steps
    # Predict all tokens
    logits = model(masked_tokens, actions, causal_mask)
    
    # Sample from logits
    pred_tokens = sample(logits)
    
    # Mask scheduling: keep some, remask others
    num_keep = ceil(cosine_schedule(k/K) * N)
    masked_tokens = remask_strategy(pred_tokens, num_keep)

return predicted_future
```

### Metrics Computed
- **MSE**: Mean Squared Error (pixel space)
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index

## 5. KEY FILE STRUCTURE

```
CoPilot4D/
├── tests/mnist_diffusion/
│   ├── simple_model.py              # Model architecture
│   ├── moving_mnist_cached.py       # Dataset with caching
│   ├── train_mnist_fast.py          # Training script
│   ├── generate_validation_results.py  # Evaluation
│   └── train_mnist_full.py          # Full training (3 objectives)
├── outputs/mnist_diffusion_fast/
│   ├── best_model.pt                # Best checkpoint (7.7 MB)
│   ├── checkpoint_epoch10.pt        # Periodic checkpoint
│   └── results_epoch10/             # Generated samples
│       ├── future_pred/             # Conditional prediction results
│       └── generation/              # Unconditional generation
└── data/mnist_cache/
    └── actions_ego0_fs32_0_2000.pkl # Pre-computed actions
```

## 6. TRAINING OBJECTIVES

### Fast Training (Current)
- **Single objective**: Future prediction only (past → future)
- **Simpler**: No joint/individual denoising
- **Faster convergence**: Focused task

### Full Training (Alternative)
- **Three objectives**:
  1. Future prediction (50%)
  2. Joint denoising (40%) - mask random frames bidirectionally
  3. Individual denoising (10%) - single frame

## 7. HYPERPARAMETERS SUMMARY

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model size | 0.67M params | 128 dim, 2 layers |
| Checkpoint | 7.7 MB | With optimizer state |
| Data | 2000 train / 500 val | Cached subset |
| Input | 32×32 frames | 1024 tokens per frame |
| Tokens | 16 levels | Discrete quantization |
| Actions | 2D [dx, dy] | Ego-centric continuous |
| Training | 50 epochs | ~3 min/epoch |
| Batch size | 4 | 8.7 GB GPU memory |

## 8. KEY DIFFERENCES FROM FULL MODEL

| Aspect | Fast (Current) | Full |
|--------|---------------|------|
| Parameters | 0.67M | 14.6M |
| Layers | 2 | 6 |
| Embed dim | 128 | 384 |
| Objectives | 1 (future only) | 3 (future+joint+indiv) |
| Data | 2000 seq | 8000 seq |
| Checkpoint | 7.7 MB | 223 MB |
| Training time | ~2 hours | ~8 hours |
