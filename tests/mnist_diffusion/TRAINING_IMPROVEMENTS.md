# Training Improvement Strategies

## 1. Model Architecture Changes

### Smaller Model (Recommended First)
```python
# Current (14.6M params)
embed_dim=384, num_layers=6, num_heads=6

# Small (1.8M params) - 8x faster, less overfitting
embed_dim=192, num_layers=4, num_heads=4

# Tiny (0.8M params) - For debugging
embed_dim=128, num_layers=4, num_heads=4
```

### Architecture Tweaks
- **Add dropout** (0.1) - Currently missing in the model
- **Use pre-LN** transformer - More stable training
- **Gradient checkpointing** - Fit larger batches

## 2. Training Recipe Changes

### Learning Rate Schedule
```python
# Current: Constant LR
# Better: Warmup + Cosine Decay
warmup_steps = 1000
cosine_steps = total_steps - warmup_steps
peak_lr = 3e-4
min_lr = 1e-6
```

### Objective Simplification
Current: 3 objectives (future 50%, joint 40%, individual 10%)
```python
# Try: Single objective first
objective = "future_only"  # 100% future prediction
# Then gradually add complexity
```

### Masking Strategy
```python
# Current: Random spatial masking
# Better: Progressive masking
step = epoch / total_epochs
mask_ratio = 0.5 + 0.5 * step  # Start easy (0.5), end hard (1.0)
```

## 3. Data Improvements

### Data Augmentation
```python
# For Moving MNIST:
- Random horizontal flip
- Small random rotations
- Brightness/contrast jitter
```

### Curriculum Learning
```python
# Start with shorter sequences
seq_len_schedule = [5, 10, 15, 20]  # Increase over epochs
```

## 4. Sampling Improvements

### Temperature Annealing
```python
# During sampling, use lower temperature
temperature = 0.8  # Instead of 1.0
# Sharper predictions, less noise
```

### Top-k Sampling
```python
# Instead of full softmax, use top-k
k = min(5, vocab_size)
# Prevents low-probability token selection
```

## 5. Debugging Checklist

If model still doesn't learn:

- [ ] **Overfit on single batch** - Should get near-zero loss
- [ ] **Check gradient flow** - Print max gradient norms
- [ ] **Verify tokenization** - Visualize tokens → frames
- [ ] **Simplify to frame-wise** - Remove temporal dimension first
- [ ] **Check loss computation** - Masked positions only

## Recommended Experiment Order

1. **Tiny model + single objective** (debug mode)
   - 128 dim, 2 layers
   - Future prediction only
   - Overfit on 100 samples

2. **Small model + full objective**
   - 192 dim, 4 layers
   - All 3 objectives
   - Full dataset

3. **Add improvements incrementally**
   - Add dropout
   - Add LR schedule
   - Add data augmentation

4. **Scale up if working**
   - 256 dim, 4-6 layers
   - Longer training
