# Moving MNIST Training Fixes - Summary

## ✅ All Fixes Implemented

### 1. Ego-Centric Actions (Fixed)
**File:** `moving_mnist_precomputed.py`

**Problem:** Original actions averaged motion of both digits → ambiguous signal

**Solution:** Track single digit motion as continuous [dx, dy]

```python
# New method: _generate_actions_ego_centric()
# Returns: (N, T, 2) array with normalized displacement [-1, 1]

# Usage:
dataset = MovingMNISTPrecomputed(
    use_ego_centric=True,  # Enable ego-centric
    ego_digit_id=0,        # Track largest digit (0) or second (1)
)
```

**Verification:**
- Action std: 0.186 (good variance)
- Range: [-0.595, +0.565] (normalized displacement)

---

### 2. Joint Modeling Mask (Fixed)
**File:** `train_mnist_full.py` (DiscreteDiffusionMasker)

**Problem:** Joint denoising was using causal mask instead of bidirectional

**Solution:** Use `None` (no mask) for joint denoising objective

```python
elif objective == "joint_denoise":
    # Bidirectional: all frames attend to all frames
    temporal_mask = None  # Changed from _make_causal_mask()
```

**Added method:**
```python
@staticmethod
def _make_bidirectional_mask(T, device):
    """All-zeros mask (all frames attend to all frames)."""
    return torch.zeros((T, T), device=device)
```

---

### 3. Model Architecture Verified
**File:** `simple_model.py`

**Action injection works correctly:**
```python
# Actions projected and added to all spatial positions
action_emb = self.action_proj(actions)  # (B, T, D)
x = x + action_emb[:, :, None, :]       # Broadcast to (B, T, N, D)
```

---

## Verification Results

Run: `python tests/mnist_diffusion/verify_fixes.py --ego --steps 100`

```
============================================================
VERIFICATION SUMMARY
============================================================
Actions         ✅ PASS (std=0.186, good variance)
Joint Mask      ✅ PASS (bidirectional attention)
Overfit         ✅ PASS (93.0% accuracy)

🎉 All tests passed! Ready for full training.
```

**Overfit Test Details:**
- Model: 128 dim, 2 layers (663K params)
- 100 steps on single sequence
- Final accuracy: **93%** (should be >80%)
- Loss: 1.33 → 0.21

---

## Training Commands

### Small Model (Recommended for Testing)
```bash
python tests/mnist_diffusion/train_mnist_small.py \
    --embed_dim 192 \
    --num_layers 4 \
    --num_heads 4 \
    --frame_size 32 \
    --batch_size 4 \
    --epochs 20 \
    --use_ego_centric
```

### Full Model (Current Setup)
```bash
python tests/mnist_diffusion/train_mnist_full.py \
    --embed_dim 384 \
    --num_layers 6 \
    --num_heads 6 \
    --frame_size 32 \
    --batch_size 1 \
    --epochs 50 \
    --use_ego_centric
```

### With DataLoader Support
Update `create_mnist_dataloaders()` call:
```python
train_loader, val_loader = create_mnist_dataloaders(
    use_ego_centric=True,
    ego_digit_id=0,
    # ... other params
)
```

---

## Key Changes Summary

| File | Lines Changed | Purpose |
|------|--------------|---------|
| `moving_mnist_precomputed.py` | +80 | Ego-centric action generation |
| `train_mnist_full.py` | +10 | Fix joint modeling mask |
| `verify_fixes.py` | +250 | Verification script (NEW) |

---

## Expected Training Progress

With these fixes, you should see:

**Epoch 1:**
- Train loss: ~0.20 → 0.15
- Val loss: ~0.35
- Samples: Blurry but coherent motion

**Epoch 5:**
- Train loss: ~0.10
- Val loss: ~0.20
- Samples: Sharp digits, correct motion

**Epoch 10+:**
- Train loss: ~0.05
- Val loss: ~0.15
- Samples: High quality generation

---

## Debugging Checklist

If still seeing sparkles/noise:

1. **Verify actions:**
   ```python
   python verify_fixes.py --ego
   ```
   Should show: `Action std: >0.1`

2. **Verify overfitting:**
   ```python
   python verify_fixes.py --ego --steps 200
   ```
   Should show: `Acc > 0.8`

3. **Check sampling:**
   - Temperature: Try 0.8 instead of 1.0
   - Steps: Use 20+ sampling steps
   - EMA: Use EMA weights for sampling

---

## Files Modified

1. `tests/mnist_diffusion/moving_mnist_precomputed.py`
   - Added `_generate_actions_ego_centric()`
   - Added `use_ego_centric` parameter
   - Updated `create_mnist_dataloaders()`

2. `tests/mnist_diffusion/train_mnist_full.py`
   - Fixed joint denoising mask (None instead of causal)
   - Added `_make_bidirectional_mask()`

3. `tests/mnist_diffusion/verify_fixes.py` (NEW)
   - Comprehensive verification script
   - Tests actions, masks, overfitting
