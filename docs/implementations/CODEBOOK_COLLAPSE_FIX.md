# Codebook Collapse Diagnosis and Fix

## Problem Analysis

### Observed Symptoms (from loss_curves.png)

| Metric | Observation | Healthy Pattern |
|--------|-------------|-----------------|
| **VQ Loss** | 0.02 → 0.09 (increasing) | Should decrease to <0.05 |
| **Spikes** | ~0.13 at steps 500, 1000, 1500 | Should be stable |
| **Trend** | Upward after step 2000 | Should flatten |
| **Variance** | High throughout | Should decrease |

### Root Causes Identified

#### 1. **Inverted Loss Coefficients (Critical)**

The original code had the loss coefficients swapped:

```python
# ORIGINAL (WRONG):
commitment_cost=0.25    # Was being used for commitment
codebook_cost=1.0       # Was being used for codebook

# PAPER SPECIFICATION:
# L_vq = 0.25 * ||sg[E(o)] - z_hat||² + 1.0 * ||sg[z_hat] - E(o)||²
codebook_cost=0.25      # Codebook moves toward encoder (small)  
commitment_cost=1.0     # Encoder commits to codebook (large)
```

**Impact**: The encoder wasn't being pushed hard enough to commit to codebook entries, causing drift.

#### 2. **Missing EMA Codebook Update**

The original implementation used pure gradient updates for the codebook:

```python
# ORIGINAL: Codebook updated via gradient descent
# Problem: Noisy gradients cause codebook to drift
```

**Solution**: Use Exponential Moving Average (EMA) for codebook updates:

```python
# FIXED: EMA-based codebook update
self.cluster_size = self.cluster_size * decay + (1 - decay) * encodings.sum(dim=0)
self.embed_avg = self.embed_avg * decay + (1 - decay) * embed_sum.t()
self.embed = self.embed_avg / cluster_size.unsqueeze(1)
```

**Benefits**:
- More stable codebook updates
- Less sensitive to batch variance
- Better maintains codebook diversity

#### 3. **Over-Aggravating Dead Code Detection**

Original code checked for dead codes **every training step**:

```python
# ORIGINAL: Called every forward pass during training
self._check_and_reinit_dead_codes()  # Runs every step!
```

**Problem**: K-Means re-initialization is computationally expensive and disrupts training stability when run too frequently.

**Solution**: Check only every N steps (e.g., 100):

```python
# FIXED: Check periodically
if self.iteration_count % self.reinit_every != 0:
    return  # Skip check
```

#### 4. **Poor Codebook Initialization**

Random initialization without normalization:

```python
# ORIGINAL
self.embed = torch.randn(codebook_size, codebook_dim)
```

**Solution**: Unit sphere normalization:

```python
# FIXED: Normalized initialization
self.embed.data = F.normalize(self.embed.data, dim=1) * (codebook_dim ** 0.5)
```

## Implementation Changes

### New File: `vector_quantizer_fixed.py`

Created a fixed version with the following key improvements:

1. **EMA-based codebook updates** (decay=0.99)
2. **Correct loss coefficients** (codebook_cost=0.25, commitment_cost=1.0)
3. **Periodic dead code check** (every 100 steps)
4. **Unit sphere initialization**
5. **Perplexity and usage metrics** for monitoring

### Updated Files

#### `tokenizer_model.py`
- Import `VectorQuantizerFixed` instead of `VectorQuantizer`
- Pass correct loss coefficient mapping
- Return `vq_metrics` in forward output

#### `config.py`
- Updated comments to clarify loss coefficient semantics
- Swapped values to match paper:
  - `vq_commitment_cost: 1.0` (was 0.25)
  - `vq_codebook_cost: 0.25` (was 1.0)

#### `train_tokenizer.py`
- Log perplexity and codebook usage during training
- Better progress bar display

#### `tests/test_tokenizer_shapes.py`
- Updated to use `VectorQuantizerFixed`
- Added metrics validation

## Expected Behavior After Fix

### VQ Loss Pattern

| Phase | Before (Broken) | After (Fixed) |
|-------|-----------------|---------------|
| Step 0 | ~0.01 | ~0.1-0.5 |
| Step 1K | Increasing with spikes | Decreasing smoothly |
| Step 5K | ~0.09 (worse than start) | ~0.01-0.02 |
| Trend | ↑ Diverging | ↓ Converging |

### Codebook Usage

| Metric | Before | After |
|--------|--------|-------|
| Active codes | <200 (20%) | >800 (80%+) |
| Perplexity | <100 | ~600-900 |
| Dead codes | >800 | <50 |

## Monitoring Guide

### Key Metrics to Watch

```python
# During training, monitor these values:
{
    "vq_loss": 0.02,        # Should decrease over time
    "perplexity": 800,      # Should be high (>500)
    "usage": 950,           # Active codes out of 1024
}
```

### Healthy Training Signs

1. **VQ Loss**: Steady decrease, no spikes >0.1 after warmup
2. **Perplexity**: High (>600) and stable
3. **Usage**: >80% of codebook active
4. **Depth Loss**: Improving (this is the main objective)

### Unhealthy Signs

1. **VQ Loss increasing**: Codebook collapse
2. **Perplexity <200**: Only few codes being used
3. **Usage <300**: Severe codebook collapse
4. **Spikes in VQ loss**: Instability from re-initialization

## Testing the Fix

```bash
# Test the fixed VQ layer
python -c "
from copilot4d.tokenizer.vector_quantizer_fixed import VectorQuantizerFixed
import torch

vq = VectorQuantizerFixed(dim=256, codebook_size=1024, codebook_dim=1024)
x = torch.randn(2, 64, 256)
quantized, indices, loss, metrics = vq(x)

print(f'Loss: {loss.item():.4f}')
print(f'Perplexity: {metrics[\"perplexity\"]:.1f}')
print(f'Usage: {metrics[\"usage\"]:.0f}/1024')
"
```

## Next Steps

1. **Restart training** with the fixed implementation
2. **Monitor perplexity** - should be >600 within first 1000 steps
3. **Check codebook usage** - should be >800 active codes
4. **Verify VQ loss trend** - should decrease to <0.05 by step 5K

## References

- Paper: CoPilot4D (Section 4.1, Appendix A.2.1)
- VQ-VAE Original: "Neural Discrete Representation Learning" (Oord et al.)
- EMA Best Practices: "Video Compression with Rate-Distortion Autoencoders"
