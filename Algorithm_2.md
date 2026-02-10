Here's a **concrete numerical example** with actual matrix values for Algorithm 2:

## Setup
- **Batch size**: 1
- **Future tokens**: 1 frame × 5 positions ($N=5$)  
- **Vocabulary**: IDs {0,1,2,3,4,5,6,7}, **Mask ID = 8**
- **Iterations**: $K=4$ (steps $k=3,2,1,0$)
- **Schedule**: $\gamma(t) = \cos(\frac{\pi t}{2})$

---

## Step 1: Initialize ($\mathbf{x}^4$)

All tokens start masked:

```python
x^4 = [8, 8, 8, 8, 8]  # [MASK, MASK, MASK, MASK, MASK]
is_masked = [True, True, True, True, True]
```

---

## Iteration 1: $k=3$ (Keep $M = \lceil\cos(\frac{3\pi}{8}) \times 5\rceil = \lceil0.383 \times 5\rceil = 2$ tokens)

### Step 3: Forward Pass & Sampling
Model predicts logits for all 5 positions (random example values):

| Position | Logits [0-7] | Softmax Prob | Sampled Token |
|----------|--------------|--------------|---------------|
| 0 | [2.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] | [0.60, 0.22, ...] | **0** |
| 1 | [0.1, 0.1, 3.0, 0.1, 0.1, 0.1, 0.1, 0.1] | [0.05, 0.05, 0.75, ...] | **2** |
| 2 | [1.0, 2.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] | [0.27, 0.73, ...] | **1** |
| 3 | [0.1, 0.1, 0.1, 4.0, 0.1, 0.1, 0.1, 0.1] | [0.02, 0.02, 0.02, 0.88, ...] | **3** |
| 4 | [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] | [0.125, 0.125, ...] | **6** |

```python
x_tilde_0 = [0, 2, 1, 3, 6]  # Sampled values
```

### Step 4: Calculate Confidence ($l_k$)
Confidence = Log probability + Gumbel(0,1) × (3/4)

| Pos | Token | Log Prob | Gumbel Noise | Weight ($\frac{3}{4}$) | Confidence |
|-----|-------|----------|--------------|------------------------|------------|
| 0 | 0 | $\ln(0.60) = -0.51$ | 0.2 | 0.15 | **-0.36** |
| 1 | 2 | $\ln(0.75) = -0.29$ | -0.1 | -0.075 | **-0.365** |
| 2 | 1 | $\ln(0.73) = -0.31$ | 0.5 | 0.375 | **+0.065** |
| 3 | 3 | $\ln(0.88) = -0.13$ | 0.8 | 0.6 | **+0.47** |
| 4 | 6 | $\ln(0.125) = -2.08$ | -0.5 | -0.375 | **-2.455** |

### Step 5: Lock Previously Unmasked
Currently all are masked (`is_masked = [T,T,T,T,T]`), so no changes to confidence.

### Step 6-7: Select Top-M and Remask
Sort by confidence: Pos 3 (0.47) > Pos 2 (0.065) > Pos 0 (-0.36) > Pos 1 (-0.365) > Pos 4 (-2.455)

Keep top $M=2$: Positions **3** and **2**

```python
# After remasking
x^3 = [8, 8, 1, 3, 8]  
#       ↑  ↑     ↑     ↑
#    remasked   kept  remasked
is_masked = [True, True, False, False, True]
```

**Result after $k=3$**: 2 tokens revealed (positions 2,3), 3 still masked.

---

## Iteration 2: $k=2$ (Keep $M = \lceil\cos(\frac{\pi}{4}) \times 5\rceil = \lceil0.707 \times 5\rceil = 4$ tokens)

### Step 3: Forward Pass (Conditioned on $x^3$)
Model now sees `[8,8,1,3,8]` and predicts again (new random samples):

| Pos | Sampled $\tilde{x}^0$ |
|-----|----------------------|
| 0 | **5** |
| 1 | **2** |
| 2 | **1** (same as before, model is confident) |
| 3 | **3** (same as before) |
| 4 | **7** |

```python
x_tilde_0 = [5, 2, 1, 3, 7]
```

### Step 4: Calculate Confidence
| Pos | Log Prob | Gumbel | Weight (2/4=0.5) | Raw Conf |
|-----|----------|--------|------------------|----------|
| 0 | -0.8 | 0.3 | 0.15 | **-0.65** |
| 1 | -0.4 | 0.1 | 0.05 | **-0.35** |
| 2 | -0.1 | 0.9 | 0.45 | **+0.35** |
| 3 | -0.2 | 0.4 | 0.2 | **0.0** |
| 4 | -1.5 | -0.2 | -0.1 | **-1.6** |

### Step 5: CRITICAL STEP - Lock Unmasked
Positions 2 and 3 were unmasked in $x^3$. Set their confidence to $+\infty$:

| Pos | Was Masked? | Confidence After Step 5 |
|-----|-------------|-------------------------|
| 0 | Yes | -0.65 |
| 1 | Yes | -0.35 |
| 2 | **No** | **+∞** |
| 3 | **No** | **+∞** |
| 4 | Yes | -1.6 |

### Step 6-7: Select Top-M ($M=4$)
Sorted: Pos 2 (∞), Pos 3 (∞), Pos 0 (-0.65), Pos 1 (-0.35), Pos 4 (-1.6)

Keep top 4: Positions **2, 3, 0, 1**

```python
# After remasking
x^2 = [5, 2, 1, 3, 8]
#                    ↑
#               remasked (not in top 4)
is_masked = [False, False, False, False, True]
```

**Key observation**: Positions 2 and 3 kept their values (1 and 3) from the previous step, even though the model sampled new values (1 and 3 again) and calculated new confidences. The $+\infty$ prevented them from being remasked.

---

## Iteration 3: $k=1$ (Keep $M = \lceil\cos(\frac{\pi}{8}) \times 5\rceil = \lceil0.924 \times 5\rceil = 5$ tokens)

### Step 3: Forward Pass
Model sees `[5,2,1,3,8]`:

| Pos | Sampled |
|-----|---------|
| 0 | **5** |
| 1 | **2** |
| 2 | **4** (model changed its mind, but...) |
| 3 | **3** |
| 4 | **0** |

### Step 4-5: Confidence + Lock
Raw confidence calculation, then set positions 0,1,2,3 to $+\infty$ (they were unmasked in $x^2$):

| Pos | Confidence | After Lock |
|-----|------------|------------|
| 0 | -0.2 | **+∞** |
| 1 | -0.5 | **+∞** |
| 2 | -0.1 | **+∞** |
| 3 | -0.3 | **+∞** |
| 4 | -0.8 | -0.8 |

### Step 6-7: Select Top-M ($M=5$)
Top 5 of 5 positions = all positions.

```python
x^1 = [5, 2, 1, 3, 0]
is_masked = [False, False, False, False, False]
```

Note: Position 2 stayed as **1** (from $x^2$) despite the model sampling **4** this iteration. The lock preserved the previous value because $M=5$ includes all, but even if $M$ were smaller, position 2's $\infty$ would guarantee inclusion.

---

## Iteration 4: $k=0$ (Keep $M = \lceil\cos(0) \times 5\rceil = 5$ tokens)

### Final Step
All positions already unmasked. Model predicts final refinement:

| Pos | Final Sampled |
|-----|---------------|
| 0 | **5** |
| 1 | **2** |
| 2 | **1** |
| 3 | **3** |
| 4 | **0** |

Since $M=5$ and nothing is masked, we keep all:

```python
x^0 = [5, 2, 1, 3, 0]
```

---

## Final Result

```python
# Evolution over iterations
x^4 = [8, 8, 8, 8, 8]   # Initial (all mask)
x^3 = [8, 8, 1, 3, 8]   # Step 1: 2 unmasked (positions 2,3)
x^2 = [5, 2, 1, 3, 8]   # Step 2: 4 unmasked (added 0,1), kept 2,3 locked
x^1 = [5, 2, 1, 3, 0]   # Step 3: All unmasked (added 4), kept 0,1,2,3 locked
x^0 = [5, 2, 1, 3, 0]   # Step 4: Final refinement (no masks left)
```

**Without Step 5 (the bug)**: At $k=2$, position 2 might have been remasked if its new confidence (-0.1) was low, causing instability. With Step 5, the set of unmasked tokens grows monotonically: {∅} → {2,3} → {0,1,2,3} → {0,1,2,3,4}.