Here is the step-by-step breakdown of **Algorithm 2 (Sampling)** and how it maps to the visual pipeline in **Figure 11**, followed by the implementation comparison.

---

## **Algorithm 2: Discrete Diffusion Sampling**

The algorithm performs **iterative parallel decoding**: starting from fully masked tokens, it progressively unmasks subsets of tokens based on model confidence, allowing previously decoded tokens to be revised (resampled) in later steps.

### **Step-by-Step Breakdown**

| Step | Pseudocode | Explanation |
|------|------------|-------------|
| **1** | $\mathbf{x}_K = \text{all mask tokens}$ | Initialize future frames with all `[MASK]` tokens (fully unknown). |
| **2** | **for** $k = K-1, \dots, 0$ **do** | Iterate backwards through diffusion steps (e.g., 10 steps). |
| **3** | $\tilde{\mathbf{x}}_0 \sim p_\theta(\cdot \mid \mathbf{x}_{k+1})$ | **Prediction**: Model predicts logits for all positions; sample candidate tokens $\tilde{\mathbf{x}}_0$ from the categorical distribution (or take argmax). |
| **4** | $l_k = \log p_\theta(\tilde{\mathbf{x}}_0 \mid \mathbf{x}_{k+1}) + \text{Gumbel}(0,1) \cdot k/K$ | **Confidence scoring**: Calculate log-likelihood of the sampled tokens, add annealed Gumbel noise for stochasticity (temperature decreases as $k \to 0$). |
| **5** | On non-mask indices of $\mathbf{x}_{k+1}$: $l_k \leftarrow +\infty$ | **Preserve decoded tokens**: Already-unmasked positions get infinite confidence, ensuring they remain in the selected set (top-$M$) and can be resampled if the model changes its prediction. |
| **6** | $M = \lceil \gamma(k/K) N \rceil$ | **Schedule**: Calculate how many tokens to unmask. $\gamma(u) = \cos(u\pi/2)$, so $M$ increases as $k$ decreases (fewer masks early, all unmasked at $k=0$). |
| **7** | $\mathbf{x}_k \leftarrow \tilde{\mathbf{x}}_0$ on top-$M$ indices of $l_k$ | **Update**: Replace tokens at the $M$ most confident positions with the newly sampled values. Keep other positions as they were (masked or previously decoded). |
| **9** | **return** $\mathbf{x}_0$ | Final clean tokens after $K$ refinement steps. |

---

## **Figure 11: Visual Sampling Pipeline**

The diagram illustrates **one iteration** (step $k$) of Algorithm 2:

1. **Input**: Partially masked tokens (gray cubes = `[MASK]`, blue/white = already decoded)
2. **World Model**: Processes current state $\mathbf{x}_{k+1}$ → outputs **CFG Logits** (Classifier-Free Guidance applied)
3. **Sampling per location**: Sample $\tilde{\mathbf{x}}_0$ from logits + calculate log-likelihoods
4. **Binary Mask Branch**: 
   - Identify which positions are still masked
   - Set log-likelihoods of **non-masked** positions to $+\infty$ (protects them from being dropped)
5. **Selection**:
   - Add Gumbel noise for stochastic sampling
   - Sort by confidence
   - Select top-$M = \lceil \gamma(k/K)N \rceil$ locations
6. **Output**: Update tokens → new state $\mathbf{x}_k$ with fewer masks
7. **Loop**: Feed $\mathbf{x}_k$ back as input for step $k-1$ (until $k=0$)

---

## **Implementation vs. Pseudocode**

### **Key Differences**

| Aspect | Pseudocode (Mathematical) | Implementation (PyTorch) |
|--------|--------------------------|--------------------------|
| **State** | $\mathbf{x}_k$ (abstract sequence) | `x: LongTensor[B, T, H, W]` (batch, time, height, width) |
| **Masking** | Conceptual "mask token" index | Special integer ID (e.g., `1024`) in vocabulary |
| **Step 3 (Sampling)** | $\sim p_\theta(\cdot)$ | `probs = F.softmax(logits, -1); x_pred = torch.multinomial(probs)` or `argmax` for greedy |
| **Step 4 (Confidence)** | Log-probability + Gumbel noise | `log_probs = torch.log(probs.gather(-1, x_pred.unsqueeze(-1)))`<br>`gumbel = -torch.log(-torch.log(torch.rand_like(log_probs).clamp(1e-8)))`<br>`confidence = log_probs + gumbel * (k/K)` |
| **Step 5 (Protect)** | $l_k \leftarrow +\infty$ | `confidence[mask == False] = float('inf')` |
| **Step 6 (Schedule)** | $\gamma(u) = \cos(u\pi/2)$ | `gamma = math.cos((k/K) * math.pi / 2)`<br>`M = math.ceil(gamma * N)` |
| **Step 7 (Update)** | $\leftarrow$ on top-$M$ indices | `_, indices = torch.topk(confidence, M, dim=-1)`<br>`x.scatter_(1, indices, x_pred.gather(1, indices))` |
| **Memory** | Stores $\mathbf{x}_K, \mathbf{x}_{K-1}, \dots$ | Single tensor updated **in-place** (overwritten each step) |
| **Batching** | Implicit | Explicit handling of batch dimension; masks may vary per sample (use `torch.where`) |

### **Critical Implementation Details**

**1. Resampling Capability**
Unlike standard autoregressive models or original MaskGIT, line 5 (`$l_k \leftarrow +\infty$` on non-mask indices) allows tokens decoded in earlier steps ($k+1$) to be **revised** in later steps ($k$) if the model's confidence changes. In code:
```python
# Already decoded positions keep high confidence but still get updated
confidence[~mask_positions] = float('inf')
# ... topk selection includes them ...
# x is updated with new predictions at ALL top-M positions, including previously unmasked ones
```

**2. Classifier-Free Guidance (CFG)**
Figure 11 shows "CFG Logits" entering the sampling. In practice, this requires **two forward passes** (or one batched pass) per step:
```python
# Conditional: with past context
logits_cond = model(x, actions, context_mask=causal)
# Unconditional: drop context (or use empty context)
logits_uncond = model(x, actions_zeroed, context_mask=identity)
# Combine
logits = logits_uncond + w * (logits_cond - logits_uncond)
```

**3. Parallel Decoding**
While the loop runs $K$ times (e.g., 10 iterations), each iteration decodes **$M$ tokens in parallel** (not sequentially). This is why $K \ll N$ (10 steps vs. 16,384 tokens) is feasible, unlike autoregressive sampling which requires $N$ sequential steps.

**4. Temperature Annealing**
The term $\text{Gumbel}(0,1) \cdot k/K$ in Algorithm 2 line 4 means:
- Early steps ($k \approx K$): High temperature (noisy sampling, explore diverse modes)
- Late steps ($k \approx 0$): Low temperature (deterministic, refine details)