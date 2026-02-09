"""
Visualization of the three masking schedules in Algorithm 1.

This script demonstrates how each training objective works with concrete examples.
"""

import random
import math


def cosine_mask_schedule(u: float) -> float:
    """γ(u) = cos(u * π/2)"""
    return math.cos(u * math.pi / 2)


def visualize_masking_schedules():
    """
    Visualize the three training objectives with a concrete example.
    
    Setup:
    - T = 6 frames (3 past + 3 future)
    - H = W = 4 (small grid for visualization)
    - Codebook size = 1024
    - Mask token = 1024
    """
    
    print("=" * 80)
    print("ALGORITHM 1: THREE MASKING SCHEDULES")
    print("=" * 80)
    
    # Configuration
    T, H, W = 6, 4, 4
    num_past = 3
    codebook_size = 1024
    mask_token = 1024
    noise_eta = 20.0
    
    # Sample u0 and u1 for mask/noise ratios
    u0 = 0.7  # For mask ratio
    u1 = 0.5  # For noise ratio
    
    mask_ratio = cosine_mask_schedule(u0)
    noise_ratio = u1 * noise_eta / 100.0
    
    print(f"\nCommon Parameters:")
    print(f"  T = {T} frames (3 past + 3 future)")
    print(f"  Grid = {H}×{W} = {H*W} tokens per frame")
    print(f"  u₀ = {u0} → mask_ratio = γ({u0}) = {mask_ratio:.3f}")
    print(f"  u₁ = {u1} → noise_ratio = {u1}×{noise_eta}% = {noise_ratio:.1%}")
    print(f"  Mask token ID = {mask_token}")
    print(f"  Random tokens sampled from [0, {codebook_size-1}]")
    
    # Calculate expected numbers
    N = H * W
    num_mask = math.ceil(mask_ratio * N)
    num_noise = int(noise_ratio * (N - num_mask))
    
    print(f"\n  For {N} tokens per frame:")
    print(f"    → ~{num_mask} tokens masked with [M]")
    print(f"    → ~{num_noise} tokens noised with [R] (from unmasked {N - num_mask})")
    
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("OBJECTIVE 1: FUTURE PREDICTION (50% probability)")
    print("=" * 80)
    print("Description: Past frames are GT, Future frames are fully masked+noised")
    print("Temporal Mask: CAUSAL (past can attend to past, future can attend to past+future)")
    print()
    
    print("Token Status per Frame:")
    for t in range(T):
        if t < num_past:
            print(f"  Frame {t} (PAST):  GT (unmasked)     → [1][2][3][4]...")
        else:
            print(f"  Frame {t} (FUTURE): {num_mask} masked + {num_noise} noised → [M][M][R][M]...")
    
    print("\nTemporal Attention Mask (Causal):")
    print("      T0   T1   T2   T3   T4   T5")
    for i in range(T):
        row = []
        for j in range(T):
            if j > i:
                row.append(" -  ")  # Blocked
            else:
                row.append(" 0  ")  # Allowed
        print(f"  T{i} " + "".join(row))
    print("  (0 = can attend, - = blocked)")
    
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("OBJECTIVE 2: JOINT DENOISE (40% probability)")
    print("=" * 80)
    print("Description: ALL frames partially masked+noised")
    print("Temporal Mask: CAUSAL")
    print()
    
    print("Token Status per Frame:")
    for t in range(T):
        frame_type = "PAST" if t < num_past else "FUTURE"
        print(f"  Frame {t} ({frame_type}): {num_mask} masked + {num_noise} noised → [M][R][M][M]...")
    
    print("\nTemporal Attention Mask: SAME AS FUTURE PREDICTION (Causal)")
    print("Key Difference: Past frames are ALSO corrupted (not GT)")
    
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("OBJECTIVE 3: INDIVIDUAL DENOISE (10% probability)")
    print("=" * 80)
    print("Description: Each frame independently masked+noised")
    print("Temporal Mask: IDENTITY (each frame only attends to itself)")
    print("Purpose: Learn unconditional generative model for Classifier-Free Guidance (CFG)")
    print()
    
    print("Token Status per Frame:")
    for t in range(T):
        frame_type = "PAST" if t < num_past else "FUTURE"
        print(f"  Frame {t} ({frame_type}): {num_mask} masked + {num_noise} noised (INDEPENDENT)")
    
    print("\nTemporal Attention Mask (Identity):")
    print("      T0   T1   T2   T3   T4   T5")
    for i in range(T):
        row = []
        for j in range(T):
            if i == j:
                row.append(" 0  ")  # Allowed (self only)
            else:
                row.append(" -  ")  # Blocked
        print(f"  T{i} " + "".join(row))
    print("  (Each frame only attends to itself - unconditional generation)")
    
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    
    comparison = """
    ┌─────────────────────┬─────────────────────┬─────────────────────┬─────────────────────┐
    │       Aspect        │ Future Prediction   │   Joint Denoise     │ Individual Denoise  │
    ├─────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
    │ Probability         │       50%           │        40%          │        10%          │
    ├─────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
    │ Past Frames         │   Ground Truth      │  Masked + Noised    │  Masked + Noised    │
    │                     │   (unmasked)        │  (partially)        │  (independently)    │
    ├─────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
    │ Future Frames       │ Fully Masked+Noised│  Masked + Noised    │  Masked + Noised    │
    │                     │                     │  (partially)        │  (independently)    │
    ├─────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
    │ Temporal Mask       │      Causal         │      Causal         │      Identity       │
    │                     │  (past→past,        │  (past→past,        │  (self-attention    │
    │                     │   future→all)       │   future→all)       │   only)             │
    ├─────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
    │ Training Goal       │  Predict future     │  Jointly model      │  Unconditional      │
    │                     │  from past          │  past + future      │  generation (CFG)   │
    ├─────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
    │ Loss Applied On     │   All positions     │   All positions     │   All positions     │
    │                     │ (masked+unmasked)   │ (masked+unmasked)   │ (masked+unmasked)   │
    └─────────────────────┴─────────────────────┴─────────────────────┴─────────────────────┘
    
    Key Implementation Details:
    ─────────────────────────
    • Mask ratio: γ(u₀) = cos(u₀·π/2) where u₀ ~ Uniform(0,1)
    • Noise ratio: u₁·η% where u₁ ~ Uniform(0,1), η = 20%
    • Mask token ID: 1024 (separate from codebook [0-1023])
    • Random noise tokens: sampled uniformly from [0, 1023]
    • Loss: Cross-entropy on ALL token positions (not just masked)
    • Label smoothing: 0.1 applied during loss computation
    """
    print(comparison)


def create_visual_diagram():
    """Create ASCII art diagrams showing the masking patterns."""
    
    diagram = """
    ╔══════════════════════════════════════════════════════════════════════════════════════╗
    ║                    VISUAL MASKING PATTERNS (6 frames, 4×4 grid)                      ║
    ╚══════════════════════════════════════════════════════════════════════════════════════╝
    
    Legend:
      [0-9] = Original token (codebook index)
      [M]   = Mask token (ID 1024)
      [R]   = Random noise token (random codebook index)
    
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │ 1. FUTURE PREDICTION (50%)                                                          │
    │    "Condition on past, predict future"                                              │
    │                                                                                     │
    │    Frame 0 (Past):  [1][2][3][4]   ← Ground Truth (unmasked)                        │
    │                     [5][6][7][8]                                                    │
    │                     [9][0][1][2]                                                    │
    │                     [3][4][5][6]                                                    │
    │    Frame 1 (Past):  [7][8][9][0]   ← Ground Truth (unmasked)                        │
    │                     [1][2][3][4]                                                    │
    │                     [5][6][7][8]                                                    │
    │                     [9][0][1][2]                                                    │
    │    Frame 2 (Past):  [3][4][5][6]   ← Ground Truth (unmasked)                        │
    │                     [7][8][9][0]                                                    │
    │                     [1][2][3][4]                                                    │
    │                     [5][6][7][8]                                                    │
    │    ────────────────────────────────────────                                         │
    │    Frame 3 (Fut):   [M][M][R][M]   ← Masked + some Noised                           │
    │                     [M][R][M][M]     (γ(u₀)≈60% masked, u₁·20% of rest noised)      │
    │                     [M][M][M][R]                                                    │
    │                     [R][M][M][M]                                                    │
    │    Frame 4 (Fut):   [M][R][M][M]   ← Masked + some Noised                           │
    │                     [M][M][R][M]                                                    │
    │                     [R][M][M][M]                                                    │
    │                     [M][M][M][R]                                                    │
    │    Frame 5 (Fut):   [M][M][M][R]   ← Masked + some Noised                           │
    │                     [R][M][M][M]                                                    │
    │                     [M][R][M][M]                                                    │
    │                     [M][M][R][M]                                                    │
    │                                                                                     │
    │    Temporal Flow:  [0]──→[1]──→[2]──→[3]──→[4]──→[5]                                │
    │                      ↑     ↑     ↑     ↓     ↓     ↓                                │
    │                    Past frames can only attend to previous past frames              │
    │                    Future frames can attend to all past AND previous future         │
    └─────────────────────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │ 2. JOINT DENOISE (40%)                                                              │
    │    "Joint modeling of past and future"                                              │
    │                                                                                     │
    │    Frame 0 (Past):  [M][R][M][M]   ← Masked + Noised                                │
    │                     [M][M][R][M]                                                    │
    │                     [R][M][M][M]                                                    │
    │                     [M][M][M][R]                                                    │
    │    Frame 1 (Past):  [M][M][R][M]   ← Masked + Noised                                │
    │                     [R][M][M][M]                                                    │
    │                     [M][M][M][R]                                                    │
    │                     [M][R][M][M]                                                    │
    │    Frame 2 (Past):  [R][M][M][M]   ← Masked + Noised                                │
    │                     [M][M][R][M]                                                    │
    │                     [M][R][M][M]                                                    │
    │                     [M][M][M][R]                                                    │
    │    ────────────────────────────────────────                                         │
    │    Frame 3 (Fut):   [M][M][M][R]   ← Masked + Noised                                │
    │                     [M][R][M][M]                                                    │
    │                     [R][M][M][M]                                                    │
    │                     [M][M][R][M]                                                    │
    │    Frame 4 (Fut):   [M][R][M][M]   ← Masked + Noised                                │
    │                     [M][M][M][R]                                                    │
    │                     [M][M][R][M]                                                    │
    │                     [R][M][M][M]                                                    │
    │    Frame 5 (Fut):   [R][M][M][M]   ← Masked + Noised                                │
    │                     [M][M][R][M]                                                    │
    │                     [M][R][M][M]                                                    │
    │                     [M][M][M][R]                                                    │
    │                                                                                     │
    │    Note: All frames treated the same way! Both past and future are corrupted.       │
    │    Model must jointly denoise the entire sequence.                                  │
    └─────────────────────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │ 3. INDIVIDUAL DENOISE (10%)                                                         │
    │    "Model each frame individually"                                                  │
    │                                                                                     │
    │    Frame 0:  [M][R][M][M]   ← Independently masked+noised                           │
    │              [M][M][R][M]                                                           │
    │              [R][M][M][M]                                                           │
    │              [M][M][M][R]                                                           │
    │                  ↓                                                                  │
    │              (NO temporal attention - frame 0 attends ONLY to frame 0)              │
    │                                                                                     │
    │    Frame 1:  [M][M][R][M]   ← Independently masked+noised                           │
    │              [R][M][M][M]                                                           │
    │              [M][M][M][R]                                                           │
    │              [M][R][M][M]                                                           │
    │                  ↓                                                                  │
    │              (NO temporal attention - frame 1 attends ONLY to frame 1)              │
    │                                                                                     │
    │    Frame 2:  [R][M][M][M]   ← Independently masked+noised                           │
    │              [M][M][R][M]                                                           │
    │              [M][R][M][M]                                                           │
    │              [M][M][M][R]                                                           │
    │                  ↓                                                                  │
    │              (NO temporal attention - frame 2 attends ONLY to frame 2)              │
    │                                                                                     │
    │    ... (same pattern for frames 3, 4, 5)                                            │
    │                                                                                     │
    │    Purpose: Learn unconditional generation for Classifier-Free Guidance (CFG)       │
    │    During inference, CFG requires: logit = cond + w × (cond - uncond)               │
    │                                                                                     │
    │    Temporal Flow:  [0]  [1]  [2]  [3]  [4]  [5]                                     │
    │                    ↓    ↓    ↓    ↓    ↓    ↓                                       │
    │                   Self Self Self Self Self Self  (identity attention)               │
    └─────────────────────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │ TEMPORAL ATTENTION MASK VISUALIZATION                                               │
    │                                                                                     │
    │ Causal Mask (Future Pred & Joint Denoise):           Identity Mask (Individual):    │
    │                                                                                     │
    │      T0 T1 T2 T3 T4 T5                                  T0 T1 T2 T3 T4 T5          │
    │  T0 [ 0  -  -  -  -  - ]                            T0 [ 0  -  -  -  -  - ]          │
    │  T1 [ 0  0  -  -  -  - ]                            T1 [ -  0  -  -  -  - ]          │
    │  T2 [ 0  0  0  -  -  - ]                            T2 [ -  -  0  -  -  - ]          │
    │  T3 [ 0  0  0  0  -  - ]                            T3 [ -  -  -  0  -  - ]          │
    │  T4 [ 0  0  0  0  0  - ]                            T4 [ -  -  -  -  0  - ]          │
    │  T5 [ 0  0  0  0  0  0 ]                            T5 [ -  -  -  -  -  0 ]          │
    │                                                                                     │
    │  0 = can attend, - = blocked (-inf)                 0 = can attend, - = blocked     │
    │  Lower triangular allows attending to self          Only diagonal allows self-only  │
    │  and all previous frames.                           attention.                      │
    └─────────────────────────────────────────────────────────────────────────────────────┘
    """
    print(diagram)


if __name__ == "__main__":
    # Run visualization
    visualize_masking_schedules()
    print("\n\n")
    create_visual_diagram()
