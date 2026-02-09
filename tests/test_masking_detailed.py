#!/usr/bin/env python3
"""
Detailed test of Algorithm 1 masking with synthetic data.

This script:
1. Creates synthetic input (all 1's)
2. Applies each of the 3 masking schedules
3. Prints detailed intermediate variables
4. Shows the transformation step-by-step
"""

import random
import math
from typing import Tuple, List


# ==============================================================================
# Configuration
# ==============================================================================
CODEBOOK_SIZE = 1024
MASK_TOKEN_ID = 1024  # Codebook size
NOISE_ETA = 20.0      # 20% noise


def cosine_mask_schedule(u: float) -> float:
    """γ(u) = cos(u * π/2)"""
    return math.cos(u * math.pi / 2)


def apply_random_masking_detailed(
    tokens: List[int],
    u0: float,
    u1: float,
    frame_name: str = ""
) -> Tuple[List[int], List[bool], dict]:
    """
    Apply random masking with detailed logging.
    
    Returns:
        (masked_tokens, was_masked, debug_info)
    """
    N = len(tokens)
    original = tokens.copy()
    
    # Step 1: Sample mask ratio from u0
    mask_ratio = cosine_mask_schedule(u0)
    num_mask = math.ceil(mask_ratio * N)
    
    # Step 2: Sample noise ratio from u1
    noise_ratio = u1 * NOISE_ETA / 100.0
    num_unmasked = N - num_mask
    num_noise = min(int(noise_ratio * num_unmasked), num_unmasked) if num_unmasked > 0 else 0
    
    # Random permutation for mask/noise selection
    all_indices = list(range(N))
    random.shuffle(all_indices)
    
    mask_indices = set(all_indices[:num_mask])
    unmasked_indices = all_indices[num_mask:]
    noise_indices = set(unmasked_indices[:num_noise]) if num_noise > 0 else set()
    
    # Apply masking and noise
    masked_tokens = tokens.copy()
    was_masked = [False] * N
    
    for i in mask_indices:
        masked_tokens[i] = MASK_TOKEN_ID
        was_masked[i] = True
    
    noise_details = []
    for i in noise_indices:
        random_token = random.randint(0, CODEBOOK_SIZE - 1)
        noise_details.append((i, original[i], random_token))
        masked_tokens[i] = random_token
    
    # Collect debug info
    debug_info = {
        'u0': u0,
        'u1': u1,
        'mask_ratio': mask_ratio,
        'num_mask': num_mask,
        'noise_ratio': noise_ratio,
        'num_noise': num_noise,
        'mask_indices': sorted(mask_indices),
        'noise_indices': sorted(noise_indices),
        'noise_details': noise_details,
        'kept_unchanged': [i for i in range(N) if i not in mask_indices and i not in noise_indices]
    }
    
    return masked_tokens, was_masked, debug_info


def print_matrix(tokens: List[int], H: int, W: int, title: str = ""):
    """Print tokens as a 2D matrix."""
    if title:
        print(f"\n{title}")
    for i in range(H):
        row = tokens[i*W:(i+1)*W]
        formatted = []
        for t in row:
            if t == MASK_TOKEN_ID:
                formatted.append("  M")
            elif t == 1:
                formatted.append("  1")
            else:
                formatted.append(f"{t:3d}")
        print("  [" + "][".join(formatted) + "]")


def print_debug_info(info: dict, frame_name: str):
    """Print detailed debug information."""
    print(f"\n  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │ {frame_name:^55} │")
    print(f"  └─────────────────────────────────────────────────────────┘")
    print(f"    Parameters:")
    print(f"      u₀ = {info['u0']:.4f} → mask_ratio = γ({info['u0']:.4f}) = {info['mask_ratio']:.4f}")
    print(f"      u₁ = {info['u1']:.4f} → noise_ratio = {info['u1']:.4f} × {NOISE_ETA}% = {info['noise_ratio']:.4f}")
    print(f"    Masking:")
    print(f"      Total tokens: N = {info['num_mask'] + len(info['kept_unchanged']) + info['num_noise']}")
    print(f"      num_mask = ⌈{info['mask_ratio']:.4f} × N⌉ = {info['num_mask']} tokens → MASK_ID({MASK_TOKEN_ID})")
    print(f"      Mask indices: {info['mask_indices']}")
    print(f"    Noise:")
    print(f"      Unmasked remaining: {len(info['kept_unchanged']) + info['num_noise']}")
    print(f"      num_noise = ⌊{info['noise_ratio']:.4f} × {len(info['kept_unchanged']) + info['num_noise']}⌋ = {info['num_noise']} tokens")
    if info['noise_details']:
        print(f"      Noise details (idx: original → random):")
        for idx, orig, new in info['noise_details']:
            print(f"        [{idx}]: {orig} → {new}")
    else:
        print(f"      No noise applied")
    print(f"    Kept unchanged: {info['kept_unchanged']}")


def test_objective_future_prediction():
    """Test Objective 1: Future Prediction."""
    print("\n" + "="*80)
    print("OBJECTIVE 1: FUTURE PREDICTION (Condition on past, predict future)")
    print("="*80)
    print("Setup: T=6 frames (3 past + 3 future), 4x4 grid, all tokens = 1 initially")
    
    T, H, W = 6, 4, 4
    N = H * W
    num_past = 3
    
    # Initialize all tokens to 1
    all_frames = [[1] * N for _ in range(T)]
    
    # Fixed random values for reproducibility
    u0_values = [0.3, 0.5, 0.7, 0.4, 0.6, 0.8]  # One per frame
    u1_values = [0.2, 0.4, 0.6, 0.3, 0.5, 0.7]
    
    print(f"\n┌{'─'*78}┐")
    print(f"│{'GROUND TRUTH: All tokens = 1':^78}│")
    print(f"└{'─'*78}┘")
    for t in range(T):
        frame_type = "PAST" if t < num_past else "FUTURE"
        print_matrix(all_frames[t], H, W, f"Frame {t} ({frame_type}, GT)")
    
    # Apply masking
    masked_frames = []
    all_debug_info = []
    
    print(f"\n{'─'*80}")
    print("APPLYING MASKING:")
    print(f"{'─'*80}")
    
    for t in range(T):
        if t < num_past:
            # Past frames: keep GT
            masked_frames.append(all_frames[t].copy())
            print(f"\n  Frame {t} (PAST): Kept as Ground Truth (unmasked)")
        else:
            # Future frames: apply masking
            masked, was_masked, debug_info = apply_random_masking_detailed(
                all_frames[t], u0_values[t], u1_values[t], f"Frame {t} (FUTURE)"
            )
            masked_frames.append(masked)
            all_debug_info.append((t, debug_info))
            print_debug_info(debug_info, f"Frame {t} (FUTURE)")
    
    # Print results
    print(f"\n{'─'*80}")
    print("RESULT: Masked Tokens")
    print(f"{'─'*80}")
    for t in range(T):
        frame_type = "PAST (GT)" if t < num_past else "FUTURE (MASKED)"
        print_matrix(masked_frames[t], H, W, f"Frame {t} ({frame_type})")
    
    # Show temporal mask
    print(f"\n{'─'*80}")
    print("TEMPORAL ATTENTION MASK (Causal - Lower Triangular):")
    print(f"{'─'*80}")
    print("      T0   T1   T2   T3   T4   T5")
    for i in range(T):
        row = []
        for j in range(T):
            if j > i:
                row.append("  - ")
            else:
                row.append("  0 ")
        print(f"  T{i} " + "".join(row))
    print("\n  0 = can attend, - = blocked")
    print("  Past frames attend to: themselves only")
    print("  Future frames attend to: all past + themselves + earlier future")


def test_objective_joint_denoise():
    """Test Objective 2: Joint Denoise."""
    print("\n" + "="*80)
    print("OBJECTIVE 2: JOINT DENOISE (Joint modeling of past and future)")
    print("="*80)
    print("Setup: T=6 frames, 4x4 grid, all tokens = 1 initially")
    
    T, H, W = 6, 4, 4
    N = H * W
    num_past = 3
    
    # Initialize all tokens to 1
    all_frames = [[1] * N for _ in range(T)]
    
    # Fixed random values
    u0_values = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
    u1_values = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
    
    print(f"\n┌{'─'*78}┐")
    print(f"│{'GROUND TRUTH: All tokens = 1':^78}│")
    print(f"└{'─'*78}┘")
    
    # Apply masking to ALL frames
    masked_frames = []
    
    print(f"\n{'─'*80}")
    print("APPLYING MASKING TO ALL FRAMES:")
    print(f"{'─'*80}")
    
    for t in range(T):
        masked, was_masked, debug_info = apply_random_masking_detailed(
            all_frames[t], u0_values[t], u1_values[t], f"Frame {t}"
        )
        masked_frames.append(masked)
        frame_type = "PAST" if t < num_past else "FUTURE"
        print_debug_info(debug_info, f"Frame {t} ({frame_type})")
    
    # Print results
    print(f"\n{'─'*80}")
    print("RESULT: All Frames Masked")
    print(f"{'─'*80}")
    for t in range(T):
        frame_type = "PAST" if t < num_past else "FUTURE"
        print_matrix(masked_frames[t], H, W, f"Frame {t} ({frame_type}, masked)")
    
    print(f"\nKey Difference from Future Prediction:")
    print(f"  • Past frames are ALSO masked (not kept as GT)")
    print(f"  • Model must jointly denoise entire sequence")
    print(f"  • Same causal temporal mask")


def test_objective_individual_denoise():
    """Test Objective 3: Individual Denoise."""
    print("\n" + "="*80)
    print("OBJECTIVE 3: INDIVIDUAL DENOISE (Model each frame individually)")
    print("="*80)
    print("Setup: T=6 frames, 4x4 grid, all tokens = 1 initially")
    
    T, H, W = 6, 4, 4
    N = H * W
    num_past = 3
    
    # Initialize all tokens to 1
    all_frames = [[1] * N for _ in range(T)]
    
    # Fixed random values
    u0_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    u1_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    
    print(f"\n┌{'─'*78}┐")
    print(f"│{'GROUND TRUTH: All tokens = 1':^78}│")
    print(f"└{'─'*78}┘")
    
    # Apply masking independently to each frame
    masked_frames = []
    
    print(f"\n{'─'*80}")
    print("APPLYING INDEPENDENT MASKING TO EACH FRAME:")
    print(f"{'─'*80}")
    
    for t in range(T):
        masked, was_masked, debug_info = apply_random_masking_detailed(
            all_frames[t], u0_values[t], u1_values[t], f"Frame {t}"
        )
        masked_frames.append(masked)
        frame_type = "PAST" if t < num_past else "FUTURE"
        print_debug_info(debug_info, f"Frame {t} ({frame_type}, INDEPENDENT)")
    
    # Print results
    print(f"\n{'─'*80}")
    print("RESULT: All Frames Masked Independently")
    print(f"{'─'*80}")
    for t in range(T):
        frame_type = "PAST" if t < num_past else "FUTURE"
        print_matrix(masked_frames[t], H, W, f"Frame {t} ({frame_type}, independent)")
    
    # Show temporal mask
    print(f"\n{'─'*80}")
    print("TEMPORAL ATTENTION MASK (Identity - Diagonal Only):")
    print(f"{'─'*80}")
    print("      T0   T1   T2   T3   T4   T5")
    for i in range(T):
        row = []
        for j in range(T):
            if i == j:
                row.append("  0 ")
            else:
                row.append("  - ")
        print(f"  T{i} " + "".join(row))
    print("\n  0 = can attend, - = blocked")
    print("  Each frame attends ONLY to itself (no temporal information)")
    print("  Purpose: Learn unconditional generation for Classifier-Free Guidance")


def compare_all_objectives():
    """Side-by-side comparison of all three objectives."""
    print("\n" + "="*80)
    print("SIDE-BY-SIDE COMPARISON: Same input, different masking")
    print("="*80)
    
    T, H, W = 6, 4, 4
    N = H * W
    num_past = 3
    
    # Same initial data and random seeds for fair comparison
    random.seed(42)
    
    # Initialize all tokens to 1
    all_frames = [[1] * N for _ in range(T)]
    
    # Use same u0/u1 for all to show the difference clearly
    u0 = 0.6
    u1 = 0.5
    
    print(f"\nCommon parameters:")
    print(f"  u₀ = {u0} → mask_ratio = {cosine_mask_schedule(u0):.4f}")
    print(f"  u₁ = {u1} → noise_ratio = {u1 * NOISE_ETA / 100:.4f}")
    
    # Future Prediction
    print(f"\n{'─'*80}")
    print("1. FUTURE PREDICTION:")
    future_pred = []
    for t in range(T):
        if t < num_past:
            future_pred.append(all_frames[t].copy())
        else:
            random.seed(42 + t)  # Deterministic for comparison
            masked, _, _ = apply_random_masking_detailed(all_frames[t], u0, u1)
            future_pred.append(masked)
    
    for t in range(T):
        frame_type = "PAST" if t < num_past else "FUTURE"
        status = "GT" if t < num_past else "MASKED"
        print_matrix(future_pred[t], H, W, f"  Frame {t} ({frame_type}, {status})")
    
    # Joint Denoise
    print(f"\n{'─'*80}")
    print("2. JOINT DENOISE:")
    joint = []
    for t in range(T):
        random.seed(42 + t)
        masked, _, _ = apply_random_masking_detailed(all_frames[t], u0, u1)
        joint.append(masked)
    
    for t in range(T):
        frame_type = "PAST" if t < num_past else "FUTURE"
        print_matrix(joint[t], H, W, f"  Frame {t} ({frame_type}, MASKED)")
    
    # Individual Denoise
    print(f"\n{'─'*80}")
    print("3. INDIVIDUAL DENOISE:")
    individual = []
    for t in range(T):
        random.seed(42 + t)
        masked, _, _ = apply_random_masking_detailed(all_frames[t], u0, u1)
        individual.append(masked)
    
    for t in range(T):
        frame_type = "PAST" if t < num_past else "FUTURE"
        print_matrix(individual[t], H, W, f"  Frame {t} ({frame_type}, MASKED)")
    
    # Summary
    print(f"\n{'='*80}")
    print("KEY DIFFERENCES SUMMARY:")
    print(f"{'='*80}")
    print("""
    Future Prediction:  Past = GT (all 1's), Future = Masked
                        └── Model learns to predict future from clean past
    
    Joint Denoise:      Past = Masked, Future = Masked  
                        └── Model learns joint denoising (harder task)
    
    Individual:         Same masking as Joint, but IDENTITY temporal mask
                        └── Each frame processed independently
                        └── Trains unconditional generation for CFG
    """)


def demonstrate_mask_schedule():
    """Demonstrate the cosine mask schedule."""
    print("\n" + "="*80)
    print("COSINE MASK SCHEDULE: γ(u) = cos(u·π/2)")
    print("="*80)
    
    print("\n  u values → γ(u) (mask ratio):")
    print("  " + "-" * 40)
    
    u_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    for u in u_values:
        gamma = cosine_mask_schedule(u)
        bar = "█" * int(gamma * 40)
        print(f"  u={u:.1f} → γ(u)={gamma:.4f} |{bar:<40s}|")
    
    print("\n  Interpretation:")
    print("    u ≈ 0 → γ(u) ≈ 1.0: Almost all tokens masked")
    print("    u = 0.5 → γ(u) ≈ 0.7: ~70% tokens masked")  
    print("    u ≈ 1 → γ(u) ≈ 0.0: Almost no tokens masked")
    print("\n  During training, u ~ Uniform(0,1), so mask ratio varies from 0 to 1")


if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    
    # Run all tests
    demonstrate_mask_schedule()
    test_objective_future_prediction()
    test_objective_joint_denoise()
    test_objective_individual_denoise()
    compare_all_objectives()
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
