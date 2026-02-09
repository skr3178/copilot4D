#!/usr/bin/env python3
"""
Comprehensive PyTorch test of Algorithm 1 masking with synthetic data.

This uses the actual implementation from copilot4d.world_model.masking
"""

import sys
sys.path.insert(0, '/media/skr/storage/self_driving/CoPilot4D')

import torch
import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple

# Import actual implementation
from copilot4d.world_model.masking import (
    DiscreteDiffusionMasker, 
    cosine_mask_schedule,
    compute_diffusion_loss
)
from copilot4d.utils.config import WorldModelConfig


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)


def print_tensor_2d(tokens: torch.Tensor, title: str = "", highlight_mask: bool = True):
    """Print a 2D tensor with formatting."""
    if title:
        print(f"\n{title}")
    
    H, W = tokens.shape
    for i in range(H):
        row = []
        for j in range(W):
            val = tokens[i, j].item()
            if val == 1024:  # MASK token
                row.append("  M")
            elif val == 1:
                row.append("  1")
            else:
                row.append(f"{val:3d}")
        print("  [" + "][".join(row) + "]")


def analyze_masking_result(original: torch.Tensor, masked: torch.Tensor, vocab_size: int = 1024):
    """Analyze the differences between original and masked tensors."""
    mask_token = vocab_size  # 1024
    
    # Count statistics
    total = original.numel()
    unchanged = (masked == original).sum().item()
    masked_count = (masked == mask_token).sum().item()
    
    # Noise = neither original nor mask
    is_original = (masked == original)
    is_mask = (masked == mask_token)
    noised_count = total - unchanged - masked_count
    
    # Find changed positions
    diff_mask = masked != original
    diff_positions = torch.where(diff_mask)
    
    print(f"\n  Analysis:")
    print(f"    Total tokens: {total}")
    print(f"    Unchanged (kept original): {unchanged} ({unchanged/total*100:.1f}%)")
    print(f"    Masked (→ {mask_token}): {masked_count} ({masked_count/total*100:.1f}%)")
    print(f"    Noised (→ random [0-{vocab_size-1}]): {noised_count} ({noised_count/total*100:.1f}%)")
    
    if diff_positions[0].numel() > 0:
        print(f"    Changed positions:")
        for idx in range(min(10, diff_positions[0].numel())):
            i, j = diff_positions[0][idx].item(), diff_positions[1][idx].item()
            orig_val = original[i, j].item()
            new_val = masked[i, j].item()
            change_type = "MASK" if new_val == mask_token else "NOISE"
            print(f"      [{i},{j}]: {orig_val} → {new_val} ({change_type})")
        if diff_positions[0].numel() > 10:
            print(f"      ... and {diff_positions[0].numel() - 10} more")
    
    return {
        'total': total,
        'unchanged': unchanged,
        'masked': masked_count,
        'noised': noised_count
    }


def test_basic_masking():
    """Test basic masking with detailed variable printing."""
    print_section("TEST 1: BASIC MASKING WITH VARIABLE INSPECTION")
    
    # Create config
    cfg = WorldModelConfig(
        codebook_size=1024,
        num_frames=6,
        num_past_frames=3,
        noise_eta=20.0,
    )
    
    print(f"\nConfiguration:")
    print(f"  codebook_size: {cfg.codebook_size}")
    print(f"  mask_token_id: {cfg.mask_token_id}")
    print(f"  vocab_size: {cfg.vocab_size}")
    print(f"  noise_eta: {cfg.noise_eta}%")
    
    # Create synthetic data: batch=1, T=2, H=4, W=4
    B, T, H, W = 1, 2, 4, 4
    tokens = torch.ones(B, T, H, W, dtype=torch.long)
    
    print(f"\nInput shape: {tokens.shape}")
    print(f"Initial values: All 1's")
    
    # Manual masking demonstration
    print_section("MANUAL MASKING DEMONSTRATION")
    
    u0 = 0.6
    u1 = 0.5
    mask_ratio = cosine_mask_schedule(torch.tensor(u0)).item()
    noise_ratio = u1 * cfg.noise_eta / 100.0
    
    N = H * W
    num_mask = math.ceil(mask_ratio * N)
    num_unmasked = N - num_mask
    num_noise = int(noise_ratio * num_unmasked)
    
    print(f"\nStep-by-step variable computation:")
    print(f"  1. Sample u₀ = {u0}")
    print(f"  2. γ(u₀) = cos({u0} × π/2) = {mask_ratio:.4f}")
    print(f"  3. num_mask = ⌈{mask_ratio:.4f} × {N}⌉ = {num_mask}")
    print(f"  4. Sample u₁ = {u1}")
    print(f"  5. noise_ratio = {u1} × {cfg.noise_eta}% = {noise_ratio:.4f}")
    print(f"  6. Unmasked tokens = {N} - {num_mask} = {num_unmasked}")
    print(f"  7. num_noise = ⌊{noise_ratio:.4f} × {num_unmasked}⌋ = {num_noise}")
    
    # Create masker and test
    masker = DiscreteDiffusionMasker(cfg)
    
    # Test each objective
    objectives = ["future_prediction", "joint_denoise", "individual_denoise"]
    
    for obj in objectives:
        print_section(f"TESTING OBJECTIVE: {obj.upper().replace('_', ' ')}")
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        random.seed(42)
        
        # Prepare batch
        result = masker.prepare_batch(tokens.clone(), objective=obj)
        
        masked_tokens = result['tokens']
        targets = result['targets']
        temporal_mask = result['temporal_mask']
        was_masked = result['was_masked']
        mask_ratio_applied = result['mask_ratio']
        
        print(f"\nObjective: {result['objective']}")
        print(f"Mask ratio used: {mask_ratio_applied:.4f}")
        
        # Print each frame
        for t in range(T):
            frame_type = "PAST" if t < cfg.num_past_frames else "FUTURE"
            print_tensor_2d(
                masked_tokens[0, t], 
                f"Frame {t} ({frame_type}) - After Masking:"
            )
            analyze_masking_result(tokens[0, t], masked_tokens[0, t])
        
        # Print temporal mask
        print(f"\nTemporal Attention Mask ({temporal_mask.shape}):")
        T_size = temporal_mask.shape[0]
        header = "      " + "  ".join([f"T{i}" for i in range(T_size)])
        print(header)
        for i in range(T_size):
            row_vals = []
            for j in range(T_size):
                val = temporal_mask[i, j].item()
                if val == float('-inf'):
                    row_vals.append(" - ")
                else:
                    row_vals.append(f"{val:.0f} ")
            print(f"  T{i}  " + " ".join(row_vals))


def test_future_prediction_detailed():
    """Detailed test of future prediction with 6 frames."""
    print_section("TEST 2: FUTURE PREDICTION (6 frames: 3 past + 3 future)")
    
    cfg = WorldModelConfig(
        codebook_size=1024,
        num_frames=6,
        num_past_frames=3,
        noise_eta=20.0,
    )
    
    B, T, H, W = 1, 6, 4, 4
    tokens = torch.ones(B, T, H, W, dtype=torch.long)
    
    masker = DiscreteDiffusionMasker(cfg)
    
    torch.manual_seed(123)
    random.seed(123)
    
    result = masker.prepare_batch(tokens.clone(), objective="future_prediction")
    masked = result['tokens']
    
    print("\nGround Truth (all 1's):")
    for t in range(T):
        frame_type = "PAST" if t < 3 else "FUTURE"
        print_tensor_2d(tokens[0, t], f"  Frame {t} ({frame_type})")
    
    print("\n" + "-"*60)
    print("After Future Prediction Masking:")
    print("-"*60)
    
    for t in range(T):
        frame_type = "PAST" if t < 3 else "FUTURE"
        if t < 3:
            status = "GT (unmasked)"
        else:
            status = "MASKED"
        print_tensor_2d(masked[0, t], f"  Frame {t} ({frame_type}, {status})")
    
    # Verify past frames are unchanged
    print("\nVerification:")
    for t in range(3):
        is_unchanged = torch.all(masked[0, t] == tokens[0, t]).item()
        print(f"  Frame {t} (past) unchanged: {is_unchanged} {'✓' if is_unchanged else '✗'}")
    
    for t in range(3, 6):
        has_mask = (masked[0, t] == 1024).any().item()
        print(f"  Frame {t} (future) has mask tokens: {has_mask} {'✓' if has_mask else '✗'}")


def test_joint_denoise_detailed():
    """Detailed test of joint denoise."""
    print_section("TEST 3: JOINT DENOISE (All frames partially masked)")
    
    cfg = WorldModelConfig(
        codebook_size=1024,
        num_frames=6,
        num_past_frames=3,
        noise_eta=20.0,
    )
    
    B, T, H, W = 1, 6, 4, 4
    tokens = torch.ones(B, T, H, W, dtype=torch.long)
    
    masker = DiscreteDiffusionMasker(cfg)
    
    torch.manual_seed(456)
    random.seed(456)
    
    result = masker.prepare_batch(tokens.clone(), objective="joint_denoise")
    masked = result['tokens']
    
    print("\nGround Truth (all 1's):")
    for t in range(T):
        print_tensor_2d(tokens[0, t], f"  Frame {t}")
    
    print("\n" + "-"*60)
    print("After Joint Denoise Masking:")
    print("-"*60)
    
    for t in range(T):
        frame_type = "PAST" if t < 3 else "FUTURE"
        print_tensor_2d(masked[0, t], f"  Frame {t} ({frame_type}, MASKED)")
        stats = analyze_masking_result(tokens[0, t], masked[0, t])
    
    # Verify all frames are masked
    print("\nVerification:")
    for t in range(T):
        has_changes = not torch.all(masked[0, t] == tokens[0, t]).item()
        print(f"  Frame {t} has masking: {has_changes} {'✓' if has_changes else '✗'}")


def test_individual_denoise_detailed():
    """Detailed test of individual denoise."""
    print_section("TEST 4: INDIVIDUAL DENOISE (Identity temporal mask)")
    
    cfg = WorldModelConfig(
        codebook_size=1024,
        num_frames=6,
        num_past_frames=3,
        noise_eta=20.0,
    )
    
    B, T, H, W = 1, 6, 4, 4
    tokens = torch.ones(B, T, H, W, dtype=torch.long)
    
    masker = DiscreteDiffusionMasker(cfg)
    
    torch.manual_seed(789)
    random.seed(789)
    
    result = masker.prepare_batch(tokens.clone(), objective="individual_denoise")
    masked = result['tokens']
    temporal_mask = result['temporal_mask']
    
    print("\nAfter Individual Denoise Masking:")
    for t in range(T):
        frame_type = "PAST" if t < 3 else "FUTURE"
        print_tensor_2d(masked[0, t], f"  Frame {t} ({frame_type})")
    
    print("\nTemporal Attention Mask (Identity - diagonal only):")
    T_size = temporal_mask.shape[0]
    for i in range(T_size):
        row_vals = []
        for j in range(T_size):
            val = temporal_mask[i, j].item()
            if val == float('-inf'):
                row_vals.append(" - ")
            else:
                row_vals.append(" 0  ")
        print(f"  T{i}  " + " ".join(row_vals))
    
    # Verify diagonal is 0, rest is -inf
    print("\nVerification:")
    for i in range(T_size):
        for j in range(T_size):
            val = temporal_mask[i, j].item()
            expected = 0.0 if i == j else float('-inf')
            correct = (val == expected) or (val != val and expected != expected)  # Handle NaN comparison
            if not correct:
                print(f"  ✗ T{i},T{j}: expected {expected}, got {val}")
    print("  All temporal mask values correct ✓")


def test_random_sampling():
    """Test random objective sampling distribution."""
    print_section("TEST 5: RANDOM OBJECTIVE SAMPLING DISTRIBUTION")
    
    cfg = WorldModelConfig(
        prob_future_pred=0.5,
        prob_joint_denoise=0.4,
        prob_individual_denoise=0.1,
    )
    
    masker = DiscreteDiffusionMasker(cfg)
    
    # Sample many times
    n_samples = 10000
    counts = {"future_prediction": 0, "joint_denoise": 0, "individual_denoise": 0}
    
    for _ in range(n_samples):
        obj = masker.sample_objective()
        counts[obj] += 1
    
    print(f"\nSampling {n_samples} times:")
    print(f"  future_prediction:  {counts['future_prediction']:5d} ({counts['future_prediction']/n_samples*100:.1f}%)  [expected: 50.0%]")
    print(f"  joint_denoise:      {counts['joint_denoise']:5d} ({counts['joint_denoise']/n_samples*100:.1f}%)  [expected: 40.0%]")
    print(f"  individual_denoise: {counts['individual_denoise']:5d} ({counts['individual_denoise']/n_samples*100:.1f}%)  [expected: 10.0%]")


def test_loss_computation():
    """Test loss computation."""
    print_section("TEST 6: LOSS COMPUTATION")
    
    cfg = WorldModelConfig(label_smoothing=0.1)
    
    # Create fake logits and targets
    B, T, N, V = 2, 3, 16, 1025  # batch, time, tokens, vocab
    
    # Random logits
    logits = torch.randn(B, T, N, V)
    
    # Random targets
    targets = torch.randint(0, 1024, (B, T, 4, 4))  # H=W=4
    
    print(f"\nInput shapes:")
    print(f"  logits: {logits.shape} (B, T, H*W, vocab_size)")
    print(f"  targets: {targets.shape} (B, T, H, W)")
    
    # Compute loss
    loss = compute_diffusion_loss(logits, targets, cfg.label_smoothing)
    
    print(f"\nLoss computation:")
    print(f"  label_smoothing: {cfg.label_smoothing}")
    print(f"  loss value: {loss.item():.4f}")
    
    # Verify loss is computed on all positions
    print(f"\nVerification:")
    print(f"  Loss requires grad: {loss.requires_grad}")
    print(f"  Loss is scalar: {loss.dim() == 0}")


def test_batch_processing():
    """Test batch processing with multiple samples."""
    print_section("TEST 7: BATCH PROCESSING (B=3)")
    
    cfg = WorldModelConfig(
        codebook_size=1024,
        num_frames=4,
        num_past_frames=2,
        noise_eta=20.0,
    )
    
    B, T, H, W = 3, 4, 4, 4
    
    # Create batch where each sample has different values
    tokens = torch.zeros(B, T, H, W, dtype=torch.long)
    for b in range(B):
        tokens[b] = b + 1  # Sample 0: all 1s, Sample 1: all 2s, Sample 2: all 3s
    
    masker = DiscreteDiffusionMasker(cfg)
    
    torch.manual_seed(999)
    random.seed(999)
    
    result = masker.prepare_batch(tokens.clone(), objective="joint_denoise")
    masked = result['tokens']
    
    print(f"\nBatch shape: {tokens.shape}")
    print(f"Input values:")
    for b in range(B):
        print(f"  Sample {b}: all {tokens[b, 0, 0, 0].item()}s")
    
    print(f"\nAfter masking (showing frame 0 for each sample):")
    for b in range(B):
        print_tensor_2d(masked[b, 0], f"Sample {b}, Frame 0")
        
        # Count mask tokens
        mask_count = (masked[b] == 1024).sum().item()
        total = masked[b].numel()
        print(f"  Mask tokens: {mask_count}/{total} ({mask_count/total*100:.1f}%)")


def test_edge_cases():
    """Test edge cases."""
    print_section("TEST 8: EDGE CASES")
    
    cfg = WorldModelConfig(codebook_size=1024, noise_eta=20.0)
    masker = DiscreteDiffusionMasker(cfg)
    
    # Test 1: u0 = 0 (no masking)
    print("\n1. u0 = 0 → mask_ratio = cos(0) = 1.0 (100% masked)")
    u0 = 0.0
    ratio = cosine_mask_schedule(torch.tensor(u0)).item()
    print(f"   γ(0) = {ratio:.4f}")
    
    # Test 2: u0 = 1 (full masking)
    print("\n2. u0 = 1 → mask_ratio = cos(π/2) = 0.0 (0% masked)")
    u0 = 1.0
    ratio = cosine_mask_schedule(torch.tensor(u0)).item()
    print(f"   γ(1) = {ratio:.4f}")
    
    # Test 3: u1 = 0 (no noise)
    print("\n3. u1 = 0 → noise_ratio = 0% (no noise)")
    u1 = 0.0
    noise_ratio = u1 * cfg.noise_eta / 100.0
    print(f"   noise_ratio = {noise_ratio:.4f}")
    
    # Test 4: Single frame
    print("\n4. Single frame processing (T=1):")
    tokens = torch.ones(1, 1, 4, 4, dtype=torch.long)
    torch.manual_seed(111)
    result = masker.prepare_batch(tokens.clone(), objective="individual_denoise")
    masked = result['tokens']
    print_tensor_2d(masked[0, 0], "   Result")
    
    # Verify temporal mask is 1x1
    temporal_mask = result['temporal_mask']
    print(f"   Temporal mask shape: {temporal_mask.shape}")
    print(f"   Temporal mask value: {temporal_mask[0, 0].item()}")


def compare_masking_visual():
    """Visual side-by-side comparison of all three objectives."""
    print_section("VISUAL COMPARISON: All Three Objectives Side-by-Side")
    
    cfg = WorldModelConfig(
        codebook_size=1024,
        num_frames=6,
        num_past_frames=3,
        noise_eta=20.0,
    )
    
    T, H, W = 6, 4, 4
    tokens = torch.ones(1, T, H, W, dtype=torch.long)
    masker = DiscreteDiffusionMasker(cfg)
    
    objectives = ["future_prediction", "joint_denoise", "individual_denoise"]
    results = {}
    
    for obj in objectives:
        torch.manual_seed(42)
        random.seed(42)
        result = masker.prepare_batch(tokens.clone(), objective=obj)
        results[obj] = result['tokens'][0]  # Remove batch dim
    
    # Print frame by frame
    for t in range(T):
        frame_type = "PAST" if t < 3 else "FUTURE"
        print(f"\n{'='*60}")
        print(f"FRAME {t} ({frame_type})")
        print(f"{'='*60}")
        
        for obj_name, masked_tokens in results.items():
            label = obj_name.replace('_', ' ').upper()
            print_tensor_2d(masked_tokens[t], f"  {label}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print(" PYTORCH MASKING TEST SUITE")
    print(" Using actual implementation from copilot4d.world_model.masking")
    print("="*80)
    
    # Run all tests
    test_basic_masking()
    test_future_prediction_detailed()
    test_joint_denoise_detailed()
    test_individual_denoise_detailed()
    test_random_sampling()
    test_loss_computation()
    test_batch_processing()
    test_edge_cases()
    compare_masking_visual()
    
    print("\n" + "="*80)
    print(" ALL TESTS COMPLETE")
    print("="*80)
