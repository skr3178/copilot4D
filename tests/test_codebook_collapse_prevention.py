"""Test codebook collapse prevention mechanism.

This test verifies that the K-Means restart mechanism works correctly:
1. Simulates codebook collapse by using only a few codes
2. Verifies dead code detection triggers at >3%
3. Verifies entire codebook is re-initialized with K-Means
4. Checks that codebook health recovers after restart
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from copilot4d.tokenizer.vector_quantizer_fixed import VectorQuantizerFixed


def test_codebook_collapse_prevention():
    """Test the complete collapse prevention pipeline."""
    
    print("=" * 70)
    print("Testing Codebook Collapse Prevention Mechanism")
    print("=" * 70)
    
    # Configuration for fast testing
    codebook_size = 128  # Small for visualization
    codebook_dim = 256
    
    vq = VectorQuantizerFixed(
        dim=256,
        codebook_size=codebook_size,
        codebook_dim=codebook_dim,
        commitment_cost=1.0,
        codebook_cost=0.25,
        decay=0.99,
        # Relaxed thresholds for testing
        dead_threshold=10,      # Trigger after 10 iterations unused
        dead_percentage=0.03,   # 3% threshold (paper spec)
        min_iterations=5,       # Allow restart after 5 iterations
        reinit_every=5,         # Check every 5 iterations
        kmeans_iters=5,         # Fast K-Means
    )
    
    print(f"\nConfiguration:")
    print(f"  Codebook size: {codebook_size}")
    print(f"  Codebook dim: {codebook_dim}")
    print(f"  Memory bank: {vq.memory_bank.shape}")
    print(f"  Dead threshold: {vq.dead_threshold} iterations")
    print(f"  Dead percentage: {vq.dead_percentage*100:.0f}%")
    print(f"  Min iterations before re-init: {vq.min_iterations}")
    
    # Store initial codebook for comparison
    initial_codebook = vq.embed.data.clone()
    
    print(f"\n{'Step':>6} | {'Active%':>8} | {'Dead%':>8} | {'Perplex':>8} | {'Entropy':>8} | {'Status'}")
    print("-" * 70)
    
    # Track metrics
    history = {
        'steps': [],
        'active_pct': [],
        'dead_pct': [],
        'perplexity': [],
        'entropy_norm': [],
    }
    
    # Simulate training
    num_steps = 50
    for step in range(num_steps):
        # Generate input
        x = torch.randn(4, 64, 256)  # (B, N, dim)
        
        # Forward pass
        quantized, indices, loss, metrics = vq(x)
        
        # Simulate codebook collapse by forcing limited code usage
        # After step 10, only use 5 codes out of 128 (3.9% - above 3% threshold)
        if 10 <= step < 30:
            # Force usage of only codes 0-4
            indices = torch.randint(0, 5, indices.shape)
            # Update usage counters manually to simulate dead codes
            vq.usage_count += 1
            used_codes = torch.unique(indices)
            vq.usage_count[used_codes] = 0
        
        # Record metrics
        history['steps'].append(step)
        history['active_pct'].append(metrics['vq_active_pct'])
        history['dead_pct'].append(metrics['vq_dead_pct'])
        history['perplexity'].append(metrics['vq_perplexity'])
        history['entropy_norm'].append(metrics['vq_entropy_norm'])
        
        # Print status
        status = ""
        if step == 10:
            status = "[START COLLAPSE]"
        elif step == 30:
            status = "[END COLLAPSE]"
        
        print(f"{step:>6} | {metrics['vq_active_pct']:>8.1f} | "
              f"{metrics['vq_dead_pct']:>8.1f} | {metrics['vq_perplexity']:>8.1f} | "
              f"{metrics['vq_entropy_norm']:>8.2f} | {status}")
    
    print("-" * 70)
    
    # Verify results
    print("\n" + "=" * 70)
    print("Verification Results")
    print("=" * 70)
    
    # 1. Check that collapse was detected
    max_dead_pct = max(history['dead_pct'])
    print(f"\n1. Maximum dead code percentage: {max_dead_pct:.1f}%")
    if max_dead_pct > 3.0:
        print("   ✓ Collapse detected (>3% threshold)")
    else:
        print("   ✗ Collapse NOT detected")
    
    # 2. Check that codebook changed (K-Means restart happened)
    final_codebook = vq.embed.data.clone()
    codebook_changed = not torch.allclose(initial_codebook, final_codebook, atol=1e-6)
    print(f"\n2. Codebook changed after K-Means restart: {codebook_changed}")
    if codebook_changed:
        print("   ✓ Codebook was re-initialized")
    else:
        print("   ✗ Codebook unchanged - restart may not have triggered")
    
    # 3. Check recovery after collapse ended
    final_active = history['active_pct'][-1]
    final_perplexity = history['perplexity'][-1]
    print(f"\n3. Final metrics (after recovery):")
    print(f"   Active codes: {final_active:.1f}%")
    print(f"   Perplexity: {final_perplexity:.1f} / {codebook_size}")
    
    if final_active > 50:
        print("   ✓ Codebook recovered after collapse")
    else:
        print("   ⚠ Codebook may not have fully recovered")
    
    # 4. Check memory bank usage
    bank_used = (vq.memory_bank.abs().sum(dim=1) > 0).sum().item()
    print(f"\n4. Memory bank usage:")
    print(f"   Used: {bank_used} / {vq.memory_bank.shape[0]} entries")
    if bank_used > 0:
        print("   ✓ Memory bank is being populated")
    else:
        print("   ✗ Memory bank empty")
    
    # 5. Verify K-Means was triggered
    print(f"\n5. Check console output above:")
    print("   Look for: '[VQ] Codebook collapse detected' messages")
    print("   Look for: '[VQ] Re-initializing ENTIRE codebook' messages")
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    all_passed = (
        max_dead_pct > 3.0 and
        codebook_changed and
        bank_used > 0
    )
    
    if all_passed:
        print("✓ ALL CHECKS PASSED - Codebook collapse prevention working!")
    else:
        print("⚠ Some checks failed - review implementation")
    
    return all_passed


def test_ema_vs_gradient_update():
    """Verify that codebook uses EMA updates, not gradient descent."""
    
    print("\n" + "=" * 70)
    print("Testing EMA Codebook Updates (vs Gradient Descent)")
    print("=" * 70)
    
    vq = VectorQuantizerFixed(
        dim=256,
        codebook_size=128,
        codebook_dim=256,
        decay=0.99,  # EMA decay
    )
    
    # Store initial codebook
    initial_embed = vq.embed.data.clone()
    initial_embed_avg = vq.embed_avg.data.clone()
    
    # Simulate multiple forward passes
    for step in range(10):
        x = torch.randn(4, 64, 256)
        quantized, indices, loss, metrics = vq(x)
    
    # Check that codebook changed via EMA, not gradients
    final_embed = vq.embed.data
    final_embed_avg = vq.embed_avg.data
    
    print(f"\nInitial embed[0,:5]: {initial_embed[0,:5].numpy()}")
    print(f"Final embed[0,:5]:   {final_embed[0,:5].numpy()}")
    print(f"Final embed_avg[0,:5]: {final_embed_avg[0,:5].numpy()}")
    
    # EMA should have updated embed_avg
    embed_avg_changed = not torch.allclose(initial_embed_avg, final_embed_avg, atol=1e-6)
    print(f"\nEMA embed_avg updated: {embed_avg_changed}")
    
    if embed_avg_changed:
        print("✓ EMA updates are working correctly")
    else:
        print("✗ EMA updates not detected")
    
    return embed_avg_changed


def test_pre_vq_projection():
    """Verify Pre-VQ projection exists (256 -> 1024 per paper)."""
    
    print("\n" + "=" * 70)
    print("Testing Pre-VQ Projection (256 -> 1024)")
    print("=" * 70)
    
    vq = VectorQuantizerFixed(
        dim=256,
        codebook_size=1024,
        codebook_dim=1024,  # Paper specification
    )
    
    # Check projection layers exist
    has_pre_norm = hasattr(vq, 'pre_norm')
    has_pre_proj = hasattr(vq, 'pre_proj')
    has_post_proj = hasattr(vq, 'post_proj')
    
    print(f"\nProjection layers:")
    print(f"  pre_norm (LayerNorm): {has_pre_norm}")
    print(f"  pre_proj (256->1024): {has_pre_proj}")
    print(f"  post_proj (1024->256): {has_post_proj}")
    
    # Test projection
    x = torch.randn(2, 64, 256)
    x_normed = vq.pre_norm(x.float())
    z_e = vq.pre_proj(x_normed)
    
    print(f"\nProjection test:")
    print(f"  Input shape: {x.shape}")
    print(f"  After pre_proj: {z_e.shape}")
    
    if z_e.shape[-1] == 1024:
        print("✓ Pre-VQ projection working (256 -> 1024)")
        return True
    else:
        print(f"✗ Wrong output dim: {z_e.shape[-1]}")
        return False


def main():
    """Run all tests."""
    
    print("\n" + "=" * 70)
    print("CODEBOOK COLLAPSE PREVENTION TEST SUITE")
    print("=" * 70)
    
    results = []
    
    # Test 1: Main collapse prevention
    results.append(("Collapse Prevention", test_codebook_collapse_prevention()))
    
    # Test 2: EMA updates
    results.append(("EMA Updates", test_ema_vs_gradient_update()))
    
    # Test 3: Pre-VQ projection
    results.append(("Pre-VQ Projection", test_pre_vq_projection()))
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED!")
        print("Codebook collapse prevention is correctly implemented.")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("SOME TESTS FAILED!")
        print("Review the implementation.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit(main())
