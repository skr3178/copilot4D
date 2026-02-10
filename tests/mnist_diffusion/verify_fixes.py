#!/usr/bin/env python
"""
Verification script for Moving MNIST fixes.
Tests ego-centric actions and model overfitting capability.
"""
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.mnist_diffusion.moving_mnist_precomputed import MovingMNISTPrecomputed
from tests.mnist_diffusion.simple_model import SimpleVideoTransformer


def verify_actions(use_ego_centric=False):
    """Check actions are meaningful and have variance."""
    print("=" * 60)
    print(f"Verifying Actions ({'Ego-Centric' if use_ego_centric else 'Standard'})")
    print("=" * 60)
    
    ds = MovingMNISTPrecomputed(
        data_path='mnist_test_seq.1.npy',
        num_sequences=10,
        frame_size=32,
        use_ego_centric=use_ego_centric,
        ego_digit_id=0,
    )
    
    sample = ds[0]
    actions = sample['actions']
    
    print(f"Action shape: {actions.shape}")
    print(f"Action range: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"Action mean: {actions.mean():.3f}")
    print(f"Action std: {actions.std():.3f}")
    
    # Check correlation with actual motion
    frames = sample['frames']
    print(f"\nFrame-to-frame motion vs actions:")
    for t in range(min(5, len(frames) - 1)):
        # Compute frame difference as proxy for motion
        diff = (frames[t+1] - frames[t]).abs().mean()
        
        if use_ego_centric:
            # For ego-centric: action magnitude
            action_mag = actions[t].norm()
            print(f"  Frame {t}: PixelDiff={diff:.3f}, ActionMag={action_mag:.3f}, "
                  f"Action=[{actions[t][0]:+.2f}, {actions[t][1]:+.2f}]")
        else:
            # For one-hot: which direction
            action_idx = actions[t].argmax()
            action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NONE']
            action_name = action_names[action_idx] if actions[t].sum() > 0 else 'NONE'
            print(f"  Frame {t}: PixelDiff={diff:.3f}, Action={action_name}")
    
    # Verify actions have variance (not all zeros)
    if actions.std() < 0.01:
        print("\n❌ WARNING: Actions have very low variance!")
        return False
    else:
        print(f"\n✅ Actions have good variance (std={actions.std():.3f})")
        return True


def verify_overfit(use_ego_centric=False, num_steps=200):
    """Overfit single batch to verify model can learn."""
    print("\n" + "=" * 60)
    print("Overfit Test (Single Sequence)")
    print("=" * 60)
    
    action_dim = 2 if use_ego_centric else 4
    
    # Load single sequence
    ds = MovingMNISTPrecomputed(
        data_path='mnist_test_seq.1.npy',
        num_sequences=1,
        frame_size=32,
        use_ego_centric=use_ego_centric,
        ego_digit_id=0,
    )
    
    batch = ds[0]
    
    # Add batch dimension
    tokens = batch['tokens'].unsqueeze(0)    # [1, 20, 32, 32]
    actions = batch['actions'].unsqueeze(0)  # [1, 20, action_dim]
    
    print(f"Token shape: {tokens.shape}")
    print(f"Action shape: {actions.shape}")
    
    # Create small model for fast testing
    model = SimpleVideoTransformer(
        vocab_size=16,
        mask_token_id=16,
        num_frames=20,
        height=32,
        width=32,
        embed_dim=128,    # Small for fast convergence
        num_layers=2,     # Minimal layers
        num_heads=4,
        action_dim=action_dim,
        dropout=0.0,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    tokens = tokens.to(device)
    actions = actions.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params/1e3:.1f}K")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"\nOverfitting for {num_steps} steps...")
    print("-" * 40)
    
    # Use future prediction objective (simplest)
    num_past = 10
    past_tokens = tokens[:, :num_past]
    future_tokens = tokens[:, num_past:]
    past_actions = actions[:, :num_past]
    future_actions = actions[:, num_past:]
    
    losses = []
    accuracies = []
    
    for i in range(num_steps):
        # Corrupt future tokens (masking)
        mask_ratio = 0.5 + 0.3 * (i / num_steps)  # Increase difficulty
        
        future_masked = future_tokens.clone()
        mask = torch.rand_like(future_masked.float()) < mask_ratio
        future_masked[mask] = 16  # MASK token
        
        # Concatenate
        full_tokens = torch.cat([past_tokens, future_masked], dim=1)
        full_actions = torch.cat([past_actions, future_actions], dim=1)
        
        # Create causal mask
        T = full_tokens.shape[1]
        causal_mask = torch.full((T, T), float("-inf"), device=device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        
        # Forward
        logits = model(full_tokens, full_actions, causal_mask)
        
        # Loss on future only
        future_logits = logits[:, num_past:]  # [1, T_future, N, V]
        B, T_f, N, V = future_logits.shape
        
        future_logits_flat = future_logits.reshape(B * T_f * N, V)
        future_targets_flat = future_tokens.reshape(-1)
        
        loss = F.cross_entropy(future_logits_flat, future_targets_flat)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            pred = future_logits.argmax(dim=-1)  # [B, T_future, N]
            # Reshape future_tokens to match
            future_tokens_flat = future_tokens.reshape(B, T_f, N)
            acc = (pred == future_tokens_flat).float().mean()
        
        losses.append(loss.item())
        accuracies.append(acc.item())
        
        if i % 40 == 0 or i == num_steps - 1:
            print(f"Step {i:3d}: Loss={loss.item():.4f}, Acc={acc.item():.3f}, Mask={mask_ratio:.2f}")
    
    # Final evaluation
    final_acc = accuracies[-1]
    print("-" * 40)
    
    if final_acc > 0.8:
        print(f"✅ Overfit test PASSED (Acc={final_acc:.3f} > 0.8)")
        print("   Model can memorize - architecture is working")
        return True
    elif final_acc > 0.5:
        print(f"⚠️  Overfit test PARTIAL (Acc={final_acc:.3f})")
        print("   Model learning but slowly - may need more steps")
        return True
    else:
        print(f"❌ Overfit test FAILED (Acc={final_acc:.3f} < 0.5)")
        print("   Check: action injection, masking, loss computation")
        return False


def verify_joint_modeling_mask():
    """Verify joint modeling uses bidirectional attention."""
    print("\n" + "=" * 60)
    print("Joint Modeling Mask Test")
    print("=" * 60)
    
    from train_mnist_full import DiscreteDiffusionMasker
    
    masker = DiscreteDiffusionMasker(vocab_size=16, mask_token_id=16)
    
    # Create test tokens
    B, T, H, W = 2, 5, 4, 4
    tokens = torch.randint(0, 16, (B, T, H, W))
    
    # Test causal mask (future prediction)
    causal_mask = masker._make_causal_mask(T, tokens.device)
    print(f"Causal mask shape: {causal_mask.shape}")
    print(f"Causal mask upper triangle should be -inf:")
    print(causal_mask[:3, :3])
    
    # Test identity mask (individual)
    identity_mask = masker._make_identity_mask(T, tokens.device)
    print(f"\nIdentity mask shape: {identity_mask.shape}")
    print(f"Identity mask diagonal should be 0, rest -inf:")
    print(identity_mask[:3, :3])
    
    # Test bidirectional mask
    bidir_mask = masker._make_bidirectional_mask(T, tokens.device)
    print(f"\nBidirectional mask shape: {bidir_mask.shape}")
    print(f"Bidirectional mask should be all zeros:")
    print(bidir_mask[:3, :3])
    
    # Verify joint denoise uses bidirectional (no mask or all zeros)
    batch = masker.prepare_batch(tokens, objective="joint_denoise")
    temporal_mask = batch["temporal_mask"]
    
    if temporal_mask is None:
        print("\n✅ Joint modeling uses bidirectional (None = no mask)")
        return True
    elif temporal_mask.abs().sum() == 0:
        print("\n✅ Joint modeling uses all-zeros mask (bidirectional)")
        return True
    else:
        print(f"\n⚠️  Joint modeling has unexpected mask")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ego', action='store_true', help='Use ego-centric actions')
    parser.add_argument('--steps', type=int, default=200, help='Training steps for overfit test')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("MOVING MNIST FIX VERIFICATION")
    print("=" * 60)
    
    results = []
    
    # Test 1: Actions
    results.append(("Actions", verify_actions(use_ego_centric=args.ego)))
    
    # Test 2: Joint modeling mask
    results.append(("Joint Mask", verify_joint_modeling_mask()))
    
    # Test 3: Overfitting
    results.append(("Overfit", verify_overfit(use_ego_centric=args.ego, num_steps=args.steps)))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:<15} {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Ready for full training.")
        print("\nNext steps:")
        print("  1. Train with: python train_mnist_full.py --use_ego_centric")
        print("  2. Monitor validation loss")
        print("  3. Check sampling quality after epoch 5+")
    else:
        print("\n⚠️  Some tests failed. Review output above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
