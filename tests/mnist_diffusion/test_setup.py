"""Quick test to verify the Moving MNIST diffusion setup works."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np

from tests.mnist_diffusion.moving_mnist_precomputed import MovingMNISTPrecomputed
from tests.mnist_diffusion.simple_model import SimpleVideoTransformer
from tests.mnist_diffusion.train_diffusion import DiscreteDiffusionMasker, compute_diffusion_loss


def test_dataset():
    """Test the dataset loads correctly."""
    print("=" * 60)
    print("Testing Dataset...")
    print("=" * 60)
    
    # Find data file
    possible_paths = [
        "mnist_test_seq.1.npy",
        "data/mnist_test_seq.1.npy",
        "mnist_test_seq.npy",
    ]
    
    data_path = None
    for path in possible_paths:
        if Path(path).exists():
            data_path = path
            break
    
    if data_path is None:
        print("ERROR: Could not find mnist_test_seq.npy file")
        return False
    
    print(f"Using data file: {data_path}")
    
    # Load small subset (non-preload mode to save memory)
    dataset = MovingMNISTPrecomputed(
        data_path=data_path,
        seq_len=20,
        num_sequences=10,
        num_token_levels=16,
        preload=False,  # Use memory-mapped loading
    )
    
    sample = dataset[0]
    print(f"✓ Frames shape: {sample['frames'].shape}")
    print(f"✓ Tokens shape: {sample['tokens'].shape}")
    print(f"✓ Actions shape: {sample['actions'].shape}")
    
    # Check token range
    assert sample['tokens'].min() >= 0, "Tokens below 0"
    assert sample['tokens'].max() < 16, "Tokens above 15"
    print(f"✓ Token range: [{sample['tokens'].min()}, {sample['tokens'].max()}]")
    
    print("Dataset test PASSED!")
    return True


def test_model():
    """Test the model forward pass."""
    print("\n" + "=" * 60)
    print("Testing Model...")
    print("=" * 60)
    
    # Use smaller dimensions for testing
    model = SimpleVideoTransformer(
        vocab_size=16,
        mask_token_id=16,
        num_frames=10,
        height=32,
        width=32,
        embed_dim=64,  # Smaller for testing
        num_layers=2,
        num_heads=4,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {total_params:,}")
    
    # Test forward pass
    B, T, H, W = 2, 10, 32, 32
    tokens = torch.randint(0, 17, (B, T, H, W))
    actions = torch.randn(B, T, 4)
    
    logits = model(tokens, actions)
    print(f"✓ Output shape: {logits.shape}")
    
    assert logits.shape == (B, T, H*W, 17), f"Wrong output shape: {logits.shape}"
    print("Model test PASSED!")
    return True


def test_masking():
    """Test the discrete diffusion masking."""
    print("\n" + "=" * 60)
    print("Testing Masking...")
    print("=" * 60)
    
    masker = DiscreteDiffusionMasker(
        vocab_size=16,
        mask_token_id=16,
        num_past_frames=5,
        noise_eta=5.0,
    )
    
    # Create dummy batch (smaller for memory)
    B, T, H, W = 2, 10, 32, 32
    tokens = torch.randint(0, 16, (B, T, H, W))
    
    # Test each objective
    for objective in ["future_prediction", "joint_denoise", "individual_denoise"]:
        masked_batch = masker.prepare_batch(tokens, objective=objective)
        
        masked_tokens = masked_batch["tokens"]
        was_masked = masked_batch["was_masked"]
        temporal_mask = masked_batch["temporal_mask"]
        
        # Check shapes
        assert masked_tokens.shape == tokens.shape
        assert was_masked.shape == tokens.shape
        
        # Check mask token is applied
        num_masked = (masked_tokens == 16).sum().item()
        num_was_masked = was_masked.sum().item()
        print(f"✓ {objective}: {num_masked} masked tokens, mask_ratio={masked_batch['mask_ratio']:.2f}")
        
        # Check temporal mask shape
        assert temporal_mask.shape == (T, T), f"Wrong temporal mask shape: {temporal_mask.shape}"
        
        # Verify mask properties
        if objective == "individual_denoise":
            # Identity mask: diagonal should be 0, rest -inf
            assert temporal_mask[0, 0].item() == 0.0, "Identity mask diagonal should be 0"
            assert temporal_mask[0, 1].item() == float("-inf"), "Identity mask off-diagonal should be -inf"
        else:
            # Causal mask: lower triangular should have 0 or valid values
            assert temporal_mask[0, 1].item() == float("-inf"), "Causal mask upper triangle should be -inf"
    
    print("Masking test PASSED!")
    return True


def test_loss():
    """Test the loss computation."""
    print("\n" + "=" * 60)
    print("Testing Loss Computation...")
    print("=" * 60)
    
    B, T, H, W = 2, 10, 32, 32
    V = 17  # 16 levels + mask token
    
    # Random logits and targets
    logits = torch.randn(B, T, H*W, V)
    targets = torch.randint(0, 16, (B, T, H, W))
    
    loss = compute_diffusion_loss(logits, targets, label_smoothing=0.1)
    
    print(f"✓ Loss value: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    
    # Test that loss decreases when logits match targets
    logits_perfect = torch.zeros(B, T, H*W, V)
    logits_perfect.scatter_(-1, targets.reshape(B, T, H*W, 1), 10.0)  # High logit for correct class
    
    loss_perfect = compute_diffusion_loss(logits_perfect, targets, label_smoothing=0.0)
    print(f"✓ Perfect prediction loss: {loss_perfect.item():.4f}")
    assert loss_perfect < loss, "Perfect prediction should have lower loss"
    
    print("Loss test PASSED!")
    return True


def test_training_step():
    """Test a full training step."""
    print("\n" + "=" * 60)
    print("Testing Training Step...")
    print("=" * 60)
    
    # Create small model and data
    model = SimpleVideoTransformer(
        vocab_size=16,
        mask_token_id=16,
        num_frames=5,
        height=16,
        width=16,
        embed_dim=32,
        num_layers=1,
        num_heads=2,
    )
    
    masker = DiscreteDiffusionMasker(
        vocab_size=16,
        mask_token_id=16,
        num_past_frames=2,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Create dummy batch
    B, T, H, W = 2, 5, 16, 16
    tokens = torch.randint(0, 16, (B, T, H, W))
    actions = torch.randn(B, T, 4)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    masked_batch = masker.prepare_batch(tokens)
    logits = model(masked_batch["tokens"], actions, masked_batch["temporal_mask"])
    loss = compute_diffusion_loss(logits, masked_batch["targets"])
    
    loss.backward()
    optimizer.step()
    
    print(f"✓ Training loss: {loss.item():.4f}")
    print(f"✓ Gradients computed: {any(p.grad is not None for p in model.parameters())}")
    
    print("Training step test PASSED!")
    return True


def main():
    print("\n" + "=" * 60)
    print("MOVING MNIST DISCRETE DIFFUSION - SETUP TEST")
    print("=" * 60)
    
    results = []
    
    results.append(("Dataset", test_dataset()))
    results.append(("Model", test_model()))
    results.append(("Masking", test_masking()))
    results.append(("Loss", test_loss()))
    results.append(("Training Step", test_training_step()))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name:20s}: {status}")
        all_passed = all_passed and passed
    
    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED! Ready for training.")
    else:
        print("SOME TESTS FAILED! Please fix before training.")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
