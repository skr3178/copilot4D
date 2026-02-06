#!/usr/bin/env python3
"""Test world model pipeline on pretokenized data."""

import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from copilot4d.utils.config import WorldModelConfig
from copilot4d.world_model.world_model import CoPilot4DWorldModel
from copilot4d.world_model.masking import DiscreteDiffusionMasker, compute_diffusion_loss
from copilot4d.world_model.temporal_block import make_causal_mask
from copilot4d.data.kitti_sequence_dataset import KITTISequenceDataset, sequence_collate_fn
from torch.utils.data import DataLoader


def main():
    # Load config
    config_path = "configs/world_model_64x64.yaml"
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    cfg = WorldModelConfig(**config_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Config: token_grid={cfg.token_grid_h}x{cfg.token_grid_w}, vocab_size={cfg.vocab_size}")
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = KITTISequenceDataset(cfg, split="train")
    print(f"Dataset size: {len(dataset)} samples")
    
    # Check first sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Tokens shape: {sample['tokens'].shape}")
    print(f"Actions shape: {sample['actions'].shape}")
    print(f"Token value range: [{sample['tokens'].min()}, {sample['tokens'].max()}]")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=sequence_collate_fn,
    )
    
    # Build model
    print("\nBuilding world model...")
    model = CoPilot4DWorldModel(cfg)
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    
    # Create masker
    masker = DiscreteDiffusionMasker(cfg)
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch = next(iter(dataloader))
    tokens = batch["tokens"].to(device)
    actions = batch["actions"].to(device)
    
    print(f"Batch tokens shape: {tokens.shape}")
    print(f"Batch actions shape: {actions.shape}")
    
    # Apply masking
    masked_batch = masker.prepare_batch(tokens, objective="future_prediction")
    masked_tokens = masked_batch["tokens"]
    targets = masked_batch["targets"]
    temporal_mask = masked_batch["temporal_mask"].to(device)
    
    print(f"Masked tokens shape: {masked_tokens.shape}")
    print(f"Number of masked tokens: {masked_batch['was_masked'].sum().item()}")
    print(f"Objective: {masked_batch['objective']}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(masked_tokens, actions, temporal_mask)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Expected: ({cfg.batch_size}, {cfg.num_frames}, {cfg.num_tokens_per_frame}, {cfg.vocab_size})")
    
    # Compute loss
    loss = compute_diffusion_loss(logits, targets, cfg.label_smoothing)
    print(f"Loss: {loss.item():.4f}")
    
    # Check predictions
    with torch.no_grad():
        preds = logits.argmax(dim=-1).reshape_as(targets)
        acc = (preds == targets).float().mean()
        print(f"Accuracy: {acc.item():.4f}")
    
    # Test backward pass
    print("\nTesting backward pass...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    
    masked_batch = masker.prepare_batch(tokens, objective="joint_denoise")
    masked_tokens = masked_batch["tokens"]
    targets = masked_batch["targets"]
    temporal_mask = masked_batch["temporal_mask"].to(device)
    
    logits = model(masked_tokens, actions, temporal_mask)
    loss = compute_diffusion_loss(logits, targets, cfg.label_smoothing)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Training loss: {loss.item():.4f}")
    print("Backward pass successful!")
    
    # Test a few training steps
    print("\nRunning 5 training steps...")
    model.train()
    for i in range(5):
        batch = next(iter(dataloader))
        tokens = batch["tokens"].to(device)
        actions = batch["actions"].to(device)
        
        masked_batch = masker.prepare_batch(tokens)
        masked_tokens = masked_batch["tokens"]
        targets = masked_batch["targets"]
        temporal_mask = masked_batch["temporal_mask"].to(device)
        
        logits = model(masked_tokens, actions, temporal_mask)
        loss = compute_diffusion_loss(logits, targets, cfg.label_smoothing)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        
        with torch.no_grad():
            preds = logits.argmax(dim=-1).reshape_as(targets)
            acc = (preds == targets).float().mean()
        
        print(f"  Step {i+1}: loss={loss.item():.4f}, acc={acc.item():.4f}, obj={masked_batch['objective'][:3]}")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    main()
