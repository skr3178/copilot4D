#!/usr/bin/env python3
"""End-to-end pipeline test: Tokenizer -> Pretokenize -> World Model.

Phase 1: Load frozen tokenizer, tokenize 20 KITTI frames, save to disk.
Phase 2: Load pretokenized data, train world model for 10 steps.
Phase 3: Run inference smoke test (predict 1 future frame).
"""

import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from copilot4d.utils.config import TokenizerConfig, WorldModelConfig

# ============================================================
# Phase 1: Pretokenize sample KITTI frames
# ============================================================

def phase1_pretokenize(num_frames: int = 20):
    """Load frozen tokenizer, process frames, save tokens + poses."""
    print("\n" + "=" * 60)
    print("PHASE 1: Pretokenize KITTI frames")
    print("=" * 60)

    from copilot4d.tokenizer.tokenizer_model import CoPilot4DTokenizer
    from copilot4d.data.kitti_dataset import KITTITokenizerDataset, tokenizer_collate_fn

    tokenizer_config = "configs/tokenizer_memory_efficient.yaml"
    tokenizer_ckpt = "outputs/tokenizer_memory_efficient/checkpoint_step_40000.pt"
    output_dir = Path("outputs/tokens_e2e_test")
    seq = "00"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer config
    with open(tokenizer_config, "r") as f:
        config_dict = yaml.safe_load(f)
    cfg = TokenizerConfig(**config_dict)
    print(f"Tokenizer config: grid={cfg.voxel_grid_xy}, codebook={cfg.vq_codebook_size}")

    # Build and load tokenizer
    print(f"Loading tokenizer from {tokenizer_ckpt}...")
    model = CoPilot4DTokenizer(cfg)
    checkpoint = torch.load(tokenizer_ckpt, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Tokenizer params: {num_params / 1e6:.2f}M")

    # Create dataset for sequence 00
    dataset = KITTITokenizerDataset(cfg, sequences=[seq])
    print(f"Dataset: {len(dataset)} frames in sequence {seq}")

    # Only process num_frames
    actual_frames = min(num_frames, len(dataset))
    subset = torch.utils.data.Subset(dataset, list(range(actual_frames)))
    dataloader = torch.utils.data.DataLoader(
        subset,
        batch_size=1,  # Process one at a time for simplicity
        shuffle=False,
        num_workers=2,
        collate_fn=tokenizer_collate_fn,
        pin_memory=True,
    )

    # Output directory
    seq_dir = output_dir / seq
    seq_dir.mkdir(parents=True, exist_ok=True)

    # Get poses via pykitti
    _pykitti_path = os.path.join(os.path.dirname(__file__), "..", "data", "kitti", "pykitti")
    if os.path.exists(_pykitti_path):
        sys.path.insert(0, _pykitti_path)
    from pykitti import odometry

    # pykitti expects basedir/poses/{seq}.txt - find the correct base
    kitti_base = cfg.kitti_root
    if not os.path.exists(os.path.join(kitti_base, "poses")):
        # Try dataset subdirectory
        candidate = os.path.join(kitti_base, "dataset")
        if os.path.exists(os.path.join(candidate, "poses")):
            kitti_base = candidate
    kitti_data = odometry(kitti_base, seq)
    poses = {}
    if hasattr(kitti_data, 'poses') and kitti_data.poses is not None:
        for idx, pose in enumerate(kitti_data.poses):
            if idx < actual_frames:
                poses[idx] = pose.astype(np.float64)
    else:
        for idx in range(actual_frames):
            poses[idx] = np.eye(4, dtype=np.float64)

    # Tokenize frames
    print(f"\nTokenizing {actual_frames} frames...")
    t0 = time.time()
    frame_idx = 0
    all_tokens = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            num_points = batch["num_points"].to(device)
            coords = batch["coords"].to(device)
            batch_size = batch["batch_size"]

            tokens = model.get_tokens(features, num_points, coords, batch_size)

            for i in range(batch_size):
                token_file = seq_dir / f"{frame_idx:06d}.pt"
                t = tokens[i].cpu().to(torch.int16)
                torch.save(t, token_file)
                all_tokens.append(t)
                frame_idx += 1

            if frame_idx % 5 == 0:
                print(f"  Processed {frame_idx}/{actual_frames} frames")

    # Save poses
    with open(seq_dir / "poses.pkl", "wb") as f:
        pickle.dump(poses, f)

    elapsed = time.time() - t0
    print(f"\nPhase 1 complete: {frame_idx} frames in {elapsed:.1f}s ({elapsed/frame_idx:.2f}s/frame)")

    # Verify tokens
    sample = all_tokens[0]
    print(f"\nToken verification:")
    print(f"  Shape: {sample.shape}")
    print(f"  Dtype: {sample.dtype}")
    print(f"  Range: [{sample.min()}, {sample.max()}]")
    print(f"  Unique codes: {sample.unique().numel()}")

    token_h, token_w = sample.shape
    assert sample.min() >= 0, "Token values should be >= 0"
    assert sample.max() < 1024, f"Token values should be < 1024, got {sample.max()}"
    print("  [PASS] All token checks passed")

    return str(output_dir), token_h, token_w, actual_frames


# ============================================================
# Phase 2: World Model Training
# ============================================================

def phase2_world_model_training(token_dir: str, token_h: int, token_w: int, num_steps: int = 10):
    """Train world model for a few steps on pretokenized data."""
    print("\n" + "=" * 60)
    print("PHASE 2: World Model Training")
    print("=" * 60)

    from copilot4d.world_model.world_model import CoPilot4DWorldModel
    from copilot4d.world_model.masking import DiscreteDiffusionMasker, compute_diffusion_loss
    from copilot4d.data.kitti_sequence_dataset import KITTISequenceDataset, sequence_collate_fn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load world model config matching token grid size
    wm_config = "configs/world_model_64x64.yaml"
    with open(wm_config, "r") as f:
        config_dict = yaml.safe_load(f)

    # Override token_dir and grid size to match our pretokenized data
    config_dict["token_dir"] = token_dir
    config_dict["token_grid_h"] = token_h
    config_dict["token_grid_w"] = token_w
    config_dict["train_sequences"] = ["00"]
    config_dict["val_sequences"] = ["00"]
    config_dict["batch_size"] = 1  # Small batch for GPU memory
    config_dict["amp"] = True

    cfg = WorldModelConfig(**config_dict)
    print(f"World model config: grid={cfg.token_grid_h}x{cfg.token_grid_w}, "
          f"frames={cfg.num_frames}, dims={cfg.level_dims}")

    # Create dataset
    dataset = KITTISequenceDataset(cfg, split="train")
    print(f"Dataset: {len(dataset)} samples (sequences of {cfg.num_frames} frames)")
    if len(dataset) == 0:
        print("[FAIL] No samples found! Need at least num_frames consecutive frames.")
        return None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(cfg.batch_size, len(dataset)),
        shuffle=True,
        num_workers=0,  # Simpler for testing
        collate_fn=sequence_collate_fn,
    )

    # Build model
    print("\nBuilding world model...")
    model = CoPilot4DWorldModel(cfg)
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"World model params: {num_params / 1e6:.2f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    # Masker
    masker = DiscreteDiffusionMasker(cfg)

    # Training loop
    print(f"\nTraining for {num_steps} steps...")
    print("-" * 70)
    model.train()
    losses = []
    step = 0
    data_iter = iter(dataloader)

    for step in range(num_steps):
        # Get batch (cycle through data)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        tokens = batch["tokens"].to(device)
        actions = batch["actions"].to(device)
        B, T, H, W = tokens.shape

        # Apply masking
        masked_batch = masker.prepare_batch(tokens)
        masked_tokens = masked_batch["tokens"]
        targets = masked_batch["targets"]
        temporal_mask = masked_batch["temporal_mask"]
        objective = masked_batch["objective"]

        # Forward + loss
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            logits = model(masked_tokens, actions, temporal_mask)
            loss = compute_diffusion_loss(logits, targets, cfg.label_smoothing)

        # Backward
        optimizer.zero_grad()
        if cfg.amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        # Metrics
        with torch.no_grad():
            preds = logits.argmax(dim=-1).reshape(B, T, H, W)
            acc = (preds == targets).float().mean().item()

        losses.append(loss.item())

        print(f"  Step {step:3d} | loss={loss.item():.4f} | acc={acc:.4f} | "
              f"grad_norm={grad_norm:.2f} | obj={objective[:8]:8s} | "
              f"logits={tuple(logits.shape)} | batch=({B},{T},{H},{W})")

    # Summary
    print("-" * 70)
    avg_loss_first = sum(losses[:3]) / 3
    avg_loss_last = sum(losses[-3:]) / 3
    print(f"\nPhase 2 complete:")
    print(f"  First 3 steps avg loss: {avg_loss_first:.4f}")
    print(f"  Last  3 steps avg loss: {avg_loss_last:.4f}")
    print(f"  Loss {'decreased' if avg_loss_last < avg_loss_first else 'did not decrease'}")

    # Check for NaN
    if any(np.isnan(l) for l in losses):
        print("  [FAIL] NaN loss detected!")
        return None
    else:
        print("  [PASS] No NaN losses")

    return model, cfg


# ============================================================
# Phase 3: Inference Smoke Test
# ============================================================

def phase3_inference(model, cfg, token_dir: str):
    """Run a quick inference test."""
    print("\n" + "=" * 60)
    print("PHASE 3: Inference Smoke Test")
    print("=" * 60)

    from copilot4d.world_model.inference import WorldModelSampler
    from copilot4d.data.kitti_sequence_dataset import KITTISequenceDataset, sequence_collate_fn

    device = next(model.parameters()).device
    model.eval()

    # Load a sample from the dataset
    dataset = KITTISequenceDataset(cfg, split="train")
    sample = dataset[0]
    tokens = sample["tokens"].unsqueeze(0).to(device)  # (1, T, H, W)
    actions = sample["actions"].unsqueeze(0).to(device)  # (1, T, 16)

    B, T, H, W = tokens.shape
    num_past = cfg.num_past_frames

    # Split into past and future
    past_tokens = tokens[:, :num_past]
    past_actions = actions[:, :num_past]
    future_action = actions[:, num_past:num_past + 1]

    print(f"Input: past_tokens={tuple(past_tokens.shape)}, "
          f"past_actions={tuple(past_actions.shape)}, "
          f"future_action={tuple(future_action.shape)}")

    # Create sampler with reduced steps for speed
    test_cfg = WorldModelConfig(**{
        **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
        "num_sampling_steps": 4,  # Fewer steps for testing
    })
    sampler = WorldModelSampler(model, test_cfg)

    print("Running iterative decoding (4 steps)...")
    t0 = time.time()
    predicted = sampler.predict_next_frame(past_tokens, past_actions, future_action)
    elapsed = time.time() - t0

    print(f"\nInference results:")
    print(f"  Output shape: {tuple(predicted.shape)}")
    print(f"  Value range: [{predicted.min()}, {predicted.max()}]")
    print(f"  Unique tokens: {predicted.unique().numel()}")
    print(f"  Time: {elapsed:.2f}s")

    # Checks
    assert predicted.shape == (B, H, W), f"Expected ({B}, {H}, {W}), got {tuple(predicted.shape)}"
    assert predicted.min() >= 0, f"Tokens should be >= 0"
    assert predicted.max() < cfg.codebook_size, f"Tokens should be < {cfg.codebook_size}"

    # Compare with ground truth
    gt_future = tokens[:, num_past, :, :]  # (B, H, W)
    match_rate = (predicted == gt_future).float().mean().item()
    print(f"  Match with GT future: {match_rate:.4f} (expected low for untrained model)")
    print("  [PASS] Inference smoke test passed")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("CoPilot4D End-to-End Pipeline Test")
    print("=" * 60)

    t_start = time.time()

    # Phase 1
    token_dir, token_h, token_w, num_frames = phase1_pretokenize(num_frames=20)

    # Phase 2
    result = phase2_world_model_training(token_dir, token_h, token_w, num_steps=10)
    if result is None:
        print("\n[FAIL] Phase 2 failed, skipping Phase 3")
        return

    model, cfg = result

    # Phase 3
    phase3_inference(model, cfg, token_dir)

    total = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"All phases complete in {total:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
