#!/usr/bin/env python3
"""Pre-tokenize KITTI dataset using frozen tokenizer.

Saves discrete token indices and poses for all frames to enable
fast world model training. Does NOT modify any tokenizer code.

Usage:
    python scripts/pretokenize_kitti.py \
        --tokenizer_config configs/tokenizer.yaml \
        --tokenizer_checkpoint outputs/tokenizer/checkpoint_latest.pt \
        --output_dir outputs/tokens \
        --sequences 00 01 02 03 04 05 06 07 08 09 10
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from copilot4d.utils.config import TokenizerConfig
from copilot4d.tokenizer.tokenizer_model import CoPilot4DTokenizer
from copilot4d.data.kitti_dataset import KITTITokenizerDataset, tokenizer_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-tokenize KITTI dataset")
    parser.add_argument("--tokenizer_config", type=str, required=True)
    parser.add_argument("--tokenizer_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/tokens")
    parser.add_argument("--sequences", nargs="+", default=None,
                        help="Sequences to process (default: all train+val+test)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_tokenizer(config_path: str, checkpoint_path: str, device: torch.device):
    """Load frozen tokenizer from config and checkpoint."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    cfg = TokenizerConfig(**config_dict)

    model = CoPilot4DTokenizer(cfg)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model, cfg


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from {args.tokenizer_checkpoint}")
    model, cfg = load_tokenizer(args.tokenizer_config, args.tokenizer_checkpoint, device)

    # Determine sequences
    if args.sequences:
        sequences = args.sequences
    else:
        sequences = cfg.train_sequences + cfg.val_sequences + cfg.test_sequences
    print(f"Processing sequences: {sequences}")

    # Load pykitti for pose data
    _pykitti_path = os.path.join(os.path.dirname(__file__), "..", "data", "kitti", "pykitti")
    if os.path.exists(_pykitti_path):
        sys.path.insert(0, _pykitti_path)
    from pykitti import odometry

    # Process each sequence
    for seq in sequences:
        print(f"\n=== Sequence {seq} ===")
        seq_dir = output_dir / seq
        seq_dir.mkdir(parents=True, exist_ok=True)

        # Load the pykitti dataset for poses
        kitti_data = odometry(cfg.kitti_root, seq)
        num_frames = len(kitti_data)

        # Extract poses: dict mapping frame_idx -> 4x4 numpy array
        poses = {}
        if hasattr(kitti_data, 'poses') and kitti_data.poses is not None:
            for idx, pose in enumerate(kitti_data.poses):
                poses[idx] = pose.astype(np.float64)
        else:
            print(f"  Warning: No poses for sequence {seq}, using identity")
            for idx in range(num_frames):
                poses[idx] = np.eye(4, dtype=np.float64)

        # Create single-sequence dataset for tokenization
        dataset = KITTITokenizerDataset(cfg, sequences=[seq])
        if len(dataset) == 0:
            print(f"  No data found, skipping")
            continue

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=tokenizer_collate_fn,
            pin_memory=True,
        )

        # Track frame index (sequential since shuffle=False)
        frame_idx = 0
        saved_count = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Seq {seq}"):
                # Move voxel data to device
                features = batch["features"].to(device)
                num_points = batch["num_points"].to(device)
                coords = batch["coords"].to(device)
                batch_size = batch["batch_size"]

                # Get discrete tokens from frozen tokenizer
                tokens = model.get_tokens(features, num_points, coords, batch_size)
                # tokens: (B, token_grid_h, token_grid_w)

                # Save each frame's tokens
                for i in range(batch_size):
                    token_file = seq_dir / f"{frame_idx:06d}.pt"
                    torch.save(tokens[i].cpu().to(torch.int16), token_file)
                    frame_idx += 1
                    saved_count += 1

        # Save poses
        with open(seq_dir / "poses.pkl", "wb") as f:
            pickle.dump(poses, f)

        print(f"  Saved {saved_count} token frames + poses to {seq_dir}")

    print(f"\nTokenization complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
