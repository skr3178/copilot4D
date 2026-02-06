#!/usr/bin/env python3
"""Pre-tokenize a small sample of KITTI for world model testing."""

import sys
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from copilot4d.utils.config import TokenizerConfig
from copilot4d.tokenizer.tokenizer_model import CoPilot4DTokenizer
from copilot4d.data.kitti_dataset import KITTITokenizerDataset


def main():
    # Config paths
    tokenizer_config_path = "configs/tokenizer_memory_efficient.yaml"
    checkpoint_path = "outputs/tokenizer_memory_efficient/checkpoint_step_22000.pt"
    output_dir = Path("outputs/tokens_sample")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    with open(tokenizer_config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    cfg = TokenizerConfig(**config_dict)
    
    print(f"Loading tokenizer from {checkpoint_path}")
    model = CoPilot4DTokenizer(cfg)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    
    print(f"Tokenizer loaded. Token grid size: {cfg.token_grid_size}")
    
    # Create dataset for sequence 00, first 100 frames
    print("Loading KITTI sequence 00...")
    dataset = KITTITokenizerDataset(
        cfg=cfg,
        sequences=["00"],
    )
    
    # Take only first 100 frames
    max_frames = 100
    print(f"Processing first {max_frames} frames...")
    
    # Output directory
    seq_dir = output_dir / "00"
    seq_dir.mkdir(parents=True, exist_ok=True)
    
    poses = {}
    batch_size = 4
    
    with torch.no_grad():
        for i in tqdm(range(0, min(max_frames, len(dataset)), batch_size)):
            # Get batch
            batch_items = []
            frame_indices = []
            
            for j in range(i, min(i + batch_size, max_frames, len(dataset))):
                item = dataset[j]
                batch_items.append(item)
                frame_indices.append(j)
                
                # Get sequence and frame index
                seq, frame_idx = dataset.samples[j]
                
                # Load pose from file (KITTI format: 12 numbers forming 3x4 matrix)
                pose_file = Path(cfg.kitti_root) / "dataset" / "poses" / f"{seq}.txt"
                with open(pose_file, 'r') as f:
                    pose_lines = f.readlines()
                pose_line = pose_lines[frame_idx].strip().split()
                pose = np.array([float(x) for x in pose_line]).reshape(3, 4)
                # Convert to 4x4 matrix
                pose_4x4 = np.eye(4)
                pose_4x4[:3, :] = pose
                poses[j] = pose_4x4
            
            # Collate manually for tokenization
            all_coords = []
            all_features = []
            all_num_points = []
            
            for b_idx, sample in enumerate(batch_items):
                coords = sample["coords"].copy()
                coords[:, 0] = b_idx  # set batch index
                all_coords.append(coords)
                all_features.append(sample["features"])
                all_num_points.append(sample["num_points"])
            
            coords = torch.from_numpy(np.concatenate(all_coords, axis=0))
            features = torch.from_numpy(np.concatenate(all_features, axis=0))
            num_points = torch.from_numpy(np.concatenate(all_num_points, axis=0))
            
            # Move to device
            features = features.to(device)
            num_points = num_points.to(device)
            coords = coords.to(device)
            
            # Get tokens
            tokens = model.get_tokens(
                features, num_points, coords, len(batch_items)
            )
            
            # Save tokens
            for idx, (frame_idx, token) in enumerate(zip(frame_indices, tokens)):
                token_file = seq_dir / f"{frame_idx:06d}.pt"
                torch.save(token.cpu(), token_file)
    
    # Save poses
    with open(seq_dir / "poses.pkl", "wb") as f:
        pickle.dump(poses, f)
    
    print(f"\nSaved {len(poses)} frames to {seq_dir}")
    
    # Verify by loading one sample
    sample_token = torch.load(seq_dir / "000000.pt", weights_only=True)
    print(f"Sample token shape: {sample_token.shape}")
    print(f"Token value range: [{sample_token.min()}, {sample_token.max()}]")


if __name__ == "__main__":
    main()
