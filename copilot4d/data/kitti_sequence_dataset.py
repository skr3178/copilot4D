"""KITTI sequence dataset for world model training.

Loads sequences of consecutive tokenized frames with relative pose actions.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from copilot4d.utils.config import WorldModelConfig


class KITTISequenceDataset(Dataset):
    """Dataset for training the world model on KITTI sequences.
    
    Loads pre-computed tokens and computes relative SE(3) pose actions.
    Each sample is a sequence of T consecutive frames.
    """

    def __init__(
        self,
        cfg: WorldModelConfig,
        split: str = "train",
        sequences: Optional[List[str]] = None,
    ):
        """
        Args:
            cfg: WorldModelConfig with data paths and sequence settings
            split: "train", "val", or "test"
            sequences: override sequences to use (if None, use cfg)
        """
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.num_frames = cfg.num_frames
        
        # Determine sequences to use
        if sequences is not None:
            self.sequences = sequences
        elif split == "train":
            self.sequences = cfg.train_sequences
        elif split == "val":
            self.sequences = cfg.val_sequences
        else:
            self.sequences = cfg.test_sequences
        
        # Load metadata
        self.token_dir = Path(cfg.token_dir)
        self.samples = self._build_sample_list()
        
        print(f"KITTISequenceDataset[{split}]: {len(self.samples)} samples from {len(self.sequences)} sequences")

    def _build_sample_list(self) -> List[Tuple[str, int]]:
        """Build list of valid (sequence, start_frame) samples.
        
        Returns:
            List of (sequence_id, start_frame) tuples
        """
        samples = []
        
        for seq in self.sequences:
            seq_path = self.token_dir / seq
            if not seq_path.exists():
                print(f"Warning: Token directory not found: {seq_path}")
                continue
            
            # Get list of token files
            token_files = sorted(seq_path.glob("*.pt"))
            if len(token_files) == 0:
                print(f"Warning: No token files found in {seq_path}")
                continue
            
            # Load pose file
            pose_file = seq_path / "poses.pkl"
            if not pose_file.exists():
                print(f"Warning: Pose file not found: {pose_file}")
                continue
            
            with open(pose_file, "rb") as f:
                poses = pickle.load(f)
            
            num_frames_in_seq = len(token_files)
            
            # Create samples: each sample is num_frames consecutive frames
            for start_idx in range(num_frames_in_seq - self.num_frames + 1):
                # Check if all frames have valid poses
                valid = True
                for i in range(self.num_frames):
                    frame_idx = start_idx + i
                    if frame_idx not in poses:
                        valid = False
                        break
                
                if valid:
                    samples.append((seq, start_idx))
        
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence sample.
        
        Returns:
            Dict with:
                - tokens: (T, H, W) long tensor of token indices
                - actions: (T, 16) float tensor of flattened relative SE(3)
                - seq_id: str sequence identifier
                - start_frame: int start frame index
        """
        seq, start_idx = self.samples[idx]
        seq_path = self.token_dir / seq
        
        # Load tokens for all frames in sequence
        tokens_list = []
        for i in range(self.num_frames):
            frame_idx = start_idx + i
            token_file = seq_path / f"{frame_idx:06d}.pt"
            tokens = torch.load(token_file, map_location="cpu", weights_only=True)
            tokens_list.append(tokens)
        
        tokens = torch.stack(tokens_list, dim=0)  # (T, H, W)
        
        # Load poses
        with open(seq_path / "poses.pkl", "rb") as f:
            poses = pickle.load(f)
        
        # Compute relative actions (SE(3) transforms between consecutive frames)
        actions = []
        for i in range(self.num_frames):
            frame_idx = start_idx + i
            
            if i == 0:
                # First frame: use identity or next frame's pose
                if self.num_frames > 1:
                    next_idx = start_idx + 1
                    T_curr = poses[frame_idx]
                    T_next = poses[next_idx]
                    T_rel = np.linalg.inv(T_curr) @ T_next
                else:
                    T_rel = np.eye(4)
            else:
                # Relative pose from previous frame
                prev_idx = start_idx + i - 1
                T_prev = poses[prev_idx]
                T_curr = poses[frame_idx]
                T_rel = np.linalg.inv(T_prev) @ T_curr
            
            # Flatten to 16-dim
            actions.append(T_rel.flatten())
        
        actions = torch.from_numpy(np.stack(actions, axis=0)).float()  # (T, 16)
        
        return {
            "tokens": tokens,
            "actions": actions,
            "seq_id": seq,
            "start_frame": start_idx,
        }


def sequence_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader.
    
    Stacks samples into batch tensors.
    """
    tokens = torch.stack([b["tokens"] for b in batch], dim=0)  # (B, T, H, W)
    actions = torch.stack([b["actions"] for b in batch], dim=0)  # (B, T, 16)
    
    return {
        "tokens": tokens,
        "actions": actions,
        "seq_ids": [b["seq_id"] for b in batch],
        "start_frames": [b["start_frame"] for b in batch],
    }
