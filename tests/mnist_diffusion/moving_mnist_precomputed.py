"""Moving MNIST dataset using precomputed .npy file.

Uses the downloaded mnist_test_seq.npy file and creates train/val splits.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os


class MovingMNISTPrecomputed(Dataset):
    """Moving MNIST dataset from precomputed .npy file.
    
    The data file has shape (20, 10000, 64, 64):
    - 20 timesteps per sequence
    - 10000 sequences
    - 64x64 frames
    
    Args:
        data_path: Path to mnist_test_seq.npy
        seq_len: Number of frames to use per sequence (default: 20, max: 20)
        num_sequences: Number of sequences to use (default: all)
        start_idx: Starting sequence index
        action_dim: Dimension of action vectors
        num_token_levels: Number of quantization levels for tokens
    """
    
    def __init__(
        self,
        data_path: str = "data/mnist_test_seq.npy",
        seq_len: int = 20,
        num_sequences: int = None,
        start_idx: int = 0,
        action_dim: int = 4,
        num_token_levels: int = 16,
        preload: bool = True,
        frame_size: int = 64,
    ):
        self.data_path = data_path
        self.seq_len = min(seq_len, 20)
        self.action_dim = action_dim
        self.num_token_levels = num_token_levels
        self.preload = preload
        self.frame_size = frame_size
        
        # Determine if we need to resize
        self.needs_resize = (frame_size != 64)
        
        # Load data (transposed to (N, T, H, W))
        print(f"Loading Moving MNIST from {data_path}...")
        
        if preload:
            full_data = np.load(data_path)  # (20, 10000, 64, 64)
            full_data = full_data.transpose(1, 0, 2, 3)  # (10000, 20, 64, 64)
            
            total_sequences = full_data.shape[0]
            
            # Select subset
            if num_sequences is None:
                num_sequences = total_sequences - start_idx
            end_idx = min(start_idx + num_sequences, total_sequences)
            
            self.data = full_data[start_idx:end_idx, :self.seq_len]  # (N, T, 64, 64)
            self.num_sequences = self.data.shape[0]
            self._full_data = None  # Don't need this anymore
        else:
            # Memory-mapped loading
            self._full_data = np.load(data_path, mmap_mode='r')  # (20, 10000, 64, 64)
            total_sequences = self._full_data.shape[1]
            
            if num_sequences is None:
                num_sequences = total_sequences - start_idx
            self.num_sequences = min(num_sequences, total_sequences - start_idx)
            self._start_idx = start_idx
        
        print(f"Loaded {self.num_sequences} sequences of length {self.seq_len}")
        print(f"Frame shape: ({frame_size}, {frame_size}) ({frame_size * frame_size} tokens)")
        print(f"Token levels: {num_token_levels}")
        
        if self.preload:
            # Resize data if needed
            if self.needs_resize:
                print(f"Resizing data from 64x64 to {frame_size}x{frame_size}...")
                resized_data = np.zeros((self.num_sequences, self.seq_len, frame_size, frame_size), dtype=np.uint8)
                for i in range(self.num_sequences):
                    resized_data[i] = self._resize_frames(self.data[i])
                self.data = resized_data
            
            # Pre-quantize all data
            self.tokens = self._quantize(self.data, num_token_levels)
            # Generate synthetic actions based on frame differences
            self.actions = self._generate_actions()
        else:
            self.tokens = None
            self.actions = None
        
    def _resize_frames(self, frames):
        """Resize frames to target size using simple downsampling.
        
        Args:
            frames: (T, 64, 64) or (T, H, W) array
            
        Returns:
            (T, frame_size, frame_size) array
        """
        if not self.needs_resize:
            return frames
        
        T = frames.shape[0]
        # Simple pooling-based downsampling
        from scipy.ndimage import zoom
        zoom_factor = self.frame_size / 64.0
        resized = np.zeros((T, self.frame_size, self.frame_size), dtype=frames.dtype)
        for t in range(T):
            resized[t] = zoom(frames[t], zoom_factor, order=1)
        return resized
    
    def _quantize(self, frames, num_levels):
        """Quantize uint8 frames to discrete tokens.
        
        Args:
            frames: (N, T, H, W) uint8 array [0, 255]
            
        Returns:
            (N, T, H, W) int64 array [0, num_levels-1]
        """
        # Normalize to [0, 1] then quantize
        normalized = frames.astype(np.float32) / 255.0
        tokens = (normalized * (num_levels - 1)).astype(np.int64)
        return tokens
    
    def _generate_actions(self):
        """Generate synthetic action vectors based on frame motion.
        
        Actions represent velocity changes detected between frames:
        - One-hot encoding for [up, down, left, right]
        
        Returns:
            (N, T, 4) action array
        """
        N, T, H, W = self.data.shape
        actions = np.zeros((N, T, self.action_dim), dtype=np.float32)
        
        for n in range(N):
            for t in range(1, T):
                # Compute optical flow approximation (center of mass movement)
                prev_frame = self.data[n, t-1].astype(np.float32)
                curr_frame = self.data[n, t].astype(np.float32)
                
                # Threshold to get digit region
                prev_thresh = prev_frame > 50
                curr_thresh = curr_frame > 50
                
                # Compute center of mass
                def center_of_mass(mask):
                    y_coords, x_coords = np.where(mask)
                    if len(y_coords) == 0:
                        return None
                    return np.array([y_coords.mean(), x_coords.mean()])
                
                prev_com = center_of_mass(prev_thresh)
                curr_com = center_of_mass(curr_thresh)
                
                if prev_com is not None and curr_com is not None:
                    dy, dx = curr_com - prev_com
                    
                    # Map to action
                    if abs(dy) > abs(dx):
                        if dy < 0:
                            actions[n, t] = [1, 0, 0, 0]  # up
                        else:
                            actions[n, t] = [0, 1, 0, 0]  # down
                    else:
                        if dx < 0:
                            actions[n, t] = [0, 0, 1, 0]  # left
                        else:
                            actions[n, t] = [0, 0, 0, 1]  # right
        
        return actions
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        """Get a sequence.
        
        Returns:
            dict with:
                - frames: (T, H, W) continuous frames normalized to [0, 1]
                - tokens: (T, H, W) quantized tokens [0, num_levels-1]
                - actions: (T, 4) action vectors
        """
        if self.preload:
            # Data already resized in __init__ if needed
            frames = self.data[idx].astype(np.float32) / 255.0
            tokens = self.tokens[idx]
            actions = self.actions[idx]
        else:
            # Load from memory-mapped array
            # Data is (20, 10000, 64, 64), we need (seq_len, 64, 64)
            actual_idx = self._start_idx + idx
            frames = self._full_data[:self.seq_len, actual_idx].transpose(1, 2, 0)  # (64, 64, seq_len)
            frames = frames.transpose(2, 0, 1)  # (seq_len, 64, 64)
            
            if self.needs_resize:
                frames = self._resize_frames(frames)
            
            frames = frames.astype(np.float32) / 255.0
            
            # Quantize on-the-fly
            tokens = (frames * (self.num_token_levels - 1)).astype(np.int64)
            
            # Generate actions on-the-fly
            actions = np.zeros((self.seq_len, self.action_dim), dtype=np.float32)
        
        return {
            "frames": torch.from_numpy(frames).float(),
            "tokens": torch.from_numpy(tokens).long(),
            "actions": torch.from_numpy(actions).float(),
        }


def create_mnist_dataloaders(
    data_path="mnist_test_seq.1.npy",
    seq_len=20,
    batch_size=8,
    num_train=8000,
    num_val=1000,
    num_workers=4,
    num_token_levels=16,
    frame_size=64,
):
    """Create train and validation dataloaders.
    
    Args:
        data_path: Path to mnist_test_seq.npy
        seq_len: Number of frames per sequence
        batch_size: Batch size
        num_train: Number of training sequences
        num_val: Number of validation sequences
        num_workers: DataLoader workers
        num_token_levels: Quantization levels for tokens
        
    Returns:
        train_loader, val_loader
    """
    # Training set
    train_dataset = MovingMNISTPrecomputed(
        data_path=data_path,
        seq_len=seq_len,
        num_sequences=num_train,
        start_idx=0,
        num_token_levels=num_token_levels,
        frame_size=frame_size,
    )
    
    # Validation set
    val_dataset = MovingMNISTPrecomputed(
        data_path=data_path,
        seq_len=seq_len,
        num_sequences=num_val,
        start_idx=num_train,
        num_token_levels=num_token_levels,
        frame_size=frame_size,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    print("Testing Moving MNIST Precomputed dataset...")
    
    # Check for the file in common locations
    possible_paths = [
        "mnist_test_seq.1.npy",
        "data/mnist_test_seq.1.npy",
        "mnist_test_seq.npy",
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        print("Error: Could not find mnist_test_seq.npy file")
        exit(1)
    
    print(f"Using data file: {data_path}")
    
    # Test dataset
    dataset = MovingMNISTPrecomputed(
        data_path=data_path,
        seq_len=20,
        num_sequences=100,
    )
    
    sample = dataset[0]
    print(f"\nSample shapes:")
    print(f"  Frames: {sample['frames'].shape}")
    print(f"  Tokens: {sample['tokens'].shape}")
    print(f"  Actions: {sample['actions'].shape}")
    
    print(f"\nToken statistics:")
    print(f"  Range: [{sample['tokens'].min()}, {sample['tokens'].max()}]")
    print(f"  Unique: {torch.unique(sample['tokens']).numel()}")
    
    print(f"\nAction distribution in sample:")
    print(f"  Up: {sample['actions'][:, 0].sum()}")
    print(f"  Down: {sample['actions'][:, 1].sum()}")
    print(f"  Left: {sample['actions'][:, 2].sum()}")
    print(f"  Right: {sample['actions'][:, 3].sum()}")
    
    # Test dataloaders
    print("\nTesting dataloaders...")
    train_loader, val_loader = create_mnist_dataloaders(
        data_path=data_path,
        num_train=100,
        num_val=50,
        batch_size=4,
    )
    
    batch = next(iter(train_loader))
    print(f"Train batch shapes:")
    print(f"  Frames: {batch['frames'].shape}")
    print(f"  Tokens: {batch['tokens'].shape}")
    print(f"  Actions: {batch['actions'].shape}")
    
    print("\nMoving MNIST Precomputed test passed!")
