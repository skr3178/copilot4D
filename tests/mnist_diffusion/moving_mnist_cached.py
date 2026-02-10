"""Moving MNIST with cached ego-centric actions for fast loading."""

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pickle


class MovingMNISTCached(Dataset):
    """Moving MNIST with pre-cached ego-centric actions.
    
    First run generates and caches actions to disk.
    Subsequent runs load from cache instantly.
    """
    
    def __init__(
        self,
        data_path: str = "mnist_test_seq.1.npy",
        seq_len: int = 20,
        num_sequences: int = None,
        start_idx: int = 0,
        num_token_levels: int = 16,
        frame_size: int = 32,
        use_ego_centric: bool = True,
        ego_digit_id: int = 0,
        cache_dir: str = "data/mnist_cache",
    ):
        self.data_path = data_path
        self.seq_len = min(seq_len, 20)
        self.num_token_levels = num_token_levels
        self.frame_size = frame_size
        self.use_ego_centric = use_ego_centric
        self.ego_digit_id = ego_digit_id
        self.action_dim = 2 if use_ego_centric else 4
        
        # Cache filename
        os.makedirs(cache_dir, exist_ok=True)
        cache_name = f"actions_ego{ego_digit_id}_fs{frame_size}_{start_idx}_{num_sequences}.pkl"
        self.cache_path = os.path.join(cache_dir, cache_name)
        
        # Load data
        print(f"Loading Moving MNIST from {data_path}...")
        full_data = np.load(data_path)  # (20, 10000, 64, 64)
        full_data = full_data.transpose(1, 0, 2, 3)  # (10000, 20, 64, 64)
        
        total_sequences = full_data.shape[0]
        if num_sequences is None:
            num_sequences = total_sequences - start_idx
        end_idx = min(start_idx + num_sequences, total_sequences)
        
        self.data = full_data[start_idx:end_idx, :self.seq_len]  # (N, T, 64, 64)
        self.num_sequences = self.data.shape[0]
        
        print(f"Loaded {self.num_sequences} sequences")
        
        # Resize if needed
        if frame_size != 64:
            print(f"Resizing to {frame_size}x{frame_size}...")
            from scipy.ndimage import zoom
            resized = np.zeros((self.num_sequences, self.seq_len, frame_size, frame_size), dtype=np.uint8)
            zoom_factor = frame_size / 64.0
            for i in range(self.num_sequences):
                for t in range(self.seq_len):
                    resized[i, t] = zoom(self.data[i, t], zoom_factor, order=1)
            self.data = resized
        
        # Quantize
        print(f"Quantizing to {num_token_levels} levels...")
        self.tokens = (self.data.astype(np.float32) / 255.0 * (num_token_levels - 1)).astype(np.int64)
        
        # Load or generate actions
        if use_ego_centric and os.path.exists(self.cache_path):
            print(f"Loading cached actions from {self.cache_path}")
            with open(self.cache_path, 'rb') as f:
                self.actions = pickle.load(f)
        elif use_ego_centric:
            print(f"Generating ego-centric actions (this may take a few minutes)...")
            self.actions = self._generate_actions_ego_centric_fast()
            print(f"Caching actions to {self.cache_path}")
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.actions, f)
        else:
            print("Generating standard actions...")
            self.actions = self._generate_actions_standard()
        
        print(f"Dataset ready! Action shape: {self.actions.shape}, range: [{self.actions.min():.3f}, {self.actions.max():.3f}]")
    
    def _generate_actions_ego_centric_fast(self):
        """Fast ego-centric action generation using center of mass tracking."""
        N, T, H, W = self.data.shape
        actions = np.zeros((N, T, 2), dtype=np.float32)
        
        for n in range(N):
            if n % 1000 == 0:
                print(f"  Processing sequence {n}/{N}...")
            
            prev_center = None
            
            for t in range(T):
                frame = self.data[n, t].astype(np.float32)
                
                # Threshold to get digit pixels
                binary = frame > 50
                
                # Find connected components using simple flood fill approximation
                # or use center of mass of thresholded image
                y_coords, x_coords = np.where(binary)
                
                if len(y_coords) == 0:
                    # No digit visible
                    if prev_center is not None:
                        actions[n, t] = [0.0, 0.0]
                    continue
                
                # Simple approach: if two digits, they usually don't overlap much
                # Use the center of mass, but try to track the digit closest to previous position
                if prev_center is not None and len(y_coords) > 100:  # Likely two digits
                    # Try to separate by finding two clusters
                    # Use y-coordinate histogram to find split
                    y_hist = np.sum(binary, axis=1)
                    x_hist = np.sum(binary, axis=0)
                    
                    # Find peaks in histograms
                    y_peaks = np.where(y_hist > np.max(y_hist) * 0.5)[0]
                    x_peaks = np.where(x_hist > np.max(x_hist) * 0.5)[0]
                    
                    if len(y_peaks) > 0 and len(x_peaks) > 0:
                        # Use the peak closest to previous center
                        y_center = (y_peaks[0] + y_peaks[-1]) / 2
                        x_center = (x_peaks[0] + x_peaks[-1]) / 2
                        
                        # Check if we can find two distinct centers
                        # Split by median
                        y_median = np.median(y_coords)
                        mask1 = y_coords < y_median
                        mask2 = ~mask1
                        
                        if mask1.sum() > 10 and mask2.sum() > 10:
                            center1 = np.array([y_coords[mask1].mean(), x_coords[mask1].mean()])
                            center2 = np.array([y_coords[mask2].mean(), x_coords[mask2].mean()])
                            
                            # Pick the one closer to previous center
                            dist1 = np.sum((center1 - prev_center) ** 2)
                            dist2 = np.sum((center2 - prev_center) ** 2)
                            
                            if self.ego_digit_id == 0:
                                curr_center = center1 if dist1 < dist2 else center2
                            else:
                                curr_center = center2 if dist1 < dist2 else center1
                        else:
                            curr_center = np.array([y_coords.mean(), x_coords.mean()])
                    else:
                        curr_center = np.array([y_coords.mean(), x_coords.mean()])
                else:
                    curr_center = np.array([y_coords.mean(), x_coords.mean()])
                
                if prev_center is not None:
                    dy = curr_center[0] - prev_center[0]
                    dx = curr_center[1] - prev_center[1]
                    actions[n, t] = [dx / (W / 2), dy / (H / 2)]
                
                prev_center = curr_center
            
            # Fill first frame with second frame's motion
            if T > 1:
                actions[n, 0] = actions[n, 1]
        
        return actions
    
    def _generate_actions_standard(self):
        """Generate standard one-hot actions."""
        N, T, H, W = self.data.shape
        actions = np.zeros((N, T, 4), dtype=np.float32)
        
        for n in range(N):
            for t in range(1, T):
                prev_frame = self.data[n, t-1].astype(np.float32)
                curr_frame = self.data[n, t].astype(np.float32)
                
                prev_thresh = prev_frame > 50
                curr_thresh = curr_frame > 50
                
                # Center of mass
                def center_of_mass(mask):
                    y_coords, x_coords = np.where(mask)
                    if len(y_coords) == 0:
                        return None
                    return np.array([y_coords.mean(), x_coords.mean()])
                
                prev_com = center_of_mass(prev_thresh)
                curr_com = center_of_mass(curr_thresh)
                
                if prev_com is not None and curr_com is not None:
                    dy, dx = curr_com - prev_com
                    
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
        frames = self.data[idx].astype(np.float32) / 255.0
        tokens = self.tokens[idx]
        actions = self.actions[idx]
        
        return {
            "frames": torch.from_numpy(frames).float(),
            "tokens": torch.from_numpy(tokens).long(),
            "actions": torch.from_numpy(actions).float(),
        }


def create_cached_dataloaders(
    data_path="mnist_test_seq.1.npy",
    seq_len=20,
    batch_size=4,
    num_train=8000,  # Full dataset: 8000 train
    num_val=2000,    # Full dataset: 2000 val
    num_token_levels=16,
    frame_size=32,
    use_ego_centric=True,
    ego_digit_id=0,
    cache_dir="data/mnist_cache_full",  # Full cache by default
):
    """Create dataloaders with cached actions."""
    
    train_dataset = MovingMNISTCached(
        data_path=data_path,
        seq_len=seq_len,
        num_sequences=num_train,
        start_idx=0,
        num_token_levels=num_token_levels,
        frame_size=frame_size,
        use_ego_centric=use_ego_centric,
        ego_digit_id=ego_digit_id,
        cache_dir=cache_dir,
    )
    
    val_dataset = MovingMNISTCached(
        data_path=data_path,
        seq_len=seq_len,
        num_sequences=num_val,
        start_idx=num_train,
        num_token_levels=num_token_levels,
        frame_size=frame_size,
        use_ego_centric=use_ego_centric,
        ego_digit_id=ego_digit_id,
        cache_dir=cache_dir,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 to avoid fork issues with cached data
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test
    print("Testing cached dataset...")
    train_loader, val_loader = create_cached_dataloaders(
        num_train=100,
        num_val=50,
        batch_size=4,
    )
    
    batch = next(iter(train_loader))
    print(f"Batch shapes: {batch['frames'].shape}, {batch['tokens'].shape}, {batch['actions'].shape}")
    print(f"Action range: [{batch['actions'].min():.3f}, {batch['actions'].max():.3f}]")
    print("Test passed!")
