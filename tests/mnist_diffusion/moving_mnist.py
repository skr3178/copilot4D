"""Moving MNIST dataset generator for testing discrete diffusion.

Generates synthetic videos of bouncing digits with physics-based motion.
This is faster than downloading and allows easy action conditioning.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class MovingMNIST(Dataset):
    """Moving MNIST dataset with bouncing digits.
    
    Each sequence contains T frames of 64x64 grayscale video with 1-2 digits
    bouncing with velocity and optional acceleration (actions).
    
    Args:
        num_sequences: Number of video sequences to generate
        seq_len: Number of frames per sequence (default: 20)
        img_size: Size of each frame (default: 64)
        num_digits: Number of digits per frame (1 or 2)
        digit_size: Size of each digit (default: 28)
        action_dim: Dimension of action vectors (default: 4 for up/down/left/right)
        deterministic: If True, use fixed random seed for reproducibility
    """
    
    def __init__(
        self,
        num_sequences: int = 1000,
        seq_len: int = 20,
        img_size: int = 64,
        num_digits: int = 2,
        digit_size: int = 28,
        action_dim: int = 4,
        deterministic: bool = True,
    ):
        self.num_sequences = num_sequences
        self.seq_len = seq_len
        self.img_size = img_size
        self.num_digits = num_digits
        self.digit_size = digit_size
        self.action_dim = action_dim
        
        # Load MNIST digits
        self.digits = self._load_mnist_digits(deterministic)
        
        # Generate all sequences
        self.sequences, self.velocities, self.positions = self._generate_sequences()
        
    def _load_mnist_digits(self, deterministic):
        """Load MNIST digits from torchvision or generate simple ones."""
        try:
            from torchvision import datasets
            import os
            
            # Download MNIST if needed
            mnist_root = "/tmp/mnist"
            os.makedirs(mnist_root, exist_ok=True)
            
            dataset = datasets.MNIST(
                root=mnist_root,
                train=True,
                download=True,
            )
            
            # Extract digit images
            digits = []
            for i in range(10):
                # Get first occurrence of each digit
                for img, label in dataset:
                    if label == i:
                        digits.append(np.array(img))
                        break
            return digits
        except Exception as e:
            print(f"Could not load MNIST: {e}, using synthetic digits")
            return self._create_synthetic_digits()
    
    def _create_synthetic_digits(self):
        """Create simple synthetic digits for testing."""
        digits = []
        for i in range(10):
            img = np.zeros((28, 28), dtype=np.uint8)
            # Create simple patterns
            if i == 0:
                img[5:23, 8:20] = 255
                img[8:20, 11:17] = 0
            elif i == 1:
                img[5:23, 12:16] = 255
            elif i == 2:
                img[5:8, 8:20] = 255
                img[8:14, 17:20] = 255
                img[14:17, 8:20] = 255
                img[17:23, 8:11] = 255
                img[20:23, 8:20] = 255
            else:
                # Random pattern
                img[5:23, 8:20] = np.random.randint(100, 256, (18, 12))
            digits.append(img)
        return digits
    
    def _generate_sequences(self):
        """Generate all sequences with bouncing digits."""
        sequences = []
        all_velocities = []
        all_positions = []
        
        for seq_idx in range(self.num_sequences):
            np.random.seed(seq_idx if self.num_sequences < 10000 else None)
            
            # Initialize positions and velocities for each digit
            positions = []
            velocities = []
            digit_ids = []
            
            for _ in range(self.num_digits):
                # Random starting position
                x = np.random.randint(0, self.img_size - self.digit_size)
                y = np.random.randint(0, self.img_size - self.digit_size)
                positions.append([x, y])
                
                # Random velocity (bouncing)
                vx = np.random.choice([-3, -2, 2, 3])
                vy = np.random.choice([-3, -2, 2, 3])
                velocities.append([vx, vy])
                
                # Random digit
                digit_ids.append(np.random.randint(0, 10))
            
            # Generate frames
            frames = []
            frame_positions = []
            frame_velocities = []
            
            for t in range(self.seq_len):
                frame = np.zeros((self.img_size, self.img_size), dtype=np.float32)
                
                for i, (pos, vel, digit_id) in enumerate(zip(positions, velocities, digit_ids)):
                    x, y = pos
                    
                    # Add digit to frame
                    digit = self.digits[digit_id]
                    # Resize if needed
                    if digit.shape[0] != self.digit_size:
                        from scipy.ndimage import zoom
                        zoom_factor = self.digit_size / digit.shape[0]
                        digit = zoom(digit, zoom_factor, order=1)
                    
                    x_end = min(x + self.digit_size, self.img_size)
                    y_end = min(y + self.digit_size, self.img_size)
                    digit_x_end = x_end - x
                    digit_y_end = y_end - y
                    
                    frame[y:y_end, x:x_end] = np.maximum(
                        frame[y:y_end, x:x_end],
                        digit[:digit_y_end, :digit_x_end] / 255.0
                    )
                    
                    # Update position
                    x += vel[0]
                    y += vel[1]
                    
                    # Bounce off walls
                    if x <= 0 or x >= self.img_size - self.digit_size:
                        vel[0] = -vel[0]
                        x = max(0, min(x, self.img_size - self.digit_size))
                    if y <= 0 or y >= self.img_size - self.digit_size:
                        vel[1] = -vel[1]
                        y = max(0, min(y, self.img_size - self.digit_size))
                    
                    positions[i] = [x, y]
                    velocities[i] = vel
                
                frames.append(frame)
                frame_positions.append([p.copy() for p in positions])
                frame_velocities.append([v.copy() for v in velocities])
            
            sequences.append(np.array(frames))  # (T, H, W)
            all_positions.append(frame_positions)
            all_velocities.append(frame_velocities)
        
        return np.array(sequences), all_velocities, all_positions
    
    def quantize_frames(self, frames, num_levels=16):
        """Quantize continuous frames to discrete tokens.
        
        Args:
            frames: (T, H, W) float array in [0, 1]
            num_levels: Number of quantization levels
            
        Returns:
            (T, H, W) int array with values in [0, num_levels-1]
        """
        return (frames * (num_levels - 1)).astype(np.int64)
    
    def compute_actions_from_velocities(self, velocities):
        """Compute action vectors from velocity changes.
        
        Actions represent acceleration (change in velocity):
        - [1, 0, 0, 0]: accelerate up (decrease vy)
        - [0, 1, 0, 0]: accelerate down (increase vy)
        - [0, 0, 1, 0]: accelerate left (decrease vx)
        - [0, 0, 0, 1]: accelerate right (increase vx)
        
        Args:
            velocities: List of T velocity arrays, each [num_digits, 2]
            
        Returns:
            (T, 4) action array
        """
        T = len(velocities)
        actions = np.zeros((T, self.action_dim))
        
        for t in range(1, T):
            prev_vel = np.mean(velocities[t-1], axis=0)  # Average over digits
            curr_vel = np.mean(velocities[t], axis=0)
            
            dv = curr_vel - prev_vel
            
            # Map velocity change to action
            if dv[1] < -0.5:  # Accelerated up
                actions[t] = [1, 0, 0, 0]
            elif dv[1] > 0.5:  # Accelerated down
                actions[t] = [0, 1, 0, 0]
            elif dv[0] < -0.5:  # Accelerated left
                actions[t] = [0, 0, 1, 0]
            elif dv[0] > 0.5:  # Accelerated right
                actions[t] = [0, 0, 0, 1]
            else:
                actions[t] = [0, 0, 0, 0]  # No action
        
        return actions
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        """Get a sequence.
        
        Returns:
            dict with:
                - frames: (T, H, W) continuous frames in [0, 1]
                - tokens: (T, H, W) quantized tokens
                - actions: (T, 4) action vectors
        """
        frames = self.sequences[idx]
        velocities = self.velocities[idx]
        
        # Quantize to discrete tokens
        tokens = self.quantize_frames(frames, num_levels=16)
        
        # Compute actions from velocity changes
        actions = self.compute_actions_from_velocities(velocities)
        
        return {
            "frames": torch.from_numpy(frames).float(),
            "tokens": torch.from_numpy(tokens).long(),
            "actions": torch.from_numpy(actions).float(),
        }


def create_mnist_dataloader(
    num_sequences=1000,
    seq_len=20,
    batch_size=8,
    num_workers=4,
    img_size=64,
    num_digits=2,
):
    """Create a DataLoader for Moving MNIST.
    
    Returns tokens suitable for discrete diffusion training:
    - tokens: (B, T, H, W) with values in [0, 15] (16 levels)
    - actions: (B, T, 4) one-hot action vectors
    """
    dataset = MovingMNIST(
        num_sequences=num_sequences,
        seq_len=seq_len,
        img_size=img_size,
        num_digits=num_digits,
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    # Test the dataset
    print("Testing Moving MNIST dataset...")
    
    dataset = MovingMNIST(num_sequences=10, seq_len=20)
    sample = dataset[0]
    
    print(f"Frames shape: {sample['frames'].shape}")
    print(f"Tokens shape: {sample['tokens'].shape}")
    print(f"Actions shape: {sample['actions'].shape}")
    print(f"Token range: [{sample['tokens'].min()}, {sample['tokens'].max()}]")
    print(f"Unique tokens: {torch.unique(sample['tokens']).numel()}")
    
    # Test dataloader
    loader = create_mnist_dataloader(num_sequences=100, batch_size=4)
    batch = next(iter(loader))
    
    print(f"\nBatch frames: {batch['frames'].shape}")
    print(f"Batch tokens: {batch['tokens'].shape}")
    print(f"Batch actions: {batch['actions'].shape}")
    
    print("\nMoving MNIST dataset test passed!")
