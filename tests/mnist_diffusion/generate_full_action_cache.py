#!/usr/bin/env python3
"""
Generate action cache for full MNIST dataset (10,000 sequences).
Splits into train (8000) and val (2000) sets.
"""

import numpy as np
import pickle
import os
from pathlib import Path
from tqdm import tqdm


def generate_ego_centric_actions(data, ego_digit_id=0):
    """
    Generate ego-centric actions for all sequences.
    
    Args:
        data: (N, T, H, W) array of frames
        ego_digit_id: 0 to track the digit that appears first, 1 for second
    
    Returns:
        actions: (N, T, 2) array of [dx, dy] normalized to [-1, 1]
    """
    N, T, H, W = data.shape
    actions = np.zeros((N, T, 2), dtype=np.float32)
    
    print(f"Generating actions for {N} sequences...")
    
    for n in tqdm(range(N), desc="Processing sequences"):
        prev_center = None
        
        for t in range(T):
            frame = data[n, t].astype(np.float32)
            
            # Threshold to get digit pixels
            binary = frame > 50
            y_coords, x_coords = np.where(binary)
            
            if len(y_coords) == 0:
                # No digit visible
                if prev_center is not None:
                    actions[n, t] = [0.0, 0.0]
                continue
            
            # If two digits, try to track the correct one
            if prev_center is not None and len(y_coords) > 100:  # Likely two digits
                # Split by median y-coordinate to separate digits
                y_median = np.median(y_coords)
                mask1 = y_coords < y_median
                mask2 = ~mask1
                
                if mask1.sum() > 10 and mask2.sum() > 10:
                    center1 = np.array([y_coords[mask1].mean(), x_coords[mask1].mean()])
                    center2 = np.array([y_coords[mask2].mean(), x_coords[mask2].mean()])
                    
                    # Pick the one closer to previous center
                    dist1 = np.sum((center1 - prev_center) ** 2)
                    dist2 = np.sum((center2 - prev_center) ** 2)
                    
                    if ego_digit_id == 0:
                        curr_center = center1 if dist1 < dist2 else center2
                    else:
                        curr_center = center2 if dist1 < dist2 else center1
                else:
                    curr_center = np.array([y_coords.mean(), x_coords.mean()])
            else:
                curr_center = np.array([y_coords.mean(), x_coords.mean()])
            
            # Calculate displacement from previous frame
            if prev_center is not None:
                dy = curr_center[0] - prev_center[0]
                dx = curr_center[1] - prev_center[1]
                # Normalize by half the frame size (so 1.0 = full frame width/height)
                actions[n, t] = [dx / (W / 2), dy / (H / 2)]
            
            prev_center = curr_center
        
        # Fill first frame with second frame's motion (no displacement at t=0)
        if T > 1:
            actions[n, 0] = actions[n, 1]
    
    return actions


def main():
    # Configuration
    data_path = 'mnist_test_seq.1.npy'
    cache_dir = Path('data/mnist_cache_full')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    frame_size = 32
    ego_digit_id = 0
    
    # Train/val split (80/20)
    num_train = 8000
    num_val = 2000
    
    print("=" * 60)
    print("Generating Full Action Cache for Moving MNIST")
    print("=" * 60)
    print(f"Data path: {data_path}")
    print(f"Cache dir: {cache_dir}")
    print(f"Frame size: {frame_size}x{frame_size}")
    print(f"Train sequences: {num_train}")
    print(f"Val sequences: {num_val}")
    print(f"Total: {num_train + num_val}")
    print()
    
    # Load raw data
    print("Loading raw MNIST data...")
    raw_data = np.load(data_path)  # Shape: (20, 10000, 64, 64)
    raw_data = raw_data.transpose(1, 0, 2, 3)  # Shape: (10000, 20, 64, 64)
    print(f"Raw data shape: {raw_data.shape}")
    
    # Resize to target frame size
    if frame_size != 64:
        print(f"Resizing frames to {frame_size}x{frame_size}...")
        from PIL import Image
        N, T, H, W = raw_data.shape
        resized = np.zeros((N, T, frame_size, frame_size), dtype=raw_data.dtype)
        for n in tqdm(range(N), desc="Resizing"):
            for t in range(T):
                img = Image.fromarray(raw_data[n, t])
                img = img.resize((frame_size, frame_size), Image.BILINEAR)
                resized[n, t] = np.array(img)
        raw_data = resized
        print(f"Resized data shape: {raw_data.shape}")
    
    # Split into train and val
    train_data = raw_data[:num_train]
    val_data = raw_data[num_train:num_train + num_val]
    
    print(f"\nTrain data shape: {train_data.shape}")
    print(f"Val data shape: {val_data.shape}")
    
    # Generate actions for train set
    print("\n" + "=" * 60)
    print("Generating TRAIN actions...")
    print("=" * 60)
    train_actions = generate_ego_centric_actions(train_data, ego_digit_id=ego_digit_id)
    
    # Save train actions
    train_cache_path = cache_dir / f'actions_ego{ego_digit_id}_fs{frame_size}_0_{num_train}.pkl'
    print(f"\nSaving train actions to {train_cache_path}")
    with open(train_cache_path, 'wb') as f:
        pickle.dump(train_actions, f)
    print(f"Train actions shape: {train_actions.shape}")
    print(f"Train actions range: [{train_actions.min():.3f}, {train_actions.max():.3f}]")
    
    # Generate actions for val set
    print("\n" + "=" * 60)
    print("Generating VAL actions...")
    print("=" * 60)
    val_actions = generate_ego_centric_actions(val_data, ego_digit_id=ego_digit_id)
    
    # Save val actions
    val_cache_path = cache_dir / f'actions_ego{ego_digit_id}_fs{frame_size}_{num_train}_{num_val}.pkl'
    print(f"\nSaving val actions to {val_cache_path}")
    with open(val_cache_path, 'wb') as f:
        pickle.dump(val_actions, f)
    print(f"Val actions shape: {val_actions.shape}")
    print(f"Val actions range: [{val_actions.min():.3f}, {val_actions.max():.3f}]")
    
    print("\n" + "=" * 60)
    print("Action cache generation complete!")
    print("=" * 60)
    print(f"\nTrain: {train_cache_path}")
    print(f"Val:   {val_cache_path}")
    print(f"\nTotal sequences cached: {num_train + num_val}")


if __name__ == '__main__':
    main()
