#!/usr/bin/env python3
"""
Create a 3-second GIF of top-down (BEV) LiDAR point cloud sequence.

Usage:
    python scripts/create_top_view_gif.py \
        --sequence 00 \
        --start_frame 0 \
        --output outputs/top_view_sequence.gif
"""

import argparse
import sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import imageio

sys.path.insert(0, str(Path(__file__).parent.parent))

from copilot4d.data.point_cloud_utils import filter_roi
from copilot4d.utils.config import TokenizerConfig


def load_kitti_bin(bin_path: str) -> np.ndarray:
    """Load KITTI point cloud from .bin file."""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points


def load_sequence_frames(kitti_root: str, sequence: str, start_frame: int, num_frames: int):
    """Load a sequence of point cloud frames."""
    frames = []
    seq_path = Path(kitti_root) / "dataset" / "sequences" / sequence / "velodyne"
    
    for i in range(start_frame, start_frame + num_frames):
        bin_file = seq_path / f"{i:06d}.bin"
        if bin_file.exists():
            points = load_kitti_bin(str(bin_file))
            frames.append(points)
        else:
            print(f"Warning: Frame {i} not found at {bin_file}")
            break
    
    return frames


def create_top_down_bev(points: np.ndarray, cfg: TokenizerConfig,
                        figsize: tuple = (12, 12), dpi: int = 100,
                        point_size: float = 0.2):
    """
    Create a pure top-down BEV view of the point cloud.
    """
    # Filter to ROI
    points_filtered = filter_roi(points, cfg)
    xyz = points_filtered[:, :3]
    
    if len(xyz) == 0:
        return None
    
    # Create figure with dark background
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Color by height (z) using viridis colormap
    z_values = xyz[:, 2]
    scatter = ax.scatter(xyz[:, 0], xyz[:, 1], 
                        c=z_values, s=point_size, alpha=0.9, 
                        cmap='viridis', vmin=cfg.z_min, vmax=cfg.z_max,
                        marker='.', linewidths=0)
    
    # Set limits to ROI
    ax.set_xlim(cfg.x_min, cfg.x_max)
    ax.set_ylim(cfg.y_min, cfg.y_max)
    ax.set_aspect('equal')
    
    # Remove all axes elements for clean look
    ax.axis('off')
    
    # Remove any padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Convert to numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    
    # Convert RGBA to RGB
    img_rgb = img[:, :, :3]
    
    plt.close(fig)
    
    return img_rgb


def create_top_down_bev_with_ego(points: np.ndarray, cfg: TokenizerConfig,
                                  figsize: tuple = (12, 12), dpi: int = 100,
                                  point_size: float = 0.2):
    """
    Create top-down BEV view with ego vehicle marker.
    """
    # Filter to ROI
    points_filtered = filter_roi(points, cfg)
    xyz = points_filtered[:, :3]
    
    if len(xyz) == 0:
        return None
    
    # Create figure with dark background
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Color by height (z) using viridis colormap
    z_values = xyz[:, 2]
    scatter = ax.scatter(xyz[:, 0], xyz[:, 1], 
                        c=z_values, s=point_size, alpha=0.9, 
                        cmap='viridis', vmin=cfg.z_min, vmax=cfg.z_max,
                        marker='.', linewidths=0)
    
    # Add ego vehicle marker at origin
    # Draw a simple arrow or triangle pointing forward (x direction)
    ego_size = 1.5
    ego_triangle = plt.Polygon([
        [ego_size, 0],
        [-ego_size/2, ego_size/2],
        [-ego_size/2, -ego_size/2]
    ], fill=True, color='red', alpha=0.8, zorder=10)
    ax.add_patch(ego_triangle)
    
    # Set limits to ROI
    ax.set_xlim(cfg.x_min, cfg.x_max)
    ax.set_ylim(cfg.y_min, cfg.y_max)
    ax.set_aspect('equal')
    
    # Remove all axes elements for clean look
    ax.axis('off')
    
    # Remove any padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Convert to numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    
    # Convert RGBA to RGB
    img_rgb = img[:, :, :3]
    
    plt.close(fig)
    
    return img_rgb


def create_sequence_gif(frames: list, cfg: TokenizerConfig,
                        output_path: str, fps: int = 10,
                        with_ego: bool = False):
    """
    Create a GIF showing the sequence of LiDAR frames with top-down view.
    """
    images = []
    
    print(f"Generating top-down BEV frames...")
    
    for i, points in enumerate(frames):
        if with_ego:
            img = create_top_down_bev_with_ego(points, cfg, figsize=(12, 12), dpi=100)
        else:
            img = create_top_down_bev(points, cfg, figsize=(12, 12), dpi=100)
        
        if img is not None:
            images.append(img)
        
        if (i + 1) % 5 == 0 or i == len(frames) - 1:
            print(f"  Generated {i + 1}/{len(frames)} frames...")
    
    # Save GIF - for 3 seconds at specified fps
    duration_3sec = 3.0
    num_frames_needed = int(duration_3sec * fps)
    
    # If we have fewer frames, repeat them to fill 3 seconds
    if len(images) < num_frames_needed:
        repeat_count = (num_frames_needed // len(images)) + 1
        images = (images * repeat_count)[:num_frames_needed]
    
    print(f"\nSaving GIF to {output_path}...")
    print(f"  Total frames: {len(images)} at {fps} fps = {len(images)/fps:.1f} seconds")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, images, fps=fps, loop=0)
    print(f"✅ Saved {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_root", default="data/kitti/pykitti")
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=30)
    parser.add_argument("--fps", type=int, default=10, help="FPS for GIF")
    parser.add_argument("--output", default="outputs/top_view_3sec.gif")
    parser.add_argument("--with_ego", action="store_true", 
                       help="Add ego vehicle marker")
    args = parser.parse_args()
    
    # Configuration
    cfg = TokenizerConfig()
    
    # Load frames
    print(f"Loading sequence {args.sequence} from frame {args.start_frame}...")
    frames = load_sequence_frames(args.kitti_root, args.sequence, 
                                   args.start_frame, args.num_frames)
    print(f"Loaded {len(frames)} frames")
    
    if len(frames) == 0:
        print("Error: No frames loaded!")
        return
    
    # Create GIF
    create_sequence_gif(frames, cfg, args.output, fps=args.fps, with_ego=args.with_ego)


if __name__ == "__main__":
    main()
