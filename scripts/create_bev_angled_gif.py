#!/usr/bin/env python3
"""
Create a 3-second GIF of BEV angled LiDAR point cloud sequence.

Usage:
    python scripts/create_bev_angled_gif.py \
        --sequence 00 \
        --start_frame 0 \
        --output outputs/bev_angled_sequence.gif
"""

import argparse
import sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
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


def create_bev_angled_view(points: np.ndarray, cfg: TokenizerConfig, 
                           azimuth_deg: float = 45, elevation_deg: float = 60,
                           figsize: tuple = (12, 10), dpi: int = 100):
    """
    Create a BEV angled view of the point cloud.
    
    Args:
        points: (N, 4) array [x, y, z, reflectance]
        cfg: TokenizerConfig with ROI bounds
        azimuth_deg: Azimuth angle in degrees (rotation around Z axis)
        elevation_deg: Elevation angle in degrees (looking down from above)
    """
    # Filter to ROI
    points_filtered = filter_roi(points, cfg)
    xyz = points_filtered[:, :3]
    
    if len(xyz) == 0:
        return None
    
    # Create figure with dark background
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    # Hide grid and axes for clean look
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    
    # Color points by height (z value)
    z_values = xyz[:, 2]
    colors = plt.cm.viridis((z_values - z_values.min()) / (z_values.max() - z_values.min() + 1e-6))
    
    # Scatter plot with small points for LiDAR-like appearance
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
               c=colors, s=0.5, alpha=0.8, marker='.')
    
    # Set axis limits based on ROI
    ax.set_xlim(cfg.x_min, cfg.x_max)
    ax.set_ylim(cfg.y_min, cfg.y_max)
    ax.set_zlim(cfg.z_min, cfg.z_max)
    
    # Set view angle (BEV angled)
    ax.view_init(elev=elevation_deg, azim=azimuth_deg)
    
    # Remove axis labels and ticks for clean look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Add coordinate frame arrows at origin (small)
    arrow_len = 5.0
    ax.quiver(0, 0, 0, arrow_len, 0, 0, color='red', alpha=0.8, arrow_length_ratio=0.3)
    ax.quiver(0, 0, 0, 0, arrow_len, 0, color='green', alpha=0.8, arrow_length_ratio=0.3)
    ax.quiver(0, 0, 0, 0, 0, arrow_len, color='blue', alpha=0.8, arrow_length_ratio=0.3)
    
    plt.tight_layout(pad=0)
    
    # Convert to numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return img


def create_bev_angled_view_matplotlib2d(points: np.ndarray, cfg: TokenizerConfig,
                                        view_angle_deg: float = 45,
                                        figsize: tuple = (12, 10), dpi: int = 100):
    """
    Create a BEV angled view using 2D projection (faster rendering).
    Projects 3D points to 2D with perspective based on view angle.
    """
    # Filter to ROI
    points_filtered = filter_roi(points, cfg)
    xyz = points_filtered[:, :3]
    
    if len(xyz) == 0:
        return None
    
    # Rotation matrix for view angle around Z axis
    theta = np.deg2rad(view_angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Rotate points around Z axis
    x_rot = xyz[:, 0] * cos_t - xyz[:, 1] * sin_t
    y_rot = xyz[:, 0] * sin_t + xyz[:, 1] * cos_t
    z_rot = xyz[:, 2]
    
    # Create figure with dark background
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Project to 2D with perspective (simple orthographic for BEV)
    # Use x_rot as horizontal, z as vertical (elevation view)
    # Add some depth cue using y_rot for brightness
    depth_factor = (y_rot - y_rot.min()) / (y_rot.max() - y_rot.min() + 1e-6)
    
    # Color based on height and depth
    z_normalized = (z_rot - cfg.z_min) / (cfg.z_max - cfg.z_min + 1e-6)
    
    # Create viridis-like colormap with depth variation
    colors = plt.cm.viridis(z_normalized)
    # Adjust brightness based on depth
    brightness = 0.7 + 0.3 * depth_factor
    colors[:, :3] *= brightness[:, None]
    
    # Scatter plot
    ax.scatter(x_rot, z_rot, c=colors, s=0.5, alpha=0.8, marker='.')
    
    # Set limits
    margin = 5
    ax.set_xlim(cfg.x_min - margin, cfg.x_max + margin)
    ax.set_zlim(cfg.z_min - 1, cfg.z_max + 1)
    
    # Remove axes
    ax.axis('off')
    
    plt.tight_layout(pad=0)
    
    # Convert to numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return img


def create_top_down_bev(points: np.ndarray, cfg: TokenizerConfig,
                        figsize: tuple = (12, 10), dpi: int = 100):
    """
    Create a top-down BEV view with height-based coloring.
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
    
    # Color by height (z)
    z_values = xyz[:, 2]
    scatter = ax.scatter(xyz[:, 0], xyz[:, 1], 
                        c=z_values, s=0.3, alpha=0.8, 
                        cmap='viridis', vmin=cfg.z_min, vmax=cfg.z_max,
                        marker='.')
    
    # Set limits
    ax.set_xlim(cfg.x_min, cfg.x_max)
    ax.set_ylim(cfg.y_min, cfg.y_max)
    ax.set_aspect('equal')
    
    # Remove axes
    ax.axis('off')
    
    plt.tight_layout(pad=0)
    
    # Convert to numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return img


def create_angled_perspective_view(points: np.ndarray, cfg: TokenizerConfig,
                                   figsize: tuple = (16, 10), dpi: int = 100):
    """
    Create an angled perspective view similar to the reference image.
    Uses 3D matplotlib with custom viewing angle.
    """
    # Filter to ROI
    points_filtered = filter_roi(points, cfg)
    xyz = points_filtered[:, :3]
    
    if len(xyz) == 0:
        return None
    
    # Create figure with dark background
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    # Hide axes panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    
    # Remove grid
    ax.grid(False)
    
    # Color points by height
    z_values = xyz[:, 2]
    norm = Normalize(vmin=z_values.min(), vmax=z_values.max())
    colors = plt.cm.viridis(norm(z_values))
    
    # Scatter plot with very small points for dense LiDAR look
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
               c=colors, s=0.3, alpha=0.9, marker='.', linewidths=0)
    
    # Set limits
    ax.set_xlim(cfg.x_min, cfg.x_max)
    ax.set_ylim(cfg.y_min, cfg.y_max)
    ax.set_zlim(cfg.z_min, cfg.z_max)
    
    # Set view to match reference (angled BEV from behind/side)
    ax.view_init(elev=55, azim=-75)
    
    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    plt.tight_layout(pad=0)
    
    # Convert to numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return img


def create_orbit_animation(frames: list, cfg: TokenizerConfig, 
                           output_path: str, fps: int = 30):
    """
    Create a GIF that shows both temporal sequence and rotating view.
    
    For a 3-second GIF at 30fps = 90 frames total.
    We'll show 30 unique frames with 3-frame repeats for smooth playback.
    """
    duration_3sec = 3.0  # seconds
    total_frames = int(duration_3sec * fps)  # 90 frames
    
    # Number of unique LiDAR frames to show
    num_lidar_frames = min(len(frames), 30)
    
    # Frames per LiDAR scan
    frames_per_scan = total_frames // num_lidar_frames
    
    # Azimuth angles for rotation (full 360 degree rotation)
    azimuth_angles = np.linspace(0, 360, total_frames, endpoint=False)
    
    images = []
    
    print(f"Generating {total_frames} frames for 3-second GIF at {fps} fps...")
    print(f"Using {num_lidar_frames} unique LiDAR scans")
    
    for i in range(total_frames):
        # Determine which LiDAR frame to use
        lidar_frame_idx = min(i // frames_per_scan, num_lidar_frames - 1)
        points = frames[lidar_frame_idx]
        
        # Get azimuth angle for this frame
        azimuth = azimuth_angles[i]
        
        # Create view
        img = create_bev_angled_view(points, cfg, azimuth_deg=azimuth, 
                                     elevation_deg=55, figsize=(12, 9), dpi=100)
        
        if img is not None:
            images.append(img)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{total_frames} frames...")
    
    # Save GIF
    print(f"\nSaving GIF to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, images, fps=fps, loop=0)
    print(f"✅ Saved {output_path}")
    
    return output_path


def create_simple_sequence_gif(frames: list, cfg: TokenizerConfig,
                               output_path: str, fps: int = 10):
    """
    Create a simple GIF showing the sequence of LiDAR frames with fixed angled view.
    """
    images = []
    
    print(f"Generating frames for sequence GIF...")
    
    for i, points in enumerate(frames):
        # Create angled perspective view matching the reference image style
        img = create_angled_perspective_view(points, cfg, figsize=(16, 10), dpi=100)
        
        if img is not None:
            images.append(img)
        
        if (i + 1) % 5 == 0:
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
    parser.add_argument("--output", default="outputs/bev_angled_sequence.gif")
    parser.add_argument("--orbit", action="store_true", 
                       help="Create orbiting camera animation")
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
    if args.orbit:
        create_orbit_animation(frames, cfg, args.output, fps=args.fps)
    else:
        create_simple_sequence_gif(frames, cfg, args.output, fps=args.fps)


if __name__ == "__main__":
    main()
