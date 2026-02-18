#!/usr/bin/env python3
"""Create GIF showing tokenizer reconstruction vs real LiDAR over time.

Generates a side-by-side comparison GIF showing:
1. Real LiDAR point cloud (ground truth) over time
2. Tokenizer reconstruction over time

Duration: 5 seconds (50 frames at 10Hz)

Usage:
    python scripts/create_tokenizer_timeline_gif.py \
        --checkpoint outputs/tokenizer_memory_efficient/checkpoint_step_100000.pt \
        --config configs/tokenizer_memory_efficient.yaml \
        --sequence 00 \
        --start_frame 500 \
        --output_dir outputs/tokenizer_timeline
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
import io

sys.path.insert(0, str(Path(__file__).parent.parent))

from copilot4d.tokenizer.tokenizer_model import CoPilot4DTokenizer
from copilot4d.utils.config import TokenizerConfig
from copilot4d.data.point_cloud_utils import filter_roi, generate_rays
from copilot4d.data.kitti_dataset import KITTITokenizerDataset, tokenizer_collate_fn
from pykitti import odometry


def load_config(config_path: str) -> TokenizerConfig:
    """Load config from YAML file."""
    import yaml
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return TokenizerConfig(**config_dict)


def load_model(checkpoint_path: str, cfg: TokenizerConfig, device: str = "cuda"):
    """Load tokenizer model."""
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CoPilot4DTokenizer(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    step = checkpoint.get("step", "?")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded checkpoint step {step} ({n_params:.2f}M params)")
    
    return model, step


def load_original_lidar(kitti_root: str, sequence: str, frame_idx: int, cfg: TokenizerConfig):
    """Load original LiDAR point cloud."""
    try:
        dataset = odometry(kitti_root, sequence)
        points = dataset.get_velo(frame_idx)
        points_filtered = filter_roi(points, cfg)
        return points_filtered[:, :3]
    except Exception as e:
        print(f"Warning: Could not load LiDAR frame {frame_idx}: {e}")
        return None


@torch.no_grad()
def reconstruct_frame(model, cfg, device, sequence: str, frame_idx: int):
    """Reconstruct a single frame using the tokenizer."""
    # Create dataset for this specific frame
    dataset = KITTITokenizerDataset(cfg, sequences=[sequence])
    
    # Find the index for this frame
    sample_idx = None
    for idx, (seq, frm) in enumerate(dataset.samples):
        if seq == sequence and frm == frame_idx:
            sample_idx = idx
            break
    
    if sample_idx is None:
        print(f"Warning: Frame {frame_idx} not found in dataset")
        return None
    
    # Get sample
    sample = dataset[sample_idx]
    batch = tokenizer_collate_fn([sample])
    
    # Encode
    features = batch["features"].to(device)
    num_points = batch["num_points"].to(device)
    coords = batch["coords"].to(device)
    batch_size = batch["batch_size"]
    
    bev = model.encode_voxels(features, num_points, coords, batch_size)
    encoder_out = model.encoder(bev)
    quantized, indices, _, vq_metrics = model.vq(encoder_out)
    
    # Decode to NFG
    decoder_output = model.decoder(quantized)
    nfg = model.nfg.build_nfg(decoder_output)
    
    # Get original points for ray directions
    raw_points = dataset.datasets[sequence].get_velo(frame_idx)
    points_roi = filter_roi(raw_points, cfg)
    
    if len(points_roi) == 0:
        return np.zeros((0, 3))
    
    # Generate rays from original points
    ray_data = generate_rays(points_roi, cfg)
    ray_origins = ray_data["ray_origins"]
    ray_directions = ray_data["ray_directions"]
    gt_depths = ray_data["ray_depths"]
    
    # Filter valid depths
    valid = (gt_depths >= cfg.ray_depth_min) & (gt_depths <= cfg.ray_depth_max)
    ray_origins = ray_origins[valid]
    ray_directions = ray_directions[valid]
    
    if len(ray_directions) == 0:
        return np.zeros((0, 3))
    
    # Render depths in chunks
    chunk_size = 1024
    all_depths = []
    
    origins_t = torch.from_numpy(ray_origins).unsqueeze(0).to(device)
    dirs_t = torch.from_numpy(ray_directions).unsqueeze(0).to(device)
    
    for start in range(0, len(ray_directions), chunk_size):
        end = min(start + chunk_size, len(ray_directions))
        ro = origins_t[:, start:end]
        rd = dirs_t[:, start:end]
        
        depths, _ = model.nfg.query_rays(nfg, ro, rd, cfg.ray_depth_min, cfg.ray_depth_max)
        all_depths.append(depths.squeeze(0).cpu())
    
    pred_depths = torch.cat(all_depths, dim=0)
    
    # Convert to 3D points
    pred_points = ray_origins + ray_directions * pred_depths.numpy()[:, None]
    
    return pred_points


def create_frame_image(gt_points, recon_points, cfg, frame_idx, total_frames, 
                       sequence, fps=10):
    """Create a side-by-side comparison image for a single frame."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    x_min, x_max = cfg.x_min, cfg.x_max
    y_min, y_max = cfg.y_min, cfg.y_max
    z_min, z_max = cfg.z_min, cfg.z_max
    
    # Panel 1: Ground Truth LiDAR
    ax = axes[0]
    if len(gt_points) > 0:
        sc = ax.scatter(gt_points[:, 0], gt_points[:, 1], c=gt_points[:, 2], 
                       s=0.15, cmap="viridis", vmin=z_min, vmax=z_max, alpha=0.8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title(f"Real LiDAR\n({len(gt_points):,} points)", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    if len(gt_points) > 0:
        plt.colorbar(sc, cax=cax, label="Z (m)")
    
    # Panel 2: Tokenizer Reconstruction
    ax = axes[1]
    if len(recon_points) > 0:
        sc2 = ax.scatter(recon_points[:, 0], recon_points[:, 1], c=recon_points[:, 2],
                        s=0.15, cmap="viridis", vmin=z_min, vmax=z_max, alpha=0.8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title(f"Tokenizer Reconstruction\n({len(recon_points):,} points)", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    if len(recon_points) > 0:
        plt.colorbar(sc2, cax=cax, label="Z (m)")
    
    # Overall title
    time_sec = frame_idx / fps
    fig.suptitle(f"Sequence {sequence} | Frame {frame_idx} | Time: {time_sec:.1f}s | "
                 f"Progress: {frame_idx}/{total_frames}", 
                 fontsize=14, fontweight='bold')
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Convert to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img


def create_overlay_frame(gt_points, recon_points, cfg, frame_idx, total_frames,
                         sequence, fps=10):
    """Create an overlay comparison image showing both GT and reconstruction."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    x_min, x_max = cfg.x_min, cfg.x_max
    y_min, y_max = cfg.y_min, cfg.y_max
    z_min, z_max = cfg.z_min, cfg.z_max
    
    # Plot Ground Truth in blue
    if len(gt_points) > 0:
        ax.scatter(gt_points[:, 0], gt_points[:, 1], c='dodgerblue', 
                  s=0.2, alpha=0.5, label=f'Real LiDAR ({len(gt_points):,} pts)')
    
    # Plot Reconstruction in red
    if len(recon_points) > 0:
        ax.scatter(recon_points[:, 0], recon_points[:, 1], c='red',
                  s=0.2, alpha=0.5, label=f'Reconstruction ({len(recon_points):,} pts)')
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.legend(loc='upper right', markerscale=15, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Title
    time_sec = frame_idx / fps
    ax.set_title(f"Sequence {sequence} | Frame {frame_idx} | Time: {time_sec:.1f}s\n"
                 f"Blue=Real LiDAR | Red=Tokenizer Reconstruction", 
                 fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img


def main():
    parser = argparse.ArgumentParser(description="Create tokenizer timeline GIF")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to tokenizer checkpoint")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to config YAML")
    parser.add_argument("--sequence", type=str, default="00",
                       help="KITTI sequence to use")
    parser.add_argument("--start_frame", type=int, default=500,
                       help="Starting frame index")
    parser.add_argument("--num_frames", type=int, default=50,
                       help="Number of frames (50 = 5 seconds at 10Hz)")
    parser.add_argument("--fps", type=int, default=10,
                       help="Frame rate for GIF playback")
    parser.add_argument("--output_dir", type=str, default="outputs/tokenizer_timeline",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overlay", action="store_true",
                       help="Also create overlay version (both in one frame)")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    print("Loading config...")
    cfg = load_config(args.config)
    print(f"  ROI: x=[{cfg.x_min}, {cfg.x_max}], y=[{cfg.y_min}, {cfg.y_max}]")
    print(f"  Token grid: {cfg.token_grid_size}x{cfg.token_grid_size}")
    
    # Load model
    model, ckpt_step = load_model(args.checkpoint, cfg, device)
    
    # KITTI root path
    kitti_root = getattr(cfg, 'kitti_root', 'data/kitti/dataset')
    if 'pykitti' in kitti_root:
        kitti_root = 'data/kitti/dataset'
    
    print(f"\nGenerating timeline GIF...")
    print(f"  Sequence: {args.sequence}")
    print(f"  Frames: {args.start_frame} to {args.start_frame + args.num_frames - 1}")
    print(f"  Duration: {args.num_frames / args.fps:.1f} seconds")
    print()
    
    # Generate frames
    side_by_side_frames = []
    overlay_frames = []
    
    for i in range(args.num_frames):
        frame_idx = args.start_frame + i
        print(f"Processing frame {frame_idx} ({i+1}/{args.num_frames})...")
        
        # Load real LiDAR
        gt_points = load_original_lidar(kitti_root, args.sequence, frame_idx, cfg)
        if gt_points is None:
            print(f"  Warning: Could not load frame {frame_idx}, skipping")
            continue
        
        # Generate reconstruction
        print(f"  Reconstructing...")
        recon_points = reconstruct_frame(model, cfg, device, args.sequence, frame_idx)
        
        if recon_points is None:
            print(f"  Warning: Reconstruction failed for frame {frame_idx}")
            continue
        
        # Create side-by-side image
        img = create_frame_image(
            gt_points, recon_points, cfg, 
            frame_idx, args.start_frame + args.num_frames, 
            args.sequence, args.fps
        )
        side_by_side_frames.append(img)
        
        # Save individual frame
        img.save(output_dir / f"frame_{frame_idx:06d}.png")
        
        # Create overlay if requested
        if args.overlay:
            img_overlay = create_overlay_frame(
                gt_points, recon_points, cfg,
                frame_idx, args.start_frame + args.num_frames,
                args.sequence, args.fps
            )
            overlay_frames.append(img_overlay)
            img_overlay.save(output_dir / f"overlay_{frame_idx:06d}.png")
    
    if len(side_by_side_frames) == 0:
        print("\nError: No frames generated!")
        return
    
    # Create GIFs
    print(f"\nCreating GIFs...")
    
    # Side-by-side GIF
    gif_path = output_dir / f"tokenizer_timeline_seq{args.sequence}_{args.start_frame}.gif"
    side_by_side_frames[0].save(
        gif_path,
        save_all=True,
        append_images=side_by_side_frames[1:],
        duration=int(1000 / args.fps),  # ms per frame
        loop=0
    )
    print(f"  Side-by-side GIF: {gif_path}")
    
    # Overlay GIF
    if args.overlay and overlay_frames:
        overlay_gif_path = output_dir / f"tokenizer_overlay_seq{args.sequence}_{args.start_frame}.gif"
        overlay_frames[0].save(
            overlay_gif_path,
            save_all=True,
            append_images=overlay_frames[1:],
            duration=int(1000 / args.fps),
            loop=0
        )
        print(f"  Overlay GIF: {overlay_gif_path}")
    
    print(f"\n{'='*70}")
    print(f"✅ Timeline GIF creation complete!")
    print(f"   Checkpoint: step {ckpt_step}")
    print(f"   Sequence: {args.sequence}")
    print(f"   Frames: {args.start_frame}-{args.start_frame + len(side_by_side_frames) - 1}")
    print(f"   Output: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
