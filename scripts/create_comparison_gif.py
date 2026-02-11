#!/usr/bin/env python3
"""
Create GIF comparison of:
1. Original LiDAR (ground truth)
2. Tokenizer reconstruction (from GT tokens)
3. World model prediction

Shows evolution/deterioration over a sequence of frames.

Usage:
    python scripts/create_comparison_gif.py \
        --checkpoint outputs/world_model_12gb/checkpoint_0050000.pt \
        --tokenizer outputs/tokenizer_memory_efficient/checkpoint_step_80000.pt \
        --sequence 00 \
        --start_frame 500 \
        --num_frames 10 \
        --output_dir outputs/gif_comparison
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
import io

sys.path.insert(0, str(Path(__file__).parent.parent))

from copilot4d.world_model.world_model import CoPilot4DWorldModel, WorldModelConfig
from copilot4d.world_model.masking import cosine_mask_schedule
from copilot4d.tokenizer.tokenizer_model import CoPilot4DTokenizer
from copilot4d.utils.config import TokenizerConfig
from copilot4d.data.point_cloud_utils import filter_roi
from pykitti import odometry


def load_models(world_model_ckpt: str, tokenizer_ckpt: str, device: str = "cuda"):
    """Load world model and tokenizer."""
    print("Loading models...")
    
    # Load world model
    wm_checkpoint = torch.load(world_model_ckpt, map_location=device)
    wm_cfg = wm_checkpoint["config"]
    wm_cfg.num_sampling_steps = getattr(wm_cfg, 'num_sampling_steps', 16)
    wm_cfg.choice_temperature = getattr(wm_cfg, 'choice_temperature', 4.5)
    
    world_model = CoPilot4DWorldModel(wm_cfg).to(device)
    wm_state = wm_checkpoint.get("model_state_dict", wm_checkpoint.get("model"))
    world_model.load_state_dict(wm_state)
    world_model.eval()
    
    # Load tokenizer
    tok_checkpoint = torch.load(tokenizer_ckpt, map_location=device)
    tok_cfg = tok_checkpoint.get("config")
    if tok_cfg is None:
        tok_cfg = TokenizerConfig(
            img_size=(192, 640),
            patch_size=(3, 10),
            codebook_size=1024,
            codebook_dim=256,
            num_scales=3,
        )
    
    tokenizer = CoPilot4DTokenizer(tok_cfg).to(device)
    tokenizer.load_state_dict(tok_checkpoint.get("model", tok_checkpoint))
    tokenizer.eval()
    
    print(f"  World model: {sum(p.numel() for p in world_model.parameters())/1e6:.1f}M params")
    print(f"  Tokenizer: {sum(p.numel() for p in tokenizer.parameters())/1e6:.1f}M params")
    
    return world_model, tokenizer, wm_cfg, tok_cfg


def generate_dense_rays(n_azimuth=900, elevation_min_deg=-25.0, 
                        elevation_max_deg=5.0, n_elevation=32):
    """Generate dense spherical grid of ray directions."""
    az = np.linspace(0, 2 * np.pi, n_azimuth, endpoint=False)
    el = np.deg2rad(np.linspace(elevation_min_deg, elevation_max_deg, n_elevation))
    az_grid, el_grid = np.meshgrid(az, el, indexing="ij")
    az_flat = az_grid.ravel()
    el_flat = el_grid.ravel()
    
    x = np.cos(el_flat) * np.cos(az_flat)
    y = np.cos(el_flat) * np.sin(az_flat)
    z = np.sin(el_flat)
    
    directions = np.stack([x, y, z], axis=-1).astype(np.float32)
    origins = np.zeros_like(directions)
    return origins, directions


@torch.no_grad()
def tokens_to_pointcloud(tokenizer, tokens, pose=None, n_azimuth=900, n_elevation=32,
                         device="cuda", chunk_size=2048):
    """Convert tokens to dense point cloud via spherical raycasting.
    
    Returns points in ego coordinates (no pose transformation).
    This ensures fair comparison with original LiDAR which is also in ego coords.
    """
    H, W = tokens.shape
    tokens_flat = tokens.reshape(H * W).long().to(device)
    
    # Lookup codebook embeddings
    vq = tokenizer.vq
    z_q = vq.embed[tokens_flat]
    z_q = vq.post_proj(z_q)
    z_q = z_q.unsqueeze(0)
    
    # Decode
    decoder_output = tokenizer.decoder(z_q)
    nfg = tokenizer.nfg.build_nfg(decoder_output)
    
    # Generate dense rays
    ray_origins, ray_directions = generate_dense_rays(n_azimuth, -25.0, 5.0, n_elevation)
    
    # Render depths in chunks
    all_depths = []
    N_rays = ray_directions.shape[0]
    
    ray_origins_t = torch.from_numpy(ray_origins).float().unsqueeze(0).to(device)
    ray_dirs_t = torch.from_numpy(ray_directions).float().unsqueeze(0).to(device)
    
    for start in range(0, N_rays, chunk_size):
        end = min(start + chunk_size, N_rays)
        ro = ray_origins_t[:, start:end]
        rd = ray_dirs_t[:, start:end]
        
        depths, _ = tokenizer.nfg.query_rays(
            nfg, ro, rd,
            depth_min=tokenizer.cfg.ray_depth_min,
            depth_max=tokenizer.cfg.ray_depth_max
        )
        all_depths.append(depths.squeeze(0).cpu())
    
    pred_depths = torch.cat(all_depths, dim=0)
    
    # Convert to 3D points
    pred_depths_np = pred_depths.numpy()
    cfg_max_depth = tokenizer.cfg.ray_depth_max
    valid_mask = pred_depths_np < cfg_max_depth * 0.95
    
    if valid_mask.sum() == 0:
        return np.zeros((0, 3))
    
    depths_valid = pred_depths_np[valid_mask]
    directions_valid = ray_directions[valid_mask]
    
    # Points in ego coordinates (no pose transformation)
    points = directions_valid * depths_valid[:, None]
    
    # NOTE: Removed pose rotation to keep everything in ego coordinates
    # This ensures fair comparison with original LiDAR
    # Original: R = pose[:3, :3]; points = (R.T @ points.T).T
    
    return points


def load_original_lidar(kitti_root, sequence, frame_idx, cfg):
    """Load original LiDAR point cloud."""
    try:
        dataset = odometry(kitti_root, sequence)
        points = dataset.get_velo(frame_idx)
        points_filtered = filter_roi(points, cfg)
        return points_filtered[:, :3]
    except Exception as e:
        print(f"Warning: Could not load LiDAR: {e}")
        return None


def create_frame_comparison(original_pc, tokenized_pc, predicted_pc, 
                            roi_bounds, frame_idx, total_frames, fps=10, 
                            sequence="00", start_frame=0):
    """Create a single comparison frame with all three views."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    x_min, x_max = roi_bounds['x']
    y_min, y_max = roi_bounds['y']
    z_min, z_max = roi_bounds['z']
    
    # Filter to ROI
    def filter_roi_points(pts, bounds):
        if len(pts) == 0:
            return pts
        mask = (
            (pts[:, 0] >= bounds['x'][0]) & (pts[:, 0] <= bounds['x'][1]) &
            (pts[:, 1] >= bounds['y'][0]) & (pts[:, 1] <= bounds['y'][1]) &
            (pts[:, 2] >= bounds['z'][0]) & (pts[:, 2] <= bounds['z'][1])
        )
        return pts[mask]
    
    orig_roi = filter_roi_points(original_pc, roi_bounds)
    token_roi = filter_roi_points(tokenized_pc, roi_bounds)
    pred_roi = filter_roi_points(predicted_pc, roi_bounds)
    
    # Panel 1: Original LiDAR
    ax = axes[0]
    if len(orig_roi) > 0:
        sc = ax.scatter(orig_roi[:, 0], orig_roi[:, 1], c=orig_roi[:, 2], 
                       s=0.1, cmap="viridis", vmin=z_min, vmax=z_max, alpha=0.8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Original LiDAR\n({len(orig_roi):,} points)")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(sc, cax=cax, label="Z (m)")
    
    # Panel 2: Tokenizer Reconstruction
    ax = axes[1]
    if len(token_roi) > 0:
        sc = ax.scatter(token_roi[:, 0], token_roi[:, 1], c=token_roi[:, 2],
                       s=0.1, cmap="viridis", vmin=z_min, vmax=z_max, alpha=0.8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Tokenizer Reconstruction\n({len(token_roi):,} points)")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(sc, cax=cax, label="Z (m)")
    
    # Panel 3: World Model Prediction
    ax = axes[2]
    if len(pred_roi) > 0:
        sc = ax.scatter(pred_roi[:, 0], pred_roi[:, 1], c=pred_roi[:, 2],
                       s=0.1, cmap="viridis", vmin=z_min, vmax=z_max, alpha=0.8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"World Model Prediction\n({len(pred_roi):,} points)")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(sc, cax=cax, label="Z (m)")
    
    time_sec = frame_idx / fps
    fig.suptitle(f"Frame +{frame_idx} ({time_sec:.1f}s ahead) | Seq {sequence} Frame {start_frame}+{frame_idx}", 
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img


@torch.no_grad()
def predict_tokens_autoregressive(world_model, initial_tokens, initial_actions, 
                                  num_frames, cfg, device):
    """Predict multiple future frames autoregressively."""
    predictions = []
    
    # Use last known tokens/actions as starting point
    past_tokens = initial_tokens.to(device)
    past_actions = initial_actions.to(device)
    
    for i in range(num_frames):
        # Predict next frame
        pred_frame = predict_single_frame(
            world_model, past_tokens, past_actions, cfg, device
        )
        predictions.append(pred_frame.cpu())
        
        # Update history for next prediction
        past_tokens = torch.cat([past_tokens[1:], pred_frame.unsqueeze(0)], dim=0)
        # Use last action as approximation
        past_actions = torch.cat([past_actions[1:], past_actions[-1:]], dim=0)
    
    return predictions


def predict_single_frame(world_model, past_tokens, past_actions, cfg, device):
    """Predict single future frame."""
    T_past, H, W = past_tokens.shape
    N = H * W
    mask_id = cfg.codebook_size
    
    # Single future frame
    future_token = torch.full((1, H, W), mask_id, dtype=torch.long, device=device)
    
    full_tokens = torch.cat([past_tokens, future_token], dim=0)
    full_actions = torch.cat([past_actions, past_actions[-1:]], dim=0)
    
    # Handle sequence length
    if full_tokens.shape[0] > cfg.num_frames:
        full_tokens = full_tokens[-cfg.num_frames:]
        full_actions = full_actions[-cfg.num_frames:]
        T_past = cfg.num_frames - 1
    
    T_total = full_tokens.shape[0]
    temporal_mask = torch.triu(torch.ones(T_total, T_total) * float('-inf'), diagonal=1).to(device)
    
    num_steps = cfg.num_sampling_steps
    
    for step in range(num_steps):
        logits = world_model(full_tokens.unsqueeze(0), full_actions.unsqueeze(0), temporal_mask)
        logits_future = logits[0, -1:]
        
        predictions = logits_future.argmax(dim=-1)
        future_flat = future_token.reshape(1, N)
        mask = (future_flat == mask_id)
        future_flat = torch.where(mask, predictions, future_flat)
        future_token = future_flat.reshape(1, H, W)
        full_tokens = torch.cat([full_tokens[:-1], future_token], dim=0)
        
        # Remask
        if step < num_steps - 1:
            t = (step + 1) / num_steps
            mask_ratio = cosine_mask_schedule(torch.tensor(t)).item()
            num_to_mask = int(mask_ratio * N)
            
            if num_to_mask > 0:
                confidences = logits_future.max(dim=-1).values
                flat_conf = confidences.reshape(-1)
                _, indices = torch.topk(flat_conf, k=num_to_mask, largest=False)
                future_flat = future_token.reshape(-1)
                future_flat[indices] = mask_id
                future_token = future_flat.reshape(1, H, W)
                full_tokens = torch.cat([full_tokens[:-1], future_token], dim=0)
    
    return future_token[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="World model checkpoint")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer checkpoint")
    parser.add_argument("--token_dir", default="data/kitti/tokens")
    parser.add_argument("--kitti_root", default="data/kitti/dataset")
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--start_frame", type=int, default=500)
    parser.add_argument("--num_past_frames", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=10, help="Number of future frames to predict")
    parser.add_argument("--dense_azimuth", type=int, default=900)
    parser.add_argument("--dense_elevation", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--fps", type=int, default=10, help="Frame rate for time display")
    parser.add_argument("--output_dir", default="outputs/gif_comparison")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    world_model, tokenizer, wm_cfg, tok_cfg = load_models(
        args.checkpoint, args.tokenizer, device
    )
    
    # Load past frames
    print(f"\nLoading past frames ({args.num_past_frames})...")
    seq_path = Path(args.token_dir) / args.sequence
    
    past_tokens_list = []
    for i in range(args.start_frame, args.start_frame + args.num_past_frames):
        token_file = seq_path / f"{i:06d}.pt"
        tokens = torch.load(token_file, map_location="cpu")
        past_tokens_list.append(tokens)
    past_tokens = torch.stack(past_tokens_list)
    
    # Load poses
    poses_file = seq_path / "poses.pkl"
    with open(poses_file, 'rb') as f:
        all_poses = pickle.load(f)
    
    past_poses = [all_poses[i] for i in range(args.start_frame, args.start_frame + args.num_past_frames)]
    future_poses = [all_poses[i] for i in range(args.start_frame + args.num_past_frames, 
                                               args.start_frame + args.num_past_frames + args.num_frames)]
    
    # Compute actions
    actions = []
    for i in range(1, len(past_poses)):
        T_rel = np.linalg.inv(past_poses[i-1]) @ past_poses[i]
        actions.append(T_rel.flatten())
    actions.append(np.eye(4).flatten())
    past_actions = torch.from_numpy(np.stack(actions)).float()
    
    print(f"  Past tokens: {past_tokens.shape}")
    
    # Predict future frames
    print(f"\nPredicting {args.num_frames} future frames...")
    predicted_tokens = predict_tokens_autoregressive(
        world_model, past_tokens, past_actions, args.num_frames, wm_cfg, device
    )
    
    # ROI bounds
    roi_bounds = {
        'x': (-35.0, 35.0),
        'y': (-35.0, 35.0),
        'z': (-3.0, 3.0),
    }
    
    # Generate comparison frames
    print("\nGenerating comparison frames...")
    frames = []
    
    for i in range(args.num_frames):
        frame_idx = args.start_frame + args.num_past_frames + i
        print(f"  Frame {frame_idx} (prediction +{i+1})...")
        
        # Load original LiDAR
        original_pc = load_original_lidar(
            args.kitti_root, args.sequence, frame_idx, tok_cfg
        )
        
        # Load GT tokens and reconstruct
        gt_token_file = seq_path / f"{frame_idx:06d}.pt"
        gt_tokens = torch.load(gt_token_file, map_location="cpu")
        tokenized_pc = tokens_to_pointcloud(
            tokenizer, gt_tokens, future_poses[i],
            n_azimuth=args.dense_azimuth,
            n_elevation=args.dense_elevation,
            device=device,
            chunk_size=args.chunk_size
        )
        
        # World model prediction
        predicted_pc = tokens_to_pointcloud(
            tokenizer, predicted_tokens[i], future_poses[i],
            n_azimuth=args.dense_azimuth,
            n_elevation=args.dense_elevation,
            device=device,
            chunk_size=args.chunk_size
        )
        
        # Create comparison image
        img = create_frame_comparison(
            original_pc, tokenized_pc, predicted_pc,
            roi_bounds, i+1, args.num_frames, args.fps,
            args.sequence, args.start_frame
        )
        frames.append(img)
        
        # Save individual frame
        img.save(output_dir / f'frame_{i:03d}.png')
    
    # Create GIF
    print("\nCreating GIF...")
    gif_path = output_dir / 'comparison.gif'
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=500,  # 500ms per frame = 2 fps
        loop=0
    )
    
    print(f"\n✅ GIF saved to {gif_path}")
    print(f"✅ Individual frames saved to {output_dir}")


if __name__ == "__main__":
    main()
