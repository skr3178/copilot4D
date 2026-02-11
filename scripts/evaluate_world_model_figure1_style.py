#!/usr/bin/env python3
"""
World Model Evaluation - Figure 1 Style Visualization

Generates dense BEV visualizations matching the paper's Figure 1 style:
- Current Observation (input point cloud)
- Future GT (ground truth future point cloud)
- World Model Prediction (decoded from predicted tokens)
- Prediction vs GT overlay

This uses the original LiDAR point clouds (not raycast reconstruction) for GT,
and dense raycasting through the tokenizer for predictions.

Usage:
    python scripts/evaluate_world_model_figure1_style.py \
        --checkpoint outputs/world_model_12gb/checkpoint_0050000.pt \
        --tokenizer outputs/tokenizer_memory_efficient/checkpoint_step_80000.pt \
        --sequence 00 \
        --start_frame 500 \
        --output_dir outputs/world_model_figure1_style
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from copilot4d.world_model.world_model import CoPilot4DWorldModel, WorldModelConfig
from copilot4d.world_model.masking import cosine_mask_schedule
from copilot4d.tokenizer.tokenizer_model import CoPilot4DTokenizer
from copilot4d.utils.config import TokenizerConfig
from copilot4d.data.point_cloud_utils import filter_roi


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


def load_sequence_data(token_dir: str, sequence: str, start_frame: int, num_frames: int):
    """Load tokenized frames and poses."""
    seq_path = Path(token_dir) / sequence
    
    # Load tokens
    tokens_list = []
    for i in range(start_frame, start_frame + num_frames):
        token_file = seq_path / f"{i:06d}.pt"
        tokens = torch.load(token_file, map_location="cpu")
        tokens_list.append(tokens)
    tokens = torch.stack(tokens_list)
    
    # Load poses
    poses_file = seq_path / "poses.pkl"
    if poses_file.exists():
        with open(poses_file, 'rb') as f:
            all_poses = pickle.load(f)
        poses = [all_poses[i] for i in range(start_frame, start_frame + num_frames)]
    else:
        poses = [np.eye(4) for _ in range(num_frames)]
    
    # Compute actions
    actions = []
    for i in range(1, len(poses)):
        T_rel = np.linalg.inv(poses[i-1]) @ poses[i]
        actions.append(T_rel.flatten())
    actions.append(np.eye(4).flatten())
    actions = np.stack(actions)
    
    return tokens, torch.from_numpy(actions).float(), poses


def load_original_pointcloud(kitti_root: str, sequence: str, frame_idx: int, cfg):
    """Load original LiDAR point cloud from KITTI."""
    try:
        from pykitti import odometry
        dataset = odometry(kitti_root, sequence)
        points = dataset.get_velo(frame_idx)  # (N, 4) - x, y, z, reflectance
        
        # Filter to ROI
        points_filtered = filter_roi(points, cfg)
        return points_filtered[:, :3]  # Return only xyz
    except Exception as e:
        print(f"Warning: Could not load original point cloud: {e}")
        return None


@torch.no_grad()
def predict_future_tokens(world_model, past_tokens, past_actions, num_future, cfg, device):
    """Predict future tokens using greedy decoding."""
    T_past, H, W = past_tokens.shape
    N = H * W
    mask_id = cfg.codebook_size
    
    future_tokens = torch.full((num_future, H, W), mask_id, dtype=torch.long, device=device)
    
    last_action = past_actions[-1:] if len(past_actions) > 0 else torch.zeros(1, 16)
    future_actions = last_action.repeat(num_future, 1).to(device)
    
    full_tokens = torch.cat([past_tokens.to(device), future_tokens], dim=0)
    full_actions = torch.cat([past_actions.to(device), future_actions], dim=0)
    
    # Handle sequence length limit
    if full_tokens.shape[0] > cfg.num_frames:
        keep = cfg.num_frames - num_future
        full_tokens = torch.cat([
            full_tokens[-cfg.num_frames:-num_future], 
            full_tokens[-num_future:]
        ], dim=0)
        full_actions = torch.cat([
            full_actions[-cfg.num_frames:-num_future], 
            full_actions[-num_future:]
        ], dim=0)
        T_past = keep
    
    T_total = full_tokens.shape[0]
    temporal_mask = torch.triu(torch.ones(T_total, T_total) * float('-inf'), diagonal=1).to(device)
    
    num_steps = cfg.num_sampling_steps
    
    for step in range(num_steps):
        logits = world_model(full_tokens.unsqueeze(0), full_actions.unsqueeze(0), temporal_mask)
        logits_future = logits[0, T_past:]
        
        predictions = logits_future.argmax(dim=-1)
        future_flat = future_tokens.reshape(num_future, N)
        mask = (future_flat == mask_id)
        future_flat = torch.where(mask, predictions, future_flat)
        future_tokens = future_flat.reshape(num_future, H, W)
        full_tokens = torch.cat([full_tokens[:T_past], future_tokens], dim=0)
        
        # Remask for next iteration
        if step < num_steps - 1:
            t = (step + 1) / num_steps
            mask_ratio = cosine_mask_schedule(torch.tensor(t)).item()
            num_to_mask = int(mask_ratio * N * num_future)
            
            if num_to_mask > 0:
                confidences = logits_future.max(dim=-1).values
                flat_conf = confidences.reshape(-1)
                _, indices = torch.topk(flat_conf, k=num_to_mask, largest=False)
                future_flat = future_tokens.reshape(-1)
                future_flat[indices] = mask_id
                future_tokens = future_flat.reshape(num_future, H, W)
                full_tokens = torch.cat([full_tokens[:T_past], future_tokens], dim=0)
    
    return future_tokens.cpu()


def generate_dense_rays(n_azimuth=1800, elevation_min_deg=-25.0, 
                        elevation_max_deg=5.0, n_elevation=64):
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
def tokens_to_dense_pointcloud(tokenizer, tokens, pose, n_azimuth=1800, n_elevation=64,
                                device="cuda", chunk_size=2048):
    """Convert tokens to dense point cloud via spherical raycasting."""
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
    
    points = directions_valid * depths_valid[:, None]
    
    # Apply pose rotation
    R = pose[:3, :3]
    points_aligned = (R.T @ points.T).T
    
    return points_aligned


def transform_points_to_ego(points, pose):
    """
    Transform points from world coordinates to ego-relative coordinates.
    Note: KITTI LiDAR points from pykitti are already in ego frame,
    so this function is kept for compatibility but shouldn't be used
    for original LiDAR data.
    """
    R = pose[:3, :3]
    t = pose[:3, 3]
    # World to ego: R^T @ (p - t)
    points_centered = points - t
    points_ego = (R.T @ points_centered.T).T
    return points_ego


def plot_figure1_style(current_obs, future_gt, prediction, roi_bounds, save_path, frame_idx=0):
    """Create Figure 1 style visualization with 4 panels."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    x_min, x_max = roi_bounds['x']
    y_min, y_max = roi_bounds['y']
    z_min, z_max = roi_bounds['z']
    
    # Filter to ROI
    def filter_roi_points(pts, bounds):
        mask = (
            (pts[:, 0] >= bounds['x'][0]) & (pts[:, 0] <= bounds['x'][1]) &
            (pts[:, 1] >= bounds['y'][0]) & (pts[:, 1] <= bounds['y'][1]) &
            (pts[:, 2] >= bounds['z'][0]) & (pts[:, 2] <= bounds['z'][1])
        )
        return pts[mask]
    
    current_obs_roi = filter_roi_points(current_obs, roi_bounds)
    future_gt_roi = filter_roi_points(future_gt, roi_bounds)
    pred_roi = filter_roi_points(prediction, roi_bounds)
    
    # Panel 1: Current Observation
    ax = axes[0]
    sc = ax.scatter(current_obs_roi[:, 0], current_obs_roi[:, 1], 
                    c=current_obs_roi[:, 2], s=0.1, cmap="viridis",
                    vmin=z_min, vmax=z_max, alpha=0.8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Current Observation\n({len(current_obs_roi):,} points)")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(sc, cax=cax, label="Z (m)")
    
    # Panel 2: Future GT
    ax = axes[1]
    sc = ax.scatter(future_gt_roi[:, 0], future_gt_roi[:, 1],
                    c=future_gt_roi[:, 2], s=0.1, cmap="viridis",
                    vmin=z_min, vmax=z_max, alpha=0.8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Future GT\n({len(future_gt_roi):,} points)")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(sc, cax=cax, label="Z (m)")
    
    # Panel 3: World Model Prediction
    ax = axes[2]
    sc = ax.scatter(pred_roi[:, 0], pred_roi[:, 1],
                    c=pred_roi[:, 2], s=0.1, cmap="viridis",
                    vmin=z_min, vmax=z_max, alpha=0.8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"World Model Prediction\n({len(pred_roi):,} points)")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(sc, cax=cax, label="Z (m)")
    
    # Panel 4: Prediction vs GT overlay
    ax = axes[3]
    ax.scatter(future_gt_roi[:, 0], future_gt_roi[:, 1], 
               s=0.15, c="dodgerblue", alpha=0.5, label="Ground Truth")
    ax.scatter(pred_roi[:, 0], pred_roi[:, 1],
               s=0.15, c="red", alpha=0.5, label="Prediction")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Prediction vs GT")
    ax.legend(markerscale=10, loc="upper right")
    
    fig.suptitle(f"Frame +{frame_idx} Prediction", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def compute_chamfer_distance(pred_points, gt_points):
    """Compute Chamfer Distance using scipy."""
    from scipy.spatial import cKDTree
    
    tree_gt = cKDTree(gt_points)
    tree_pred = cKDTree(pred_points)
    
    dist_pred_to_gt, _ = tree_gt.query(pred_points, k=1)
    dist_gt_to_pred, _ = tree_pred.query(gt_points, k=1)
    
    cd = np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)
    
    return cd, {
        'pred_to_gt_mean': np.mean(dist_pred_to_gt),
        'pred_to_gt_std': np.std(dist_pred_to_gt),
        'gt_to_pred_mean': np.mean(dist_gt_to_pred),
        'gt_to_pred_std': np.std(dist_gt_to_pred),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="World model checkpoint")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer checkpoint")
    parser.add_argument("--token_dir", default="data/kitti/tokens")
    parser.add_argument("--kitti_root", default="data/kitti")
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--start_frame", type=int, default=500)
    parser.add_argument("--num_past_frames", type=int, default=2)
    parser.add_argument("--num_future_frames", type=int, default=3)
    parser.add_argument("--dense_azimuth", type=int, default=1800)
    parser.add_argument("--dense_elevation", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--output_dir", default="outputs/world_model_figure1_style")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    world_model, tokenizer, wm_cfg, tok_cfg = load_models(
        args.checkpoint, args.tokenizer, device
    )
    
    # Load sequence data
    print("\nLoading sequence data...")
    total_frames = args.num_past_frames + args.num_future_frames
    tokens, actions, poses = load_sequence_data(
        args.token_dir, args.sequence, args.start_frame, total_frames
    )
    
    past_tokens = tokens[:args.num_past_frames]
    past_actions = actions[:args.num_past_frames]
    past_poses = poses[:args.num_past_frames]
    gt_future_tokens = tokens[args.num_past_frames:]
    future_poses = poses[args.num_past_frames:]
    
    print(f"  Past frames: {past_tokens.shape}")
    print(f"  Future frames: {gt_future_tokens.shape}")
    
    # Predict future
    print(f"\nPredicting {args.num_future_frames} future frames...")
    pred_tokens = predict_future_tokens(
        world_model, past_tokens, past_actions,
        args.num_future_frames, wm_cfg, device
    )
    
    # ROI bounds
    roi_bounds = {
        'x': (-35.0, 35.0),
        'y': (-35.0, 35.0),
        'z': (-3.0, 3.0),
    }
    
    # Get the last observed frame's point cloud
    print("\nLoading original LiDAR point clouds...")
    last_obs_frame = args.start_frame + args.num_past_frames - 1
    current_obs_pc = load_original_pointcloud(
        args.kitti_root, args.sequence, last_obs_frame, tok_cfg
    )
    
    if current_obs_pc is not None:
        # KITTI points are already in ego coordinates
        current_obs_pc_ego = current_obs_pc
        print(f"  Current observation: {len(current_obs_pc_ego)} points")
    else:
        current_obs_pc_ego = None
        print("  Warning: Could not load current observation")
    
    # Evaluate each frame
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    print(f"Dense scan: {args.dense_azimuth} x {args.dense_elevation} = "
          f"{args.dense_azimuth * args.dense_elevation:,} rays")
    
    all_metrics = []
    
    for i in range(args.num_future_frames):
        print(f"\nFrame +{i+1} (absolute: {args.start_frame + args.num_past_frames + i}):")
        
        # Load original future GT point cloud
        future_frame_idx = args.start_frame + args.num_past_frames + i
        future_gt_pc = load_original_pointcloud(
            args.kitti_root, args.sequence, future_frame_idx, tok_cfg
        )
        
        if future_gt_pc is not None:
            # KITTI points are already in ego coordinates
            future_gt_pc_ego = future_gt_pc
            print(f"  Original GT points: {len(future_gt_pc_ego)}")
        else:
            print(f"  Warning: Could not load GT point cloud, using raycast fallback")
            future_gt_pc_ego = None
        
        # Generate prediction via dense raycasting
        print("  Decoding predicted tokens (dense raycasting)...")
        pred_pc = tokens_to_dense_pointcloud(
            tokenizer, pred_tokens[i], future_poses[i],
            n_azimuth=args.dense_azimuth,
            n_elevation=args.dense_elevation,
            device=device,
            chunk_size=args.chunk_size
        )
        print(f"  Predicted points: {len(pred_pc)}")
        
        # Fallback to raycast GT if original not available
        if future_gt_pc_ego is None:
            print("  Decoding GT tokens (dense raycasting)...")
            future_gt_pc_ego = tokens_to_dense_pointcloud(
                tokenizer, gt_future_tokens[i], future_poses[i],
                n_azimuth=args.dense_azimuth,
                n_elevation=args.dense_elevation,
                device=device,
                chunk_size=args.chunk_size
            )
            print(f"  Raycast GT points: {len(future_gt_pc_ego)}")
        
        # Filter to ROI for metrics
        def filter_roi_points(pts, bounds):
            mask = (
                (pts[:, 0] >= bounds['x'][0]) & (pts[:, 0] <= bounds['x'][1]) &
                (pts[:, 1] >= bounds['y'][0]) & (pts[:, 1] <= bounds['y'][1]) &
                (pts[:, 2] >= bounds['z'][0]) & (pts[:, 2] <= bounds['z'][1])
            )
            return pts[mask]
        
        pred_roi = filter_roi_points(pred_pc, roi_bounds)
        gt_roi = filter_roi_points(future_gt_pc_ego, roi_bounds)
        
        print(f"  ROI points - Pred: {len(pred_roi):,}, GT: {len(gt_roi):,}")
        
        if len(pred_roi) > 0 and len(gt_roi) > 0:
            cd, cd_details = compute_chamfer_distance(pred_roi, gt_roi)
            print(f"  Chamfer Distance: {cd:.3f}m")
            
            all_metrics.append({
                'frame': i,
                'chamfer_distance': cd,
                **cd_details,
                'num_pred_points': len(pred_roi),
                'num_gt_points': len(gt_roi),
            })
        
        # Generate Figure 1 style visualization
        if current_obs_pc_ego is not None:
            # Use current obs for first panel
            obs_for_viz = current_obs_pc_ego
        else:
            # Fallback: use last past token's raycast reconstruction
            obs_for_viz = tokens_to_dense_pointcloud(
                tokenizer, past_tokens[-1], past_poses[-1],
                n_azimuth=args.dense_azimuth // 2,  # Lower res for speed
                n_elevation=args.dense_elevation // 2,
                device=device, chunk_size=args.chunk_size
            )
        
        plot_figure1_style(
            obs_for_viz,
            future_gt_pc_ego,
            pred_pc,
            roi_bounds,
            output_dir / f'figure1_frame_{i}.png',
            frame_idx=i+1
        )
        
        # Save point clouds
        npz_path = output_dir / f'pointclouds_frame_{i}.npz'
        np.savez_compressed(
            npz_path,
            current_obs=obs_for_viz,
            future_gt=future_gt_pc_ego,
            prediction=pred_pc,
        )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if all_metrics:
        avg_cd = np.mean([m['chamfer_distance'] for m in all_metrics])
        print(f"Average Chamfer Distance: {avg_cd:.3f}m")
        print(f"\nPer-frame metrics:")
        for m in all_metrics:
            print(f"  Frame +{m['frame']+1}: CD={m['chamfer_distance']:.3f}m, "
                  f"Points={m['num_pred_points']:,}/{m['num_gt_points']:,}")
    
    # Save metrics
    with open(output_dir / 'metrics.txt', 'w') as f:
        f.write("World Model Evaluation - Figure 1 Style\n")
        f.write("="*60 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Tokenizer: {args.tokenizer}\n")
        f.write(f"Sequence: {args.sequence}, Start Frame: {args.start_frame}\n")
        f.write(f"Dense scan: {args.dense_azimuth} x {args.dense_elevation} rays\n")
        f.write(f"ROI: {roi_bounds}\n\n")
        
        if all_metrics:
            f.write(f"Average Chamfer Distance: {avg_cd:.3f}m\n\n")
            f.write("Per-frame metrics:\n")
            for m in all_metrics:
                f.write(f"  Frame +{m['frame']+1}:\n")
                f.write(f"    Chamfer Distance: {m['chamfer_distance']:.3f}m\n")
                f.write(f"    Pred→GT: {m['pred_to_gt_mean']:.3f}m\n")
                f.write(f"    GT→Pred: {m['gt_to_pred_mean']:.3f}m\n")
                f.write(f"    Points: {m['num_pred_points']:,} pred, {m['num_gt_points']:,} gt\n\n")
    
    print(f"\n✅ Evaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
