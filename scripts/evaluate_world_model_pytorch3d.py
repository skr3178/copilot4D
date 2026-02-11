#!/usr/bin/env python3
"""
World Model Evaluation using PyTorch3D for point cloud metrics.

Computes:
- Chamfer Distance (PyTorch3D)
- Dense spherical scanning (50,000+ rays)
- 2D BEV visualizations matching tokenizer style
- Side-by-side BEV comparisons

Pipeline:
World Model → Tokens → BEV Features → Dense Point Cloud (via spherical raycasting) → Metrics

Usage:
    python scripts/evaluate_world_model_pytorch3d.py \
        --checkpoint outputs/world_model_12gb/checkpoint_0050000.pt \
        --tokenizer outputs/tokenizer_memory_efficient/checkpoint_step_80000.pt \
        --sequence 00 \
        --start_frame 500 \
        --dense_azimuth 1800 \
        --dense_elevation 64 \
        --output_dir outputs/world_model_eval_pytorch3d
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

# PyTorch3D imports
try:
    from pytorch3d.structures import Pointclouds
    from pytorch3d.loss import chamfer_distance
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    print("Warning: PyTorch3D not available, using scipy fallback")


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


def load_sequence(token_dir: str, sequence: str, start_frame: int, num_frames: int):
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


def generate_dense_rays(
    n_azimuth: int = 1800,
    elevation_min_deg: float = -25.0,
    elevation_max_deg: float = 5.0,
    n_elevation: int = 64,
) -> tuple:
    """Generate a dense spherical grid of ray origins and directions.
    
    Returns:
        origins: (N, 3) ray origins (all zeros at ego center)
        directions: (N, 3) unit direction vectors
    """
    az = np.linspace(0, 2 * np.pi, n_azimuth, endpoint=False)
    el = np.deg2rad(np.linspace(elevation_min_deg, elevation_max_deg, n_elevation))
    az_grid, el_grid = np.meshgrid(az, el, indexing="ij")
    az_flat = az_grid.ravel()
    el_flat = el_grid.ravel()
    
    # Spherical to Cartesian
    x = np.cos(el_flat) * np.cos(az_flat)
    y = np.cos(el_flat) * np.sin(az_flat)
    z = np.sin(el_flat)
    
    directions = np.stack([x, y, z], axis=-1).astype(np.float32)
    origins = np.zeros_like(directions)
    
    return origins, directions


@torch.no_grad()
def tokens_to_pointcloud_dense(
    tokenizer, 
    tokens, 
    pose, 
    n_azimuth=1800,
    n_elevation=64,
    max_depth=56.0, 
    device="cuda",
    chunk_size=2048
):
    """
    Convert tokens to dense point cloud using spherical raycasting.
    Returns points in ego-relative coordinates.
    
    Args:
        tokens: (H, W) token indices
        pose: (4, 4) camera/ego pose
        n_azimuth: number of azimuth angles
        n_elevation: number of elevation angles
        max_depth: maximum depth range
        chunk_size: rays per chunk to control memory
    
    Returns:
        points: (N, 3) numpy array of 3D points in ego-relative coordinates
    """
    H, W = tokens.shape
    tokens_flat = tokens.reshape(H * W).long().to(device)
    
    # 1. Lookup codebook embeddings
    vq = tokenizer.vq
    z_q = vq.embed[tokens_flat]
    z_q = vq.post_proj(z_q)
    z_q = z_q.unsqueeze(0)
    
    # 2. Decode through tokenizer decoder
    decoder_output = tokenizer.decoder(z_q)
    
    # 3. Generate dense spherical rays
    ray_origins, ray_directions = generate_dense_rays(n_azimuth, -25.0, 5.0, n_elevation)
    
    # 4. Build NFG once
    nfg = tokenizer.nfg.build_nfg(decoder_output)
    
    # 5. Render depths in chunks
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
    
    # 6. Convert depths to 3D points
    pred_depths_np = pred_depths.numpy()
    cfg_max_depth = tokenizer.cfg.ray_depth_max
    valid_mask = pred_depths_np < cfg_max_depth * 0.95
    
    if valid_mask.sum() == 0:
        return np.zeros((0, 3))
    
    depths_valid = pred_depths_np[valid_mask]
    directions_valid = ray_directions[valid_mask]
    origins_valid = ray_origins[valid_mask]
    
    # points = origin + direction * depth (in ego frame)
    points_ego = origins_valid + directions_valid * depths_valid[:, None]
    
    # Apply rotation from pose to align with world frame
    R = pose[:3, :3]
    points_aligned = (R.T @ points_ego.T).T
    
    return points_aligned


def compute_chamfer_distance_pytorch3d(pred_points, gt_points):
    """
    Compute Chamfer Distance using PyTorch3D or scipy fallback.
    
    Returns:
        cd: Chamfer distance (meters)
        metrics: dict with detailed stats
    """
    # Always use scipy for detailed stats (more reliable)
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


def crop_to_roi(points, bounds):
    """Crop points to region of interest."""
    x_min, x_max = bounds['x']
    y_min, y_max = bounds['y']
    z_min, z_max = bounds['z']
    
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    return points[mask]


# ─────────────────────────────────────────────────────────────────────────────
# BEV Visualization Functions (matching tokenizer style)
# ─────────────────────────────────────────────────────────────────────────────

def plot_bev_comparison(gt_pts, pred_pts, bounds, save_path, title_suffix=""):
    """
    Side-by-side BEV (top-down X-Y) scatter plots.
    Matches tokenizer visualization style.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    x_min, x_max = bounds['x']
    y_min, y_max = bounds['y']
    z_min, z_max = bounds['z']
    
    for ax, pts, title in [
        (axes[0], gt_pts, f"Ground Truth {title_suffix}"),
        (axes[1], pred_pts, f"Predicted {title_suffix}"),
    ]:
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=0.2,
                        cmap="viridis", vmin=z_min, vmax=z_max)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(sc, cax=cax, label="Z (m)")
    
    fig.suptitle("BEV Comparison (coloured by height)", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_bev_overlay(gt_pts, pred_pts, bounds, save_path, title_suffix=""):
    """
    Overlay GT (blue) and predicted (red) in a single BEV plot.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    x_min, x_max = bounds['x']
    y_min, y_max = bounds['y']
    
    ax.scatter(gt_pts[:, 0], gt_pts[:, 1], s=0.15, c="dodgerblue", 
               alpha=0.4, label="Ground Truth")
    ax.scatter(pred_pts[:, 0], pred_pts[:, 1], s=0.15, c="red", 
               alpha=0.4, label="Predicted")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"BEV Overlay: GT (blue) vs Predicted (red) {title_suffix}")
    ax.legend(markerscale=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_side_comparison(gt_pts, pred_pts, bounds, save_path, title_suffix=""):
    """
    Side-by-side X-Z (side view) scatter plots.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    x_min, x_max = bounds['x']
    y_min, y_max = bounds['y']
    z_min, z_max = bounds['z']
    
    for ax, pts, title in [
        (axes[0], gt_pts, f"Ground Truth (side view) {title_suffix}"),
        (axes[1], pred_pts, f"Predicted (side view) {title_suffix}"),
    ]:
        sc = ax.scatter(pts[:, 0], pts[:, 2], c=pts[:, 1], s=0.2,
                        cmap="plasma", vmin=y_min, vmax=y_max)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(title)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        plt.colorbar(sc, cax=cax, label="Y (m)")
    
    fig.suptitle("Side View Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_dense_bev(points, bounds, save_path, title="Dense Reconstruction (BEV)"):
    """
    BEV plot of dense spherical-scan reconstruction.
    """
    x_min, x_max = bounds['x']
    y_min, y_max = bounds['y']
    z_min, z_max = bounds['z']
    
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    pts = points[mask]
    
    fig, ax = plt.subplots(figsize=(12, 12))
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=0.1,
                    cmap="viridis", vmin=z_min, vmax=z_max)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(sc, cax=cax, label="Z (m)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="World model checkpoint")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer checkpoint")
    parser.add_argument("--token_dir", default="data/kitti/tokens")
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--start_frame", type=int, default=500)
    parser.add_argument("--num_past_frames", type=int, default=2)
    parser.add_argument("--num_future_frames", type=int, default=3)
    parser.add_argument("--dense_azimuth", type=int, default=1800, 
                        help="Number of azimuth angles for dense scan")
    parser.add_argument("--dense_elevation", type=int, default=64,
                        help="Number of elevation angles for dense scan")
    parser.add_argument("--chunk_size", type=int, default=2048,
                        help="Ray chunk size for rendering (tune for GPU memory)")
    parser.add_argument("--output_dir", default="outputs/world_model_eval_pytorch3d")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    world_model, tokenizer, wm_cfg, tok_cfg = load_models(
        args.checkpoint, args.tokenizer, device
    )
    
    # Load data
    print("\nLoading sequence data...")
    total_frames = args.num_past_frames + args.num_future_frames
    tokens, actions, poses = load_sequence(
        args.token_dir, args.sequence, args.start_frame, total_frames
    )
    
    past_tokens = tokens[:args.num_past_frames]
    past_actions = actions[:args.num_past_frames]
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
    
    # Evaluate each frame
    print("\nEvaluating point clouds with dense spherical scanning...")
    print(f"  Dense scan: {args.dense_azimuth} x {args.dense_elevation} = "
          f"{args.dense_azimuth * args.dense_elevation:,} rays")
    
    all_metrics = []
    
    for i in range(args.num_future_frames):
        print(f"\n{'='*60}")
        print(f"Frame {i}")
        print(f"{'='*60}")
        
        # Decode to dense point clouds
        print("  Decoding predicted tokens (dense raycasting)...")
        pred_pc = tokens_to_pointcloud_dense(
            tokenizer, pred_tokens[i], future_poses[i],
            n_azimuth=args.dense_azimuth,
            n_elevation=args.dense_elevation,
            max_depth=tok_cfg.ray_depth_max,
            device=device,
            chunk_size=args.chunk_size
        )
        
        print("  Decoding GT tokens (dense raycasting)...")
        gt_pc = tokens_to_pointcloud_dense(
            tokenizer, gt_future_tokens[i], future_poses[i],
            n_azimuth=args.dense_azimuth,
            n_elevation=args.dense_elevation,
            max_depth=tok_cfg.ray_depth_max,
            device=device,
            chunk_size=args.chunk_size
        )
        
        print(f"  Raw points - Pred: {len(pred_pc):,}, GT: {len(gt_pc):,}")
        
        # Show point ranges
        if len(pred_pc) > 0:
            print(f"  Pred range - X:[{pred_pc[:,0].min():.1f}, {pred_pc[:,0].max():.1f}], "
                  f"Y:[{pred_pc[:,1].min():.1f}, {pred_pc[:,1].max():.1f}], "
                  f"Z:[{pred_pc[:,2].min():.1f}, {pred_pc[:,2].max():.1f}]")
        
        # Crop to ROI
        pred_pc_roi = crop_to_roi(pred_pc, roi_bounds)
        gt_pc_roi = crop_to_roi(gt_pc, roi_bounds)
        
        print(f"  ROI points - Pred: {len(pred_pc_roi):,}, GT: {len(gt_pc_roi):,}")
        
        if len(pred_pc_roi) == 0 or len(gt_pc_roi) == 0:
            print("  Warning: Empty point cloud after cropping, skipping...")
            continue
        
        # Compute Chamfer Distance
        cd, cd_details = compute_chamfer_distance_pytorch3d(pred_pc_roi, gt_pc_roi)
        
        print(f"  Chamfer Distance: {cd:.3f}m")
        print(f"    Pred→GT: {cd_details['pred_to_gt_mean']:.3f}m ± {cd_details['pred_to_gt_std']:.3f}m")
        print(f"    GT→Pred: {cd_details['gt_to_pred_mean']:.3f}m ± {cd_details['gt_to_pred_std']:.3f}m")
        
        all_metrics.append({
            'frame': i,
            'chamfer_distance': cd,
            **cd_details,
            'num_pred_points': len(pred_pc_roi),
            'num_gt_points': len(gt_pc_roi),
        })
        
        # Save point clouds
        npz_path = output_dir / f'pointclouds_frame_{i}.npz'
        np.savez_compressed(
            npz_path,
            pred_points=pred_pc,
            gt_points=gt_pc,
            pred_points_roi=pred_pc_roi,
            gt_points_roi=gt_pc_roi,
        )
        print(f"  Saved point clouds to {npz_path}")
        
        # Generate BEV visualizations
        print("\n  Generating BEV visualizations...")
        suffix = f"(Frame {i})"
        
        plot_bev_comparison(
            gt_pc_roi, pred_pc_roi, roi_bounds,
            output_dir / f'bev_comparison_frame_{i}.png',
            title_suffix=suffix
        )
        
        plot_bev_overlay(
            gt_pc_roi, pred_pc_roi, roi_bounds,
            output_dir / f'bev_overlay_frame_{i}.png',
            title_suffix=suffix
        )
        
        plot_side_comparison(
            gt_pc_roi, pred_pc_roi, roi_bounds,
            output_dir / f'side_comparison_frame_{i}.png',
            title_suffix=suffix
        )
        
        plot_dense_bev(
            pred_pc, roi_bounds,
            output_dir / f'dense_pred_frame_{i}.png',
            title=f"Dense Predicted Reconstruction (Frame {i})"
        )
        
        plot_dense_bev(
            gt_pc, roi_bounds,
            output_dir / f'dense_gt_frame_{i}.png',
            title=f"Dense GT Reconstruction (Frame {i})"
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
            print(f"  Frame {m['frame']}: CD={m['chamfer_distance']:.3f}m, "
                  f"Points={m['num_pred_points']:,}/{m['num_gt_points']:,}")
    
    # Save metrics
    with open(output_dir / 'metrics.txt', 'w') as f:
        f.write("World Model Point Cloud Evaluation (PyTorch3D)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Tokenizer: {args.tokenizer}\n")
        f.write(f"Sequence: {args.sequence}, Frame: {args.start_frame}\n")
        f.write(f"Dense scan: {args.dense_azimuth} x {args.dense_elevation} rays\n")
        f.write(f"ROI: {roi_bounds}\n\n")
        
        if all_metrics:
            f.write(f"Average Chamfer Distance: {avg_cd:.3f}m\n\n")
            f.write("Per-frame metrics:\n")
            for m in all_metrics:
                f.write(f"  Frame {m['frame']}:\n")
                f.write(f"    Chamfer Distance: {m['chamfer_distance']:.3f}m\n")
                f.write(f"    Pred→GT: {m['pred_to_gt_mean']:.3f}m\n")
                f.write(f"    GT→Pred: {m['gt_to_pred_mean']:.3f}m\n")
                f.write(f"    Points: {m['num_pred_points']:,} pred, {m['num_gt_points']:,} gt\n\n")
    
    print(f"\n✅ Evaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
