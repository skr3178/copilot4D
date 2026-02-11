#!/usr/bin/env python3
"""
World Model Evaluation using Open3D for point cloud metrics.

Computes:
- Chamfer Distance (Open3D built-in)
- Point-to-point distances
- Visualizations

Pipeline:
World Model → Tokens → BEV Features → Point Cloud (via raycasting) → Metrics

Usage:
    python scripts/evaluate_world_model_open3d.py \
        --checkpoint outputs/world_model_12gb/checkpoint_0050000.pt \
        --tokenizer outputs/tokenizer_memory_efficient/checkpoint_step_80000.pt \
        --sequence 00 \
        --start_frame 500 \
        --output_dir outputs/world_model_eval_open3d
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from copilot4d.world_model.world_model import CoPilot4DWorldModel, WorldModelConfig
from copilot4d.world_model.masking import cosine_mask_schedule
from copilot4d.tokenizer.tokenizer_model import CoPilot4DTokenizer
from copilot4d.utils.config import TokenizerConfig

# Open3D imports
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not available, using fallback implementations")


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


@torch.no_grad()
def tokens_to_pointcloud(tokenizer, tokens, pose, num_rays=2048, max_depth=56.0, device="cuda"):
    """
    Convert tokens to point cloud using tokenizer's built-in raycasting.
    Returns points in ego-relative coordinates (centered at vehicle).
    
    Args:
        tokens: (H, W) token indices
        pose: (4, 4) camera/ego pose (used for coordinate frame only, not translation)
        num_rays: number of rays to cast
        max_depth: maximum depth range (from tokenizer config: ray_depth_max)
    
    Returns:
        points: (N, 3) numpy array of 3D points in ego-relative coordinates
    """
    H, W = tokens.shape
    tokens_flat = tokens.reshape(H * W).long().to(device)
    
    # 1. Lookup codebook embeddings
    vq = tokenizer.vq
    z_q = vq.embed[tokens_flat]  # (N, codebook_dim)
    z_q = vq.post_proj(z_q)  # (N, encoder_dim)
    z_q = z_q.unsqueeze(0)  # (1, N, encoder_dim)
    
    # 2. Decode through tokenizer decoder
    decoder_output = tokenizer.decoder(z_q)  # (1, dec_grid^2, dec_output_dim)
    
    # 3. Generate rays from ego position
    ray_origins, ray_directions = generate_rays(num_rays, max_depth)
    
    # 4. Use tokenizer's NFG to render depths
    ray_origins_t = torch.from_numpy(ray_origins).float().unsqueeze(0).to(device)
    ray_dirs_t = torch.from_numpy(ray_directions).float().unsqueeze(0).to(device)
    
    # Call NFG forward to get depths
    pred_depths, weights, nfg = tokenizer.nfg(
        decoder_output, ray_origins_t, ray_dirs_t
    )
    
    # 5. Convert depths to 3D points
    pred_depths_np = pred_depths[0].cpu().numpy()
    
    # Filter valid depths (not at max range)
    cfg_max_depth = tokenizer.cfg.ray_depth_max
    valid_mask = pred_depths_np < cfg_max_depth * 0.95
    
    if valid_mask.sum() == 0:
        return np.zeros((0, 3))
    
    depths_valid = pred_depths_np[valid_mask]
    directions_valid = ray_directions[valid_mask]
    origins_valid = ray_origins[valid_mask]
    
    # points = origin + direction * depth (in ego frame)
    points_ego = origins_valid + directions_valid * depths_valid[:, None]
    
    # 6. Apply rotation from pose to align with world frame (but keep centered at origin)
    # This ensures consistent orientation across frames
    R = pose[:3, :3]
    points_aligned = (R.T @ points_ego.T).T  # Apply inverse rotation
    
    return points_aligned


def generate_rays(num_rays, max_depth):
    """Generate ray origins and directions from ego center."""
    # Ray origins at ego center
    origins = np.array([[0, 0, 0]] * num_rays)
    
    # Generate directions using spherical coordinates
    # For front-facing camera: azimuth in [-45°, +45°], elevation in [-15°, +15°]
    # For 360°: azimuth in [0°, 360°]
    
    # Let's do 360° horizontal, limited vertical
    phi = np.random.uniform(0, 2 * np.pi, num_rays)  # Azimuth
    theta = np.random.uniform(-0.26, 0.26, num_rays)  # Elevation (~±15°)
    
    # Convert to Cartesian
    directions = np.stack([
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        np.sin(theta)
    ], axis=1)
    
    # Normalize
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    
    return origins, directions


@torch.no_grad()
def estimate_depths_from_nfg(nfg, occupancy, ray_origins, ray_directions, depth_samples, max_depth=20.0):
    """
    Simplified depth estimation from NFG.
    
    Args:
        nfg: (F, Z, H, W) neural feature grid
        occupancy: (Z, H, W) averaged features
        ray_origins: (1, R, 3)
        ray_directions: (1, R, 3)
        depth_samples: (S,) depth values to sample
    
    Returns:
        depths: (1, R) predicted depths
    """
    device = nfg.device
    R = ray_directions.shape[1]
    S = len(depth_samples)
    
    # For each ray, sample points along the ray
    # points = origin + direction * depth
    t = depth_samples.view(1, 1, S, 1)  # (1, 1, S, 1)
    origins = ray_origins.unsqueeze(2)  # (1, R, 1, 3)
    directions = ray_directions.unsqueeze(2)  # (1, R, 1, 3)
    
    points = origins + directions * t  # (1, R, S, 3)
    
    # Map points to NFG voxel coordinates
    # Assuming NFG covers [-max_range, max_range] in x,y and [z_min, z_max] in z
    max_range = 20.0  # meters
    z_min, z_max = -4.5, 4.5
    
    # Normalize to [-1, 1] for grid_sample
    # x, y: [-max_range, max_range] -> [-1, 1]
    # z: [z_min, z_max] -> [-1, 1]
    norm_x = points[..., 0] / max_range
    norm_y = points[..., 1] / max_range
    norm_z = (points[..., 2] - z_min) / (z_max - z_min) * 2 - 1
    
    # Stack for grid_sample: (B, R, S, 3) with order (z, y, x) for 3D
    grid = torch.stack([norm_z, norm_y, norm_x], dim=-1)  # (1, R, S, 3)
    
    # Sample occupancy at these points
    # occupancy is (Z, H, W), need to add batch and channel dims
    occ_expanded = occupancy.unsqueeze(0).unsqueeze(0)  # (1, 1, Z, H, W)
    
    # grid_sample expects (N, C, D, H, W) input and (N, D_out, H_out, W_out, 3) grid
    # We want to sample at (1, R, S) points
    grid_reshaped = grid.view(1, R, S, 1, 3)  # (1, R, S, 1, 3)
    
    # Actually, for 3D grid_sample with 5D input, grid should be (N, D_out, H_out, W_out, 3)
    # Let's reshape: treat R*S as D_out, 1 as H_out, 1 as W_out
    grid_reshaped = grid.view(1, R * S, 1, 1, 3)
    
    sampled = torch.nn.functional.grid_sample(
        occ_expanded,
        grid_reshaped,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )  # (1, 1, R*S, 1, 1)
    
    sampled = sampled.view(1, R, S)
    
    # Find first occupied sample along each ray
    # occupancy > threshold indicates surface
    threshold = 0.1
    occupied = sampled > threshold
    
    # Get first occupied for each ray
    first_occupied_idx = torch.argmax(occupied.float(), dim=2)  # (1, R)
    
    # Map back to depth
    depths = depth_samples[first_occupied_idx]
    
    # If no occupancy found, set to max_depth
    no_hit = ~occupied.any(dim=2)
    depths[no_hit] = max_depth
    
    return depths


def compute_chamfer_distance_open3d(pred_points, gt_points):
    """
    Compute Chamfer Distance using Open3D.
    
    Returns:
        cd: Chamfer distance (meters)
        metrics: dict with detailed stats
    """
    if not OPEN3D_AVAILABLE:
        # Fallback to scipy
        from scipy.spatial import cKDTree
        
        tree_gt = cKDTree(gt_points)
        tree_pred = cKDTree(pred_points)
        
        # pred to GT
        dist_pred_to_gt, _ = tree_gt.query(pred_points, k=1)
        # GT to pred
        dist_gt_to_pred, _ = tree_pred.query(gt_points, k=1)
        
        cd = np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)
        
        return cd, {
            'pred_to_gt_mean': np.mean(dist_pred_to_gt),
            'pred_to_gt_std': np.std(dist_pred_to_gt),
            'gt_to_pred_mean': np.mean(dist_gt_to_pred),
            'gt_to_pred_std': np.std(dist_gt_to_pred),
        }
    
    # Open3D implementation
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(pred_points)
    
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_points)
    
    # Compute distances
    dist_pred_to_gt = np.asarray(pcd_pred.compute_point_cloud_distance(pcd_gt))
    dist_gt_to_pred = np.asarray(pcd_gt.compute_point_cloud_distance(pcd_pred))
    
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


def visualize_pointclouds(pred_points, gt_points, output_path):
    """Create point cloud visualization."""
    fig = plt.figure(figsize=(15, 5))
    
    # Predicted
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                c='red', s=1, alpha=0.5, label='Predicted')
    ax1.set_title('Predicted Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Ground truth
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], 
                c='green', s=1, alpha=0.5, label='Ground Truth')
    ax2.set_title('Ground Truth Point Cloud')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Overlay
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                c='red', s=1, alpha=0.5, label='Predicted')
    ax3.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], 
                c='green', s=1, alpha=0.5, label='Ground Truth')
    ax3.set_title('Overlay')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="World model checkpoint")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer checkpoint")
    parser.add_argument("--token_dir", default="data/kitti/tokens")
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--start_frame", type=int, default=500)
    parser.add_argument("--num_past_frames", type=int, default=2)
    parser.add_argument("--num_future_frames", type=int, default=3)
    parser.add_argument("--num_rays", type=int, default=2048, help="Rays for point cloud generation")
    parser.add_argument("--output_dir", default="outputs/world_model_eval_open3d")
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
    
    # ROI bounds based on tokenizer config
    # ray_depth_max ~= 56.57 which covers ~40m x 40m area
    # Use slightly smaller ROI for evaluation
    roi_bounds = {
        'x': (-35.0, 35.0),  # ±35m in x
        'y': (-35.0, 35.0),  # ±35m in y
        'z': (-3.0, 3.0),    # ±3m in z (typical road height)
    }
    
    # Evaluate each frame
    print("\nEvaluating point clouds...")
    all_metrics = []
    
    for i in range(args.num_future_frames):
        print(f"\nFrame {i}:")
        
        # Decode to point clouds
        print("  Decoding predicted tokens...")
        pred_pc = tokens_to_pointcloud(
            tokenizer, pred_tokens[i], future_poses[i], 
            args.num_rays, 20.0, device
        )
        
        print("  Decoding GT tokens...")
        gt_pc = tokens_to_pointcloud(
            tokenizer, gt_future_tokens[i], future_poses[i],
            args.num_rays, 20.0, device
        )
        
        print(f"  Raw points - Pred: {len(pred_pc)}, GT: {len(gt_pc)}")
        
        # Debug: show point ranges
        if len(pred_pc) > 0:
            print(f"  Pred range - X:[{pred_pc[:,0].min():.1f}, {pred_pc[:,0].max():.1f}], "
                  f"Y:[{pred_pc[:,1].min():.1f}, {pred_pc[:,1].max():.1f}], "
                  f"Z:[{pred_pc[:,2].min():.1f}, {pred_pc[:,2].max():.1f}]")
        
        # Crop to ROI
        pred_pc = crop_to_roi(pred_pc, roi_bounds)
        gt_pc = crop_to_roi(gt_pc, roi_bounds)
        
        print(f"  ROI points - Pred: {len(pred_pc)}, GT: {len(gt_pc)}")
        
        if len(pred_pc) == 0 or len(gt_pc) == 0:
            print("  Warning: Empty point cloud after cropping, skipping...")
            continue
        
        # Compute Chamfer Distance
        cd, cd_details = compute_chamfer_distance_open3d(pred_pc, gt_pc)
        
        print(f"  Chamfer Distance: {cd:.3f}m")
        print(f"    Pred→GT: {cd_details['pred_to_gt_mean']:.3f}m ± {cd_details['pred_to_gt_std']:.3f}m")
        print(f"    GT→Pred: {cd_details['gt_to_pred_mean']:.3f}m ± {cd_details['gt_to_pred_std']:.3f}m")
        
        all_metrics.append({
            'frame': i,
            'chamfer_distance': cd,
            **cd_details,
            'num_pred_points': len(pred_pc),
            'num_gt_points': len(gt_pc),
        })
        
        # Visualize
        viz_path = output_dir / f'pointcloud_frame_{i}.png'
        visualize_pointclouds(pred_pc, gt_pc, viz_path)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    if all_metrics:
        avg_cd = np.mean([m['chamfer_distance'] for m in all_metrics])
        print(f"Average Chamfer Distance: {avg_cd:.3f}m")
        print(f"\nPer-frame metrics:")
        for m in all_metrics:
            print(f"  Frame {m['frame']}: CD={m['chamfer_distance']:.3f}m, "
                  f"Points={m['num_pred_points']}/{m['num_gt_points']}")
    
    # Save metrics
    with open(output_dir / 'metrics.txt', 'w') as f:
        f.write("World Model Point Cloud Evaluation (Open3D)\n")
        f.write("="*50 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Tokenizer: {args.tokenizer}\n")
        f.write(f"Sequence: {args.sequence}, Frame: {args.start_frame}\n")
        f.write(f"Num rays: {args.num_rays}\n")
        f.write(f"ROI: {roi_bounds}\n\n")
        
        if all_metrics:
            f.write(f"Average Chamfer Distance: {avg_cd:.3f}m\n\n")
            f.write("Per-frame metrics:\n")
            for m in all_metrics:
                f.write(f"  Frame {m['frame']}:\n")
                f.write(f"    Chamfer Distance: {m['chamfer_distance']:.3f}m\n")
                f.write(f"    Pred→GT: {m['pred_to_gt_mean']:.3f}m\n")
                f.write(f"    GT→Pred: {m['gt_to_pred_mean']:.3f}m\n")
                f.write(f"    Points: {m['num_pred_points']} pred, {m['num_gt_points']} gt\n\n")
    
    print(f"\n✅ Evaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
