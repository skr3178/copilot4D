"""Reconstruct LiDAR point clouds from a trained CoPilot4D tokenizer checkpoint.

Loads a KITTI sample, encodes it to discrete tokens, decodes back to predicted
depths via volume rendering, and compares reconstructed 3D points with ground truth.

Generates:
  1. BEV (Bird's Eye View) comparison of GT vs reconstructed points
  2. Side (X-Z) view comparison
  3. Predicted vs GT depth scatter plot
  4. Depth error histogram
  5. Per-point error coloured BEV map
  6. Dense spherical reconstruction (full scene view)

Usage:
    python scripts/reconstruct_lidar.py \
        --checkpoint outputs/tokenizer_debug/checkpoint_step_6000.pt \
        --config configs/tokenizer_debug.yaml \
        --sample_idx 0 --split val \
        --max_rays 0  \
        --output_dir outputs/reconstruction
"""

import os
import sys
import argparse
import yaml
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── project root ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from copilot4d.utils.config import TokenizerConfig
from copilot4d.tokenizer.tokenizer_model import CoPilot4DTokenizer
from copilot4d.data.kitti_dataset import KITTITokenizerDataset, tokenizer_collate_fn
from copilot4d.data.point_cloud_utils import filter_roi, voxelize_points, generate_rays


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> TokenizerConfig:
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return TokenizerConfig(**cfg_dict)


def depths_to_points(origins: torch.Tensor, directions: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
    """Convert ray origins + directions + depths to 3D xyz points.

    Args:
        origins:    (R, 3)
        directions: (R, 3)  unit vectors
        depths:     (R,)

    Returns:
        points: (R, 3)
    """
    return origins + directions * depths.unsqueeze(-1)


def generate_dense_rays(
    n_azimuth: int = 1800,
    elevation_min_deg: float = -25.0,
    elevation_max_deg: float = 5.0,
    n_elevation: int = 60,
) -> np.ndarray:
    """Generate a dense spherical grid of unit direction vectors.

    Returns:
        directions: (n_azimuth * n_elevation, 3) float32
    """
    az = np.linspace(0, 2 * np.pi, n_azimuth, endpoint=False)
    el = np.deg2rad(np.linspace(elevation_min_deg, elevation_max_deg, n_elevation))
    az_grid, el_grid = np.meshgrid(az, el, indexing="ij")  # (A, E)
    az_flat = az_grid.ravel()
    el_flat = el_grid.ravel()
    x = np.cos(el_flat) * np.cos(az_flat)
    y = np.cos(el_flat) * np.sin(az_flat)
    z = np.sin(el_flat)
    dirs = np.stack([x, y, z], axis=-1).astype(np.float32)
    return dirs


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction core
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_sample(model, batch, device, cfg):
    """Run encoder on a batch and return quantized tokens + indices."""
    features = batch["features"].to(device)
    num_points = batch["num_points"].to(device)
    coords = batch["coords"].to(device)
    batch_size = batch["batch_size"]

    with autocast(enabled=cfg.amp):
        bev = model.encode_voxels(features, num_points, coords, batch_size)
        encoder_out = model.encoder(bev)
        quantized, indices, _, vq_metrics = model.vq(encoder_out)

    token_grid = cfg.token_grid_size
    indices = indices.view(batch_size, token_grid, token_grid)
    return quantized, indices, vq_metrics


@torch.no_grad()
def render_rays(model, quantized, ray_origins, ray_directions, cfg, device, chunk_size=512):
    """Render predicted depths for arbitrary rays using the decoded NFG.

    Args:
        model: CoPilot4DTokenizer in eval mode
        quantized: (1, N_tokens, vq_dim)
        ray_origins: (1, R, 3) tensor
        ray_directions: (1, R, 3) tensor
        cfg: TokenizerConfig
        device: torch.device
        chunk_size: rays per chunk (to control memory)

    Returns:
        pred_depths: (R,) tensor on cpu
    """
    with autocast(enabled=cfg.amp):
        decoder_output = model.decoder(quantized)
        nfg = model.nfg.build_nfg(decoder_output)

    R = ray_origins.shape[1]
    all_depths = []

    for start in range(0, R, chunk_size):
        end = min(start + chunk_size, R)
        ro = ray_origins[:, start:end].to(device)
        rd = ray_directions[:, start:end].to(device)
        with autocast(enabled=cfg.amp):
            d, _ = model.nfg.query_rays(nfg, ro, rd, cfg.ray_depth_min, cfg.ray_depth_max)
        all_depths.append(d.float().cpu())

    pred_depths = torch.cat(all_depths, dim=1).squeeze(0)  # (R,)
    return pred_depths


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred_depths, gt_depths):
    """Compute depth reconstruction metrics. Inputs are 1-D tensors."""
    err = (pred_depths - gt_depths).abs()
    sq_err = (pred_depths - gt_depths) ** 2
    rel_err = err / gt_depths.clamp(min=1.0)
    metrics = {
        "MAE (m)": err.mean().item(),
        "RMSE (m)": sq_err.mean().sqrt().item(),
        "Median AE (m)": err.median().item(),
        "MeanRelErr": rel_err.mean().item(),
        "< 1m (%)": (err < 1.0).float().mean().item() * 100,
        "< 2m (%)": (err < 2.0).float().mean().item() * 100,
        "< 5m (%)": (err < 5.0).float().mean().item() * 100,
    }
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_bev_comparison(gt_pts, pred_pts, cfg, save_path):
    """Side-by-side BEV (top-down X-Y) scatter plots."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, pts, title in [
        (axes[0], gt_pts, "Ground Truth"),
        (axes[1], pred_pts, "Reconstructed"),
    ]:
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=0.2,
                        cmap="viridis", vmin=cfg.z_min, vmax=cfg.z_max)
        ax.set_xlim(cfg.x_min, cfg.x_max)
        ax.set_ylim(cfg.y_min, cfg.y_max)
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


def plot_side_comparison(gt_pts, pred_pts, cfg, save_path):
    """Side-by-side X-Z (side view) scatter plots."""
    fig, axes = plt.subplots(2, 1, figsize=(18, 8))

    for ax, pts, title in [
        (axes[0], gt_pts, "Ground Truth (side view)"),
        (axes[1], pred_pts, "Reconstructed (side view)"),
    ]:
        sc = ax.scatter(pts[:, 0], pts[:, 2], c=pts[:, 1], s=0.2,
                        cmap="plasma", vmin=cfg.y_min, vmax=cfg.y_max)
        ax.set_xlim(cfg.x_min, cfg.x_max)
        ax.set_ylim(cfg.z_min, cfg.z_max)
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


def plot_depth_scatter(gt_depths, pred_depths, save_path):
    """Scatter plot of predicted vs ground-truth depths."""
    fig, ax = plt.subplots(figsize=(8, 8))
    gt_np = gt_depths.numpy()
    pred_np = pred_depths.numpy()
    ax.scatter(gt_np, pred_np, s=0.5, alpha=0.3, c="steelblue")
    lims = [min(gt_np.min(), pred_np.min()) - 1, max(gt_np.max(), pred_np.max()) + 1]
    ax.plot(lims, lims, "r--", lw=1, label="perfect")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("GT depth (m)")
    ax.set_ylabel("Predicted depth (m)")
    ax.set_title("Depth: Predicted vs Ground Truth")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_error_histogram(gt_depths, pred_depths, save_path):
    """Histogram of absolute depth errors."""
    err = (pred_depths - gt_depths).abs().numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(err, bins=100, range=(0, min(err.max(), 30)), color="salmon",
            edgecolor="darkred", alpha=0.8)
    ax.axvline(np.median(err), color="blue", ls="--", lw=1.5, label=f"Median = {np.median(err):.2f} m")
    ax.axvline(np.mean(err), color="green", ls="--", lw=1.5, label=f"Mean = {np.mean(err):.2f} m")
    ax.set_xlabel("Absolute depth error (m)")
    ax.set_ylabel("Count")
    ax.set_title("Depth Error Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_bev_error_map(gt_pts, errors, cfg, save_path):
    """BEV scatter plot coloured by per-point depth error."""
    fig, ax = plt.subplots(figsize=(10, 10))
    norm = Normalize(vmin=0, vmax=min(errors.max(), 10))
    sc = ax.scatter(gt_pts[:, 0], gt_pts[:, 1], c=errors, s=0.4,
                    cmap="hot_r", norm=norm)
    ax.set_xlim(cfg.x_min, cfg.x_max)
    ax.set_ylim(cfg.y_min, cfg.y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("BEV Error Map (clipped at 10 m)")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(sc, cax=cax, label="Depth error (m)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_dense_reconstruction(dense_pts, cfg, save_path):
    """BEV plot of a dense spherical-scan reconstruction."""
    mask = (
        (dense_pts[:, 0] >= cfg.x_min) & (dense_pts[:, 0] <= cfg.x_max) &
        (dense_pts[:, 1] >= cfg.y_min) & (dense_pts[:, 1] <= cfg.y_max) &
        (dense_pts[:, 2] >= cfg.z_min) & (dense_pts[:, 2] <= cfg.z_max)
    )
    pts = dense_pts[mask]
    fig, ax = plt.subplots(figsize=(10, 10))
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=0.1,
                    cmap="viridis", vmin=cfg.z_min, vmax=cfg.z_max)
    ax.set_xlim(cfg.x_min, cfg.x_max)
    ax.set_ylim(cfg.y_min, cfg.y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Dense Reconstruction (spherical scan, BEV)")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(sc, cax=cax, label="Z (m)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_combined_overlay(gt_pts, pred_pts, cfg, save_path):
    """Overlay GT (blue) and reconstructed (red) in a single BEV plot."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(gt_pts[:, 0], gt_pts[:, 1], s=0.15, c="dodgerblue", alpha=0.4, label="Ground Truth")
    ax.scatter(pred_pts[:, 0], pred_pts[:, 1], s=0.15, c="red", alpha=0.4, label="Reconstructed")
    ax.set_xlim(cfg.x_min, cfg.x_max)
    ax.set_ylim(cfg.y_min, cfg.y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("BEV Overlay: GT (blue) vs Reconstructed (red)")
    ax.legend(markerscale=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Reconstruct LiDAR from CoPilot4D checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Index into the dataset split")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--max_rays", type=int, default=0,
                        help="Max rays for matched reconstruction (0 = all points)")
    parser.add_argument("--dense", action="store_true",
                        help="Also render a dense spherical scan")
    parser.add_argument("--dense_azimuth", type=int, default=1800)
    parser.add_argument("--dense_elevation", type=int, default=60)
    parser.add_argument("--output_dir", type=str, default="outputs/reconstruction")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Ray chunk size for rendering (tune for GPU memory)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load config ──────────────────────────────────────────────────────
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Config loaded from {args.config}")
    print(f"  ROI: x=[{cfg.x_min}, {cfg.x_max}], y=[{cfg.y_min}, {cfg.y_max}], z=[{cfg.z_min}, {cfg.z_max}]")
    print(f"  Voxel grid: {cfg.voxel_grid_xy}x{cfg.voxel_grid_xy}x{cfg.voxel_grid_z}")
    print(f"  Token grid: {cfg.token_grid_size}x{cfg.token_grid_size}")
    print(f"  Depth range: [{cfg.ray_depth_min:.1f}, {cfg.ray_depth_max:.1f}] m")
    print(f"  Device: {device}")

    # ── Load model ───────────────────────────────────────────────────────
    print("\nLoading model...")
    model = CoPilot4DTokenizer(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    step = ckpt.get("step", "?")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded checkpoint step {step}  ({n_params:.2f}M params)")

    # ── Load dataset sample ──────────────────────────────────────────────
    print(f"\nLoading KITTI {args.split} sample {args.sample_idx}...")
    dataset = KITTITokenizerDataset(cfg, split=args.split)
    print(f"  Dataset size: {len(dataset)}")

    # Get the raw point cloud directly for full reconstruction
    seq, frame_idx = dataset.samples[args.sample_idx]
    raw_points = dataset.datasets[seq].get_velo(frame_idx)  # (N, 4)
    print(f"  Raw point cloud: {raw_points.shape[0]} points")

    # Filter to ROI
    points_roi = filter_roi(raw_points, cfg)
    N_roi = points_roi.shape[0]
    print(f"  After ROI filter: {N_roi} points")

    # ── Prepare voxelised input (via dataset + collate) ──────────────────
    sample = dataset[args.sample_idx]
    batch = tokenizer_collate_fn([sample])

    # ── Encode to tokens ─────────────────────────────────────────────────
    print("\nEncoding to tokens...")
    quantized, indices, vq_metrics = encode_sample(model, batch, device, cfg)
    active = vq_metrics.get("vq_active_codes", "?")
    perp = vq_metrics.get("vq_perplexity", "?")
    print(f"  Token grid: {indices.shape}")
    print(f"  Active codes: {active},  Perplexity: {perp}")

    # ── Matched reconstruction (rays from all GT points) ─────────────────
    print("\n── Matched Reconstruction (1-to-1 with GT points) ──")
    all_ray_data = generate_rays(points_roi, cfg)
    all_origins = all_ray_data["ray_origins"]      # (N_roi, 3) all zeros
    all_directions = all_ray_data["ray_directions"]  # (N_roi, 3)
    all_gt_depths = all_ray_data["ray_depths"]       # (N_roi,)

    # Filter to valid depth range
    valid = (all_gt_depths >= cfg.ray_depth_min) & (all_gt_depths <= cfg.ray_depth_max)
    all_origins = all_origins[valid]
    all_directions = all_directions[valid]
    all_gt_depths = all_gt_depths[valid]
    N_valid = all_gt_depths.shape[0]
    print(f"  Valid rays (depth in [{cfg.ray_depth_min:.1f}, {cfg.ray_depth_max:.1f}]): {N_valid}")

    # Optionally cap the number of rays
    if args.max_rays > 0 and N_valid > args.max_rays:
        idx = np.random.choice(N_valid, args.max_rays, replace=False)
        all_origins = all_origins[idx]
        all_directions = all_directions[idx]
        all_gt_depths = all_gt_depths[idx]
        N_valid = args.max_rays
        print(f"  Sub-sampled to {N_valid} rays")

    # Convert to tensors
    origins_t = torch.from_numpy(all_origins).unsqueeze(0).to(device)     # (1, R, 3)
    dirs_t = torch.from_numpy(all_directions).unsqueeze(0).to(device)     # (1, R, 3)
    gt_depths_t = torch.from_numpy(all_gt_depths)                         # (R,)

    print(f"  Rendering {N_valid} rays (chunk={args.chunk_size})...")
    pred_depths_t = render_rays(model, quantized, origins_t, dirs_t, cfg, device,
                                chunk_size=args.chunk_size)

    # ── Metrics ──────────────────────────────────────────────────────────
    metrics = compute_metrics(pred_depths_t, gt_depths_t)
    print("\n── Depth Reconstruction Metrics ──")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")

    # ── Convert to 3D points ─────────────────────────────────────────────
    origins_cpu = torch.from_numpy(all_origins)
    dirs_cpu = torch.from_numpy(all_directions)
    gt_pts = depths_to_points(origins_cpu, dirs_cpu, gt_depths_t).numpy()
    pred_pts = depths_to_points(origins_cpu, dirs_cpu, pred_depths_t).numpy()
    errors_np = (pred_depths_t - gt_depths_t).abs().numpy()

    # ── Save point clouds as .npz ────────────────────────────────────────
    npz_path = os.path.join(args.output_dir, "reconstruction.npz")
    np.savez_compressed(
        npz_path,
        gt_points=gt_pts,
        pred_points=pred_pts,
        gt_depths=gt_depths_t.numpy(),
        pred_depths=pred_depths_t.numpy(),
        directions=all_directions,
        errors=errors_np,
    )
    print(f"\n  Saved point clouds to {npz_path}")

    # ── Plots ────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_bev_comparison(
        gt_pts, pred_pts, cfg,
        os.path.join(args.output_dir, "bev_comparison.png"),
    )
    plot_side_comparison(
        gt_pts, pred_pts, cfg,
        os.path.join(args.output_dir, "side_comparison.png"),
    )
    plot_depth_scatter(
        gt_depths_t, pred_depths_t,
        os.path.join(args.output_dir, "depth_scatter.png"),
    )
    plot_error_histogram(
        gt_depths_t, pred_depths_t,
        os.path.join(args.output_dir, "error_histogram.png"),
    )
    plot_bev_error_map(
        gt_pts, errors_np, cfg,
        os.path.join(args.output_dir, "bev_error_map.png"),
    )
    plot_combined_overlay(
        gt_pts, pred_pts, cfg,
        os.path.join(args.output_dir, "bev_overlay.png"),
    )

    # ── Dense spherical reconstruction ───────────────────────────────────
    if args.dense:
        print(f"\n── Dense Spherical Reconstruction ──")
        dense_dirs = generate_dense_rays(
            n_azimuth=args.dense_azimuth,
            n_elevation=args.dense_elevation,
        )
        N_dense = dense_dirs.shape[0]
        dense_origins = np.zeros_like(dense_dirs)
        print(f"  Dense rays: {N_dense}")

        dense_origins_t = torch.from_numpy(dense_origins).unsqueeze(0).to(device)
        dense_dirs_t = torch.from_numpy(dense_dirs).unsqueeze(0).to(device)

        print(f"  Rendering dense scan...")
        dense_depths_t = render_rays(model, quantized, dense_origins_t, dense_dirs_t,
                                     cfg, device, chunk_size=args.chunk_size)

        dense_pts = depths_to_points(
            torch.from_numpy(dense_origins),
            torch.from_numpy(dense_dirs),
            dense_depths_t,
        ).numpy()

        # Save
        np.savez_compressed(
            os.path.join(args.output_dir, "dense_reconstruction.npz"),
            dense_points=dense_pts,
            dense_depths=dense_depths_t.numpy(),
            dense_directions=dense_dirs,
        )

        plot_dense_reconstruction(
            dense_pts, cfg,
            os.path.join(args.output_dir, "dense_bev.png"),
        )

    # ── Summary ──────────────────────────────────────────────────────────
    summary_path = os.path.join(args.output_dir, "metrics.txt")
    with open(summary_path, "w") as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Config:     {args.config}\n")
        f.write(f"Step:       {step}\n")
        f.write(f"Split:      {args.split}, sample {args.sample_idx}\n")
        f.write(f"Sequence:   {seq}, frame {frame_idx}\n")
        f.write(f"Raw points: {raw_points.shape[0]}\n")
        f.write(f"ROI points: {N_roi}\n")
        f.write(f"Valid rays: {N_valid}\n")
        f.write(f"VQ active:  {active}\n")
        f.write(f"VQ perpl:   {perp}\n")
        f.write(f"\n── Depth Metrics ──\n")
        for k, v in metrics.items():
            f.write(f"  {k:20s}: {v:.4f}\n")
    print(f"\n  Saved metrics to {summary_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
