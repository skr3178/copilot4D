"""Training script for CoPilot4D tokenizer with comprehensive monitoring."""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from copilot4d.utils.config import TokenizerConfig
from copilot4d.tokenizer.tokenizer_model import CoPilot4DTokenizer
from copilot4d.data.kitti_dataset import KITTITokenizerDataset, tokenizer_collate_fn
from copilot4d.tokenizer.metrics import chamfer_distance_from_depths, compute_depth_metrics


def load_config(config_path: str) -> TokenizerConfig:
    """Load config from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return TokenizerConfig(**config_dict)


def create_dataloaders(cfg: TokenizerConfig):
    """Create train and validation dataloaders."""
    train_dataset = KITTITokenizerDataset(cfg, split="train")
    val_dataset = KITTITokenizerDataset(cfg, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=tokenizer_collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=tokenizer_collate_fn,
        drop_last=False,
    )

    return train_loader, val_loader


def create_optimizer(model: nn.Module, cfg: TokenizerConfig):
    """Create AdamW optimizer with weight decay."""
    # Separate parameters with/without weight decay
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "embed" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    optimizer = AdamW(param_groups, lr=cfg.lr)
    return optimizer


def create_scheduler(optimizer, cfg: TokenizerConfig):
    """Create warmup + cosine annealing learning rate scheduler."""
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=cfg.warmup_steps,
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.max_steps - cfg.warmup_steps,
        eta_min=cfg.lr * 0.01,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.warmup_steps],
    )

    return scheduler


def train_step(
    model: CoPilot4DTokenizer,
    batch: dict,
    optimizer: AdamW,
    scaler: GradScaler,
    cfg: TokenizerConfig,
    device: torch.device,
) -> dict:
    """Single training step."""
    model.train()

    # Move batch to device
    features = batch["features"].to(device)
    num_points = batch["num_points"].to(device)
    coords = batch["coords"].to(device)
    ray_origins = batch["ray_origins"].to(device)
    ray_directions = batch["ray_directions"].to(device)
    ray_depths = batch["ray_depths"].to(device)
    gt_occupancy = batch["gt_occupancy"].to(device)
    batch_size = batch["batch_size"]

    # Forward pass with AMP
    with autocast(enabled=cfg.amp):
        outputs = model(
            features=features,
            num_points=num_points,
            coords=coords,
            batch_size=batch_size,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            gt_depths=ray_depths,
            gt_occupancy=gt_occupancy,
        )

        loss = outputs["losses"]["total"]
        loss = loss / cfg.grad_accum_steps  # Scale for gradient accumulation

    # Backward pass
    if cfg.amp and scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    return {
        "loss": loss.item() * cfg.grad_accum_steps,
        "losses": outputs["losses"],
    }


@torch.no_grad()
def eval_step(
    model: CoPilot4DTokenizer,
    batch: dict,
    cfg: TokenizerConfig,
    device: torch.device,
) -> dict:
    """Single evaluation step."""
    model.eval()

    # Move batch to device
    features = batch["features"].to(device)
    num_points = batch["num_points"].to(device)
    coords = batch["coords"].to(device)
    ray_origins = batch["ray_origins"].to(device)
    ray_directions = batch["ray_directions"].to(device)
    ray_depths = batch["ray_depths"].to(device)
    gt_occupancy = batch["gt_occupancy"].to(device)
    batch_size = batch["batch_size"]

    # Forward pass
    with autocast(enabled=cfg.amp):
        outputs = model(
            features=features,
            num_points=num_points,
            coords=coords,
            batch_size=batch_size,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            gt_depths=ray_depths,
            gt_occupancy=gt_occupancy,
        )

    return {
        "loss": outputs["losses"]["total"].item(),
        "losses": outputs["losses"],
        "vq_metrics": outputs.get("vq_metrics", {}),
        "pred_depths": outputs.get("pred_depths"),
        "gt_depths": ray_depths,
        "ray_origins": ray_origins,
        "ray_directions": ray_directions,
    }


def log_training_metrics(step: int, train_outputs: dict, lr: float, output_dir: str):
    """Log detailed training metrics to file.
    
    Tracks comprehensive codebook health and reconstruction quality.
    """
    losses = train_outputs.get("losses", {})
    vq_metrics = train_outputs.get("vq_metrics", {})
    
    def to_float(val):
        """Convert tensor or any numeric to float."""
        if hasattr(val, 'item'):
            return val.item()
        return float(val) if val is not None else 0.0
    
    metrics = {
        "step": step,
        "lr": lr,
        "loss_total": to_float(train_outputs.get("loss", 0)),
        # Depth metrics
        "depth_l1": to_float(losses.get("depth_l1", 0)),
        "surface_conc": to_float(losses.get("surface_conc", 0)),
        # VQ loss components
        "vq_total": to_float(losses.get("vq", 0)),
        "vq_codebook": to_float(losses.get("vq_codebook", 0)),
        "vq_commitment": to_float(losses.get("vq_commitment", 0)),
        "vq_ratio": to_float(vq_metrics.get("vq_loss_ratio", 0)),
        # Codebook health metrics
        "vq_perplexity": to_float(vq_metrics.get("vq_perplexity", 0)),
        "vq_entropy": to_float(vq_metrics.get("vq_entropy", 0)),
        "vq_entropy_norm": to_float(vq_metrics.get("vq_entropy_norm", 0)),
        "vq_active_pct": to_float(vq_metrics.get("vq_active_pct", 0)),
        "vq_active_codes": to_float(vq_metrics.get("vq_active_codes", 0)),
        "vq_dead_codes": to_float(vq_metrics.get("vq_dead_codes", 0)),
        "vq_dead_pct": to_float(vq_metrics.get("vq_dead_pct", 0)),
        # Cluster statistics
        "vq_cluster_max": to_float(vq_metrics.get("vq_cluster_max", 0)),
        "vq_cluster_mean": to_float(vq_metrics.get("vq_cluster_mean", 0)),
        "vq_cluster_std": to_float(vq_metrics.get("vq_cluster_std", 0)),
        "vq_gini": to_float(vq_metrics.get("vq_gini", 0)),
        # Spatial skip
        "skip_bce": to_float(losses.get("skip_bce", 0)),
    }
    
    log_file = Path(output_dir) / "training_metrics.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(metrics) + "\n")


@torch.no_grad()
def evaluate_model(
    model: CoPilot4DTokenizer,
    val_loader: DataLoader,
    cfg: TokenizerConfig,
    device: torch.device,
    num_batches: int = 10,
) -> dict:
    """Comprehensive model evaluation with Chamfer distance.
    
    Returns:
        dict with aggregated metrics across validation batches
    """
    model.eval()
    
    # Accumulators for metrics
    all_losses = []
    all_depth_l1 = []
    all_chamfer = []
    all_vq_perplexity = []
    all_vq_active_pct = []
    all_vq_dead_pct = []
    all_vq_entropy = []
    
    for i, val_batch in enumerate(val_loader):
        if i >= num_batches:
            break
        
        val_outputs = eval_step(model, val_batch, cfg, device)
        
        # Collect basic metrics
        all_losses.append(val_outputs["loss"])
        all_depth_l1.append(val_outputs["losses"].get("depth_l1", 0))
        
        # VQ metrics
        vq_metrics = val_outputs.get("vq_metrics", {})
        all_vq_perplexity.append(vq_metrics.get("vq_perplexity", 0))
        all_vq_active_pct.append(vq_metrics.get("vq_active_pct", 0))
        all_vq_dead_pct.append(vq_metrics.get("vq_dead_pct", 0))
        all_vq_entropy.append(vq_metrics.get("vq_entropy", 0))
        
        # Chamfer distance (if we have depth predictions)
        if val_outputs.get("pred_depths") is not None:
            cd, cd_info = chamfer_distance_from_depths(
                val_outputs["ray_origins"],
                val_outputs["ray_directions"],
                val_outputs["pred_depths"],
                val_outputs["gt_depths"],
            )
            all_chamfer.append(cd.item())
    
    # Compute means
    metrics = {
        "loss": sum(all_losses) / len(all_losses),
        "depth_l1": sum(all_depth_l1) / len(all_depth_l1),
        "chamfer": sum(all_chamfer) / len(all_chamfer) if all_chamfer else 0.0,
        "vq_perplexity": sum(all_vq_perplexity) / len(all_vq_perplexity),
        "vq_active_pct": sum(all_vq_active_pct) / len(all_vq_active_pct),
        "vq_dead_pct": sum(all_vq_dead_pct) / len(all_vq_dead_pct),
        "vq_entropy": sum(all_vq_entropy) / len(all_vq_entropy),
    }
    
    model.train()
    return metrics


def save_eval_metrics(step: int, metrics: dict, output_dir: str):
    """Save evaluation metrics to file."""
    # Convert all values to JSON-serializable types
    json_metrics = {"step": int(step), "split": "val"}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            json_metrics[key] = float(value.item())
        elif isinstance(value, (int, float)):
            json_metrics[key] = value
        else:
            json_metrics[key] = float(value)  # Handle numpy types
    
    log_file = Path(output_dir) / "eval_metrics.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(json_metrics) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train CoPilot4D tokenizer")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    print(f"Loaded config from {args.config}")
    print(f"  Token grid size: {cfg.token_grid_size}x{cfg.token_grid_size}")
    print(f"  Voxel grid: {cfg.voxel_grid_xy}x{cfg.voxel_grid_xy}x{cfg.voxel_grid_z}")
    print(f"  Batch size: {cfg.batch_size}, Accum steps: {cfg.grad_accum_steps}")

    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(cfg)
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")

    # Create model
    print("Creating model...")
    model = CoPilot4DTokenizer(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params / 1e6:.2f}M")

    # Optimizer and scheduler
    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, cfg)
    scaler = GradScaler() if cfg.amp else None

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_step = checkpoint["step"]

    # Training loop
    print(f"\nStarting training from step {start_step}...")
    step = start_step
    optimizer.zero_grad()

    pbar = tqdm(total=cfg.max_steps, initial=start_step, desc="Training")

    while step < cfg.max_steps:
        for batch in train_loader:
            # Training step
            train_outputs = train_step(
                model, batch, optimizer, scaler, cfg, device
            )

            # Gradient accumulation
            if (step + 1) % cfg.grad_accum_steps == 0:
                if cfg.amp and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Logging
            if step % 10 == 0:
                losses = train_outputs.get("losses", {})
                
                postfix = {
                    "loss": f"{train_outputs['loss']:.3f}",
                    "depth": f"{losses.get('depth_l1', 0):.3f}",
                    "vq": f"{losses.get('vq', 0):.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.6f}",
                }
                pbar.set_postfix(postfix)
            
            # Detailed logging every 100 steps
            if step % 100 == 0:
                log_training_metrics(step, train_outputs, scheduler.get_last_lr()[0], cfg.output_dir)

            # Evaluation
            if step % cfg.eval_every_steps == 0 and step > 0:
                print(f"\n--- Eval at step {step} ---")
                eval_metrics = evaluate_model(model, val_loader, cfg, device, cfg.num_eval_batches)
                print(f"Val loss: {eval_metrics['loss']:.4f}")
                print(f"Val depth L1: {eval_metrics.get('depth_l1', 0):.4f}")
                print(f"Val Chamfer: {eval_metrics.get('chamfer', 0):.4f}")
                print(f"VQ perplexity: {eval_metrics.get('vq_perplexity', 0):.1f}")
                print(f"VQ active: {eval_metrics.get('vq_active_pct', 0):.1f}%")
                
                # Save eval metrics
                save_eval_metrics(step, eval_metrics, cfg.output_dir)

            # Checkpointing
            if step % cfg.save_every_steps == 0 and step > 0:
                checkpoint_path = os.path.join(cfg.output_dir, f"checkpoint_step_{step}.pt")
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": cfg,
                }, checkpoint_path)
                print(f"\nSaved checkpoint to {checkpoint_path}")

            step += 1
            pbar.update(1)

            if step >= cfg.max_steps:
                break

    pbar.close()

    # Final save
    final_path = os.path.join(cfg.output_dir, "checkpoint_final.pt")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": cfg,
    }, final_path)
    print(f"\nTraining complete! Final checkpoint saved to {final_path}")


if __name__ == "__main__":
    main()
