"""Training script for CoPilot4D tokenizer."""

import os
import sys
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

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
    }


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
                losses = train_outputs["losses"]
                pbar.set_postfix({
                    "loss": f"{train_outputs['loss']:.4f}",
                    "depth": f"{losses['depth_l1']:.4f}",
                    "vq": f"{losses['vq']:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.6f}",
                })

            # Evaluation
            if step % cfg.eval_every_steps == 0 and step > 0:
                print(f"\n--- Eval at step {step} ---")
                model.eval()
                eval_losses = []

                for i, val_batch in enumerate(val_loader):
                    if i >= cfg.num_eval_batches:
                        break
                    val_outputs = eval_step(model, val_batch, cfg, device)
                    eval_losses.append(val_outputs["loss"])

                avg_eval_loss = sum(eval_losses) / len(eval_losses)
                print(f"Val loss: {avg_eval_loss:.4f}")
                model.train()

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
