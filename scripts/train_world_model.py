#!/usr/bin/env python3
"""Train the CoPilot4D world model.

Discrete diffusion training on pre-tokenized KITTI sequences.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import yaml
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from copilot4d.utils.config import WorldModelConfig
from copilot4d.world_model.world_model import CoPilot4DWorldModel
from copilot4d.world_model.masking import DiscreteDiffusionMasker, compute_diffusion_loss
from copilot4d.data.kitti_sequence_dataset import KITTISequenceDataset, sequence_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Train CoPilot4D world model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    return parser.parse_args()


def load_config(config_path: str) -> WorldModelConfig:
    """Load config from YAML."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return WorldModelConfig(**config_dict)


def build_model(cfg: WorldModelConfig, device: str) -> CoPilot4DWorldModel:
    """Build and initialize world model."""
    model = CoPilot4DWorldModel(cfg)
    model.to(device)
    return model


def build_optimizer(model: CoPilot4DWorldModel, cfg: WorldModelConfig):
    """Build AdamW optimizer with weight decay."""
    # Exclude bias, norm, and embedding from weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # No weight decay for bias, norm, embedding
        if "bias" in name or "norm" in name.lower() or "embedding" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    optimizer = AdamW(
        param_groups,
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
    )
    
    return optimizer


def build_scheduler(optimizer, cfg: WorldModelConfig):
    """Build learning rate scheduler with warmup and cosine decay."""
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(step):
        if step < cfg.warmup_steps:
            # Linear warmup
            return step / cfg.warmup_steps
        else:
            # Cosine decay
            progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            decayed = (1 - cfg.cosine_min_ratio) * cosine_decay + cfg.cosine_min_ratio
            return decayed
    
    return LambdaLR(optimizer, lr_lambda)


def train_step(
    model: CoPilot4DWorldModel,
    masker: DiscreteDiffusionMasker,
    batch: dict,
    optimizer: AdamW,
    scaler: torch.cuda.amp.GradScaler,
    cfg: WorldModelConfig,
    device: str,
) -> dict:
    """Single training step.
    
    Returns:
        Dict with loss and metrics
    """
    tokens = batch["tokens"].to(device)
    actions = batch["actions"].to(device)
    
    B, T, H, W = tokens.shape
    
    # Apply discrete diffusion masking
    masked_batch = masker.prepare_batch(tokens)
    masked_tokens = masked_batch["tokens"]
    targets = masked_batch["targets"]
    temporal_mask = masked_batch["temporal_mask"]
    objective = masked_batch["objective"]
    
    with torch.cuda.amp.autocast(enabled=cfg.amp):
        # Forward pass
        logits = model(masked_tokens, actions, temporal_mask)
        # logits: (B, T, H*W, vocab_size)
        
        # Compute loss
        loss = compute_diffusion_loss(logits, targets, cfg.label_smoothing)
    
    # Backward
    optimizer.zero_grad()
    if cfg.amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
    
    # Compute accuracy
    with torch.no_grad():
        preds = logits.argmax(dim=-1).reshape(B, T, H, W)
        acc = (preds == targets).float().mean()
        
        # Per-objective metrics
        metrics = {
            "loss": loss.item(),
            "acc": acc.item(),
            "objective": objective,
        }
    
    return metrics


def train_epoch(
    model: CoPilot4DWorldModel,
    masker: DiscreteDiffusionMasker,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    cfg: WorldModelConfig,
    device: str,
    step: int,
    val_loader: DataLoader = None,
) -> int:
    """Train for one epoch.
    
    Returns:
        Updated step count
    """
    model.train()
    
    pbar = tqdm(dataloader, desc=f"Step {step}")
    for batch in pbar:
        metrics = train_step(
            model, masker, batch, optimizer, scaler, cfg, device
        )
        
        scheduler.step()
        step += 1
        
        # Update progress bar
        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({
            "loss": f"{metrics['loss']:.4f}",
            "acc": f"{metrics['acc']:.4f}",
            "obj": metrics["objective"][:3],
            "lr": f"{lr:.2e}",
        })
        
        # Logging
        if step % cfg.log_every_steps == 0:
            log_file = os.path.join(cfg.output_dir, "train_log.jsonl")
            log_metrics(step, metrics, lr, log_file)

        # Checkpointing
        if step % cfg.save_every_steps == 0:
            save_checkpoint(model, optimizer, scheduler, step, cfg, is_latest=True)
            save_checkpoint(model, optimizer, scheduler, step, cfg, is_latest=False)

        # Evaluation
        if step % cfg.eval_every_steps == 0 and val_loader is not None:
            val_metrics = validate(model, masker, val_loader, cfg, device)
            log_file = os.path.join(cfg.output_dir, "val_log.jsonl")
            log_metrics(step, val_metrics, lr, log_file)
            print(f"  Val: loss={val_metrics['val_loss']:.4f}, acc={val_metrics['val_acc']:.4f}")
        
        # Max steps check
        if step >= cfg.max_steps:
            break
    
    return step


@torch.no_grad()
def validate(
    model: CoPilot4DWorldModel,
    masker: DiscreteDiffusionMasker,
    dataloader: DataLoader,
    cfg: WorldModelConfig,
    device: str,
    num_batches: int = 10,
) -> dict:
    """Run validation and return metrics."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        tokens = batch["tokens"].to(device)
        actions = batch["actions"].to(device)
        B, T, H, W = tokens.shape

        masked_batch = masker.prepare_batch(tokens)
        masked_tokens = masked_batch["tokens"]
        targets = masked_batch["targets"]
        temporal_mask = masked_batch["temporal_mask"]

        with torch.cuda.amp.autocast(enabled=cfg.amp):
            logits = model(masked_tokens, actions, temporal_mask)
            loss = compute_diffusion_loss(logits, targets, cfg.label_smoothing)

        preds = logits.argmax(dim=-1).reshape(B, T, H, W)
        acc = (preds == targets).float().mean()

        total_loss += loss.item()
        total_acc += acc.item()
        count += 1

    model.train()
    return {
        "val_loss": total_loss / max(count, 1),
        "val_acc": total_acc / max(count, 1),
    }


def log_metrics(step: int, metrics: dict, lr: float, log_file: str = None):
    """Log metrics to console and JSONL file."""
    log_entry = {"step": step, "lr": lr, "time": time.time(), **metrics}
    parts = [f"Step {step}"]
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        elif isinstance(v, str):
            parts.append(f"{k}={v}")
    parts.append(f"lr={lr:.2e}")
    log_str = ", ".join(parts)
    print(log_str)

    if log_file:
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


def save_checkpoint(
    model: CoPilot4DWorldModel,
    optimizer: AdamW,
    scheduler,
    step: int,
    cfg: WorldModelConfig,
    is_latest: bool = False,
):
    """Save model checkpoint."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": cfg,
    }
    
    if is_latest:
        path = output_dir / "checkpoint_latest.pt"
    else:
        path = output_dir / f"checkpoint_{step:07d}.pt"
    
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def main():
    args = parse_args()
    
    # Setup
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print(f"Training world model on {device}")
    print(f"Config: {cfg}")
    
    # Build model
    model = build_model(cfg, device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    
    # Resume if specified
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = checkpoint.get("step", 0)
        print(f"Resumed from step {start_step}")
    
    # Build datasets
    train_dataset = KITTISequenceDataset(cfg, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=sequence_collate_fn,
        pin_memory=True,
    )
    
    # Validation dataset
    val_dataset = KITTISequenceDataset(cfg, split="val")
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=sequence_collate_fn,
            pin_memory=True,
        )

    # Masker for discrete diffusion
    masker = DiscreteDiffusionMasker(cfg)

    # Ensure output dir exists for logging
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Training loop
    step = start_step
    epoch = 0

    while step < cfg.max_steps:
        print(f"\nEpoch {epoch}")
        step = train_epoch(
            model, masker, train_loader, optimizer, scheduler,
            scaler, cfg, device, step, val_loader
        )
        epoch += 1
    
    print("Training complete!")
    save_checkpoint(model, optimizer, scheduler, step, cfg)


if __name__ == "__main__":
    main()
