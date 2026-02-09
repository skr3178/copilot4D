"""Train discrete diffusion on Moving MNIST.

This script tests the CoPilot4D discrete diffusion training approach:
- Algorithm 1: Absorbing-Uniform masking + noise
- 3 training objectives: future prediction, joint denoise, individual denoise
- Cross-entropy loss on all positions
"""

import os
import sys
import math
import random
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.mnist_diffusion.moving_mnist_precomputed import (
    MovingMNISTPrecomputed, create_mnist_dataloaders
)
from tests.mnist_diffusion.simple_model import SimpleVideoTransformer


# ============================================================================
# Discrete Diffusion Components (ported from copilot4d/world_model/masking.py)
# ============================================================================

def cosine_mask_schedule(u: torch.Tensor) -> torch.Tensor:
    """Mask schedule: gamma(u) = cos(u * pi/2)."""
    return torch.cos(u * math.pi / 2)


class DiscreteDiffusionMasker:
    """Prepares masked batches for discrete diffusion training."""
    
    def __init__(
        self,
        vocab_size=16,
        mask_token_id=16,
        num_past_frames=10,
        noise_eta=5.0,
        prob_future_pred=0.5,
        prob_joint_denoise=0.4,
    ):
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.num_past_frames = num_past_frames
        self.noise_eta = noise_eta
        self.prob_future_pred = prob_future_pred
        self.prob_joint_denoise = prob_joint_denoise
    
    def sample_objective(self) -> str:
        """Sample training objective according to paper probabilities."""
        r = random.random()
        if r < self.prob_future_pred:
            return "future_prediction"
        elif r < self.prob_future_pred + self.prob_joint_denoise:
            return "joint_denoise"
        else:
            return "individual_denoise"
    
    def apply_random_masking(
        self,
        tokens: torch.Tensor,
        mask_ratio: float,
        noise_ratio: float = 0.0,
    ):
        """Apply random masking and optional noise to tokens.
        
        Algorithm 1 steps 1-2:
        1. Mask ceil(gamma(u)*N) tokens with mask_token_id
        2. Add random noise to floor(eta% * unmasked) tokens
        """
        device = tokens.device
        flat_tokens = tokens.reshape(-1)
        N = flat_tokens.numel()
        
        # Number of tokens to mask
        num_mask = math.ceil(mask_ratio * N)
        
        # Random mask positions
        all_indices = torch.randperm(N, device=device)
        mask_indices = all_indices[:num_mask]
        
        # Apply mask
        masked_tokens = flat_tokens.clone()
        masked_tokens[mask_indices] = self.mask_token_id
        
        # Track which positions were masked
        was_masked = torch.zeros(N, dtype=torch.bool, device=device)
        was_masked[mask_indices] = True
        
        # Add noise to unmasked positions (Algorithm 1 step 2)
        if noise_ratio > 0 and num_mask < N:
            unmasked_indices = all_indices[num_mask:]
            num_noise = min(int(noise_ratio * len(unmasked_indices)), len(unmasked_indices))
            if num_noise > 0:
                noise_indices = unmasked_indices[:num_noise]
                # Random codebook indices (not mask token)
                random_tokens = torch.randint(
                    0, self.vocab_size, (num_noise,), device=device
                )
                masked_tokens[noise_indices] = random_tokens
        
        # Reshape back
        masked_tokens = masked_tokens.reshape(tokens.shape)
        was_masked = was_masked.reshape(tokens.shape)
        
        return masked_tokens, was_masked
    
    def prepare_batch(self, tokens: torch.Tensor, objective: str = None):
        """Prepare a batch for training with discrete diffusion.
        
        Args:
            tokens: (B, T, H, W) ground truth token indices
            objective: which training objective to use (None = random sample)
            
        Returns:
            Dict with masked tokens, targets, temporal mask, etc.
        """
        B, T, H, W = tokens.shape
        device = tokens.device
        
        # Sample objective if not specified
        if objective is None:
            objective = self.sample_objective()
        
        num_past = self.num_past_frames
        
        # Prepare output tensors
        out_tokens = tokens.clone()
        temporal_mask = None
        was_masked = torch.zeros_like(tokens, dtype=torch.bool)
        
        # Sample mask ratio: u0 ~ Uniform(0,1), ratio = gamma(u0)
        u0 = random.random()
        mask_ratio = cosine_mask_schedule(torch.tensor(u0)).item()
        
        # Sample noise ratio: u1 ~ Uniform(0,1), noise = u1 * eta%
        u1 = random.random()
        noise_ratio = u1 * self.noise_eta / 100.0
        
        if objective == "future_prediction":
            # Past frames: ground truth (no masking)
            # Future frames: masked + noised
            temporal_mask = self._make_causal_mask(T, device)
            
            for b in range(B):
                future_tokens = tokens[b, num_past:]
                masked_future, was_masked_future = self.apply_random_masking(
                    future_tokens,
                    mask_ratio=mask_ratio,
                    noise_ratio=noise_ratio,
                )
                out_tokens[b, num_past:] = masked_future
                was_masked[b, num_past:] = was_masked_future
                
        elif objective == "joint_denoise":
            # All frames partially masked + noised
            temporal_mask = self._make_causal_mask(T, device)
            
            for b in range(B):
                masked, was_m = self.apply_random_masking(
                    tokens[b],
                    mask_ratio=mask_ratio,
                    noise_ratio=noise_ratio,
                )
                out_tokens[b] = masked
                was_masked[b] = was_m
                
        elif objective == "individual_denoise":
            # Each frame independently masked + noised
            # Identity temporal mask (each frame only attends to itself)
            temporal_mask = self._make_identity_mask(T, device)
            
            for b in range(B):
                for t in range(T):
                    masked, was_m = self.apply_random_masking(
                        tokens[b, t],
                        mask_ratio=mask_ratio,
                        noise_ratio=noise_ratio,
                    )
                    out_tokens[b, t] = masked
                    was_masked[b, t] = was_m
        
        return {
            "tokens": out_tokens,
            "targets": tokens,
            "temporal_mask": temporal_mask,
            "objective": objective,
            "was_masked": was_masked,
            "mask_ratio": mask_ratio,
        }
    
    @staticmethod
    def _make_causal_mask(T: int, device: torch.device) -> torch.Tensor:
        """Lower-triangular causal mask for temporal attention."""
        mask = torch.full((T, T), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask
    
    @staticmethod
    def _make_identity_mask(T: int, device: torch.device) -> torch.Tensor:
        """Diagonal-only mask (each frame attends only to itself)."""
        mask = torch.full((T, T), float("-inf"), device=device)
        mask.fill_diagonal_(0.0)
        return mask


def compute_diffusion_loss(logits, targets, label_smoothing=0.1):
    """Compute cross-entropy loss for discrete diffusion.
    
    Loss is computed on ALL positions (masked and unmasked) as per the paper.
    
    Args:
        logits: (B, T, H*W, vocab_size+1) model predictions
        targets: (B, T, H, W) ground truth token indices
        label_smoothing: label smoothing factor
        
    Returns:
        Scalar loss tensor
    """
    B, T, N, V = logits.shape
    
    # Flatten targets
    targets_flat = targets.reshape(B, T, N)
    
    # Reshape logits for F.cross_entropy
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets_flat.reshape(-1)
    
    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        label_smoothing=label_smoothing,
    )
    
    return loss


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(model, train_loader, masker, optimizer, scaler, device, cfg, epoch, writer, global_step):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_samples = 0
    
    objectives_count = {"future_prediction": 0, "joint_denoise": 0, "individual_denoise": 0}
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        tokens = batch["tokens"].to(device)  # (B, T, H, W)
        actions = batch["actions"].to(device)  # (B, T, action_dim)
        
        # Prepare masked batch
        masked_batch = masker.prepare_batch(tokens)
        masked_tokens = masked_batch["tokens"]
        targets = masked_batch["targets"]
        temporal_mask = masked_batch["temporal_mask"]
        objective = masked_batch["objective"]
        
        objectives_count[objective] += 1
        
        # Forward pass with mixed precision
        with autocast(enabled=cfg.use_amp):
            logits = model(masked_tokens, actions, temporal_mask)
            loss = compute_diffusion_loss(logits, targets, cfg.label_smoothing)
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Logging
        batch_size = tokens.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        global_step += 1
        
        # TensorBoard logging
        if writer is not None and global_step % cfg.log_interval == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/mask_ratio", masked_batch["mask_ratio"], global_step)
            writer.add_scalar("train/objective", list(objectives_count.keys()).index(objective), global_step)
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "mask_r": f"{masked_batch['mask_ratio']:.2f}",
            "obj": objective[:4],
        })
    
    avg_loss = total_loss / total_samples
    return avg_loss, global_step, objectives_count


def validate(model, val_loader, masker, device, cfg):
    """Validate on validation set."""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    # Test each objective separately
    objective_losses = {"future_prediction": 0, "joint_denoise": 0, "individual_denoise": 0}
    objective_counts = {"future_prediction": 0, "joint_denoise": 0, "individual_denoise": 0}
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            tokens = batch["tokens"].to(device)
            actions = batch["actions"].to(device)
            
            for objective in ["future_prediction", "joint_denoise", "individual_denoise"]:
                masked_batch = masker.prepare_batch(tokens, objective=objective)
                masked_tokens = masked_batch["tokens"]
                targets = masked_batch["targets"]
                temporal_mask = masked_batch["temporal_mask"]
                
                with autocast(enabled=cfg.use_amp):
                    logits = model(masked_tokens, actions, temporal_mask)
                    loss = compute_diffusion_loss(logits, targets, 0.0)  # No label smoothing for val
                
                batch_size = tokens.size(0)
                objective_losses[objective] += loss.item() * batch_size
                objective_counts[objective] += batch_size
            
            total_samples += tokens.size(0)
    
    # Average losses
    for obj in objective_losses:
        if objective_counts[obj] > 0:
            objective_losses[obj] /= objective_counts[obj]
    
    avg_loss = sum(objective_losses.values()) / 3
    return avg_loss, objective_losses


def main():
    parser = argparse.ArgumentParser(description="Train discrete diffusion on Moving MNIST")
    
    # Data
    parser.add_argument("--data_path", type=str, default="mnist_test_seq.1.npy")
    parser.add_argument("--num_train", type=int, default=8000)
    parser.add_argument("--num_val", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_token_levels", type=int, default=16)
    parser.add_argument("--frame_size", type=int, default=64)
    
    # Model
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--use_amp", action="store_true", default=True)
    
    # Masking
    parser.add_argument("--num_past_frames", type=int, default=10)
    parser.add_argument("--noise_eta", type=float, default=5.0)
    
    # Logging
    parser.add_argument("--output_dir", type=str, default="outputs/mnist_diffusion")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    
    cfg = parser.parse_args()
    
    # Setup
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    import json
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(cfg), f, indent=2)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_mnist_dataloaders(
        data_path=cfg.data_path,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        num_train=cfg.num_train,
        num_val=cfg.num_val,
        num_workers=cfg.num_workers,
        num_token_levels=cfg.num_token_levels,
        frame_size=cfg.frame_size,
    )
    
    # Create model
    print("\nCreating model...")
    model = SimpleVideoTransformer(
        vocab_size=cfg.num_token_levels,
        mask_token_id=cfg.num_token_levels,
        num_frames=cfg.seq_len,
        height=cfg.frame_size,
        width=cfg.frame_size,
        embed_dim=cfg.embed_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    
    # Create masker
    masker = DiscreteDiffusionMasker(
        vocab_size=cfg.num_token_levels,
        mask_token_id=cfg.num_token_levels,
        num_past_frames=cfg.num_past_frames,
        noise_eta=cfg.noise_eta,
    )
    
    # Create scaler for AMP
    scaler = GradScaler(enabled=cfg.use_amp)
    
    # Create TensorBoard writer
    log_dir = output_dir / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir)
    
    # Training loop
    print("\nStarting training...")
    global_step = 0
    best_val_loss = float("inf")
    
    for epoch in range(1, cfg.epochs + 1):
        # Train
        train_loss, global_step, obj_counts = train_epoch(
            model, train_loader, masker, optimizer, scaler, device, cfg, epoch, writer, global_step
        )
        
        print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}")
        print(f"  Objectives: {obj_counts}")
        
        # Validate
        if epoch % 5 == 0 or epoch == cfg.epochs:
            val_loss, obj_losses = validate(model, val_loader, masker, device, cfg)
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"    Future Pred: {obj_losses['future_prediction']:.4f}")
            print(f"    Joint Denoise: {obj_losses['joint_denoise']:.4f}")
            print(f"    Individual Denoise: {obj_losses['individual_denoise']:.4f}")
            
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/future_pred", obj_losses["future_prediction"], epoch)
            writer.add_scalar("val/joint_denoise", obj_losses["joint_denoise"], epoch)
            writer.add_scalar("val/individual_denoise", obj_losses["individual_denoise"], epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                }, output_dir / "best_model.pt")
                print(f"  Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint
        if epoch % cfg.save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
            }, output_dir / f"checkpoint_epoch{epoch}.pt")
    
    writer.close()
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
