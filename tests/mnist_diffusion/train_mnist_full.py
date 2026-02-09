"""Full-scale training of discrete diffusion on Moving MNIST.

This uses the EXACT same diffusion process as the CoPilot4D world model:
- Algorithm 1: Absorbing-Uniform masking + noise
- 3 training objectives: future prediction (50%), joint denoise (40%), individual denoise (10%)
- Cosine mask schedule: gamma(u) = cos(u * pi/2)
- Label smoothing: 0.1

Key improvements for high-quality generation:
- Larger model: 10M+ params (8 layers, 512 dim, 8 heads)
- Full data: 8000 train / 2000 val sequences
- Longer training: 100+ epochs
- EMA for stable sampling
- Gradient clipping and mixed precision
"""

import os
import sys
import math
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import copy

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.mnist_diffusion.moving_mnist_precomputed import (
    MovingMNISTPrecomputed, create_mnist_dataloaders
)
from tests.mnist_diffusion.simple_model import SimpleVideoTransformer


# ============================================================================
# EXACT diffusion components from copilot4d/world_model/masking.py
# ============================================================================

def cosine_mask_schedule(u: torch.Tensor) -> torch.Tensor:
    """Mask schedule: gamma(u) = cos(u * pi/2)."""
    return torch.cos(u * math.pi / 2)


class DiscreteDiffusionMasker:
    """Prepares masked batches for discrete diffusion training.
    
    EXACT implementation from copilot4d/world_model/masking.py
    """

    def __init__(
        self,
        vocab_size: int = 16,
        mask_token_id: int = 16,
        num_past_frames: int = 10,
        noise_eta: float = 5.0,
        prob_future_pred: float = 0.5,
        prob_joint_denoise: float = 0.4,
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

    def prepare_batch(
        self,
        tokens: torch.Tensor,
        objective: str = None,
    ) -> Dict[str, torch.Tensor]:
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


def compute_diffusion_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """Compute cross-entropy loss for discrete diffusion.
    
    EXACT implementation from copilot4d/world_model/masking.py
    Loss is computed on ALL positions (masked and unmasked) as per the paper.
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
# EMA for model parameters
# ============================================================================

class ModelEMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False
    
    def update(self, model: torch.nn.Module):
        """Update EMA model parameters."""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def state_dict(self):
        return {
            'ema_model': self.ema_model.state_dict(),
            'decay': self.decay,
        }
    
    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict['ema_model'])
        self.decay = state_dict['decay']


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(model, train_loader, masker, optimizer, scaler, device, cfg, epoch, writer, global_step):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0
    total_samples = 0
    
    objectives_count = {"future_prediction": 0, "joint_denoise": 0, "individual_denoise": 0}
    
    # Gradient accumulation steps
    accum_steps = getattr(cfg, 'grad_accum_steps', 1)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        tokens = batch["tokens"].to(device)  # (B, T, H, W)
        actions = batch["actions"].to(device)  # (B, T, action_dim)
        
        # Prepare masked batch
        masked_batch = masker.prepare_batch(tokens)
        masked_tokens = masked_batch["tokens"]
        targets = masked_batch["targets"]
        temporal_mask = masked_batch["temporal_mask"]
        objective = masked_batch["objective"]
        
        objectives_count[objective] += tokens.size(0)
        
        # Forward pass with mixed precision
        with autocast(enabled=cfg.use_amp):
            logits = model(masked_tokens, actions, temporal_mask)
            loss = compute_diffusion_loss(logits, targets, cfg.label_smoothing)
            # Scale loss for gradient accumulation
            loss = loss / accum_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Only update weights after accumulation steps
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Logging
        batch_size = tokens.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        global_step += 1
        
        # TensorBoard logging
        if writer is not None and global_step % cfg.log_interval == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/mask_ratio", masked_batch["mask_ratio"], global_step)
        
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


# ============================================================================
# Sampling (Algorithm 2 from paper)
# ============================================================================

def cosine_schedule(t: float) -> float:
    """Cosine mask schedule: gamma(t) = cos(t * pi/2)."""
    return math.cos(t * math.pi / 2)


@torch.no_grad()
def generate_video(
    model,
    num_frames: int,
    height: int,
    width: int,
    num_steps: int = 20,
    device: str = 'cuda',
    temperature: float = 1.0,
    action_dim: int = 4,
) -> torch.Tensor:
    """Generate a video from scratch using iterative decoding (Algorithm 2).
    
    Args:
        model: trained model
        num_frames: number of frames to generate
        height: frame height
        width: frame width
        num_steps: number of iterative decoding steps (K in paper)
        device: device to use
        temperature: sampling temperature
        action_dim: action dimension
        
    Returns:
        tokens: (1, T, H, W) generated token indices
    """
    model.eval()
    
    B = 1
    vocab_size = model.vocab_size
    mask_token_id = model.mask_token_id
    N = height * width
    
    # Initialize with all mask tokens
    tokens = torch.full((B, num_frames, height, width), mask_token_id, 
                        dtype=torch.long, device=device)
    
    # Create dummy actions (straight line motion)
    actions = torch.zeros(B, num_frames, action_dim, device=device)
    actions[:, :, 1] = 1.0  # down motion
    
    # Track which positions are still masked
    is_masked = torch.ones((B, num_frames, height, width), dtype=torch.bool, device=device)
    
    K = num_steps
    causal_mask = torch.full((num_frames, num_frames), float("-inf"), device=device)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    
    # Iterative decoding: Algorithm 2
    for step_idx in range(K):
        # Map to paper's k: first iteration = K-1, last = 0
        paper_k = K - 1 - step_idx
        
        # M = ceil(gamma(k/K) * N): tokens to keep unmasked after this step
        num_keep = math.ceil(cosine_schedule(paper_k / K) * N)
        
        # Forward pass
        logits = model(tokens, actions, causal_mask)
        # logits: (B, T, H*W, vocab_size+1)
        
        # Sample from predictions
        B, T, N_total, V = logits.shape
        logits_flat = logits.reshape(B * T * N_total, V)
        
        # Exclude mask token from sampling
        logits_flat[:, mask_token_id] = float('-inf')
        
        # Apply temperature
        logits_flat = logits_flat / temperature
        
        # Sample
        probs = F.softmax(logits_flat, dim=-1)
        sampled_flat = torch.multinomial(probs, num_samples=1).squeeze(-1)
        sampled = sampled_flat.reshape(B, T, N_total)
        
        # Compute confidence
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        
        # Add Gumbel noise scaled by paper_k/K
        gumbel_scale = paper_k / K if K > 0 else 0
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(selected_log_probs) + 1e-10) + 1e-10)
        confidence = selected_log_probs + gumbel_noise * gumbel_scale
        
        # On non-mask indices: confidence = +inf (keep unmasked)
        confidence[~is_masked.reshape(B, T, N_total)] = float('inf')
        
        # Update tokens with samples
        tokens_flat = tokens.reshape(B, T, N_total)
        tokens_flat = torch.where(is_masked.reshape(B, T, N_total), sampled, tokens_flat)
        
        if paper_k > 0:
            # Keep top-M confident predictions unmasked
            confidence_flat = confidence.reshape(B, T * N_total)
            _, top_indices = torch.topk(confidence_flat, k=num_keep, dim=-1)
            
            new_mask = torch.ones((B, T * N_total), dtype=torch.bool, device=device)
            new_mask.scatter_(1, top_indices, False)
            
            # Re-mask low confidence positions
            tokens_flat[new_mask] = mask_token_id
            is_masked = new_mask.reshape(B, T, N_total)
        else:
            # Last step: keep everything unmasked
            is_masked = torch.zeros((B, T, N_total), dtype=torch.bool, device=device)
        
        tokens = tokens_flat.reshape(B, T, height, width)
    
    return tokens


@torch.no_grad()
def predict_future(
    model,
    past_frames: torch.Tensor,
    future_actions: torch.Tensor,
    num_steps: int = 10,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Predict future frames given past context.
    
    Args:
        model: trained model
        past_frames: (B, T_past, H, W) past token indices
        future_actions: (B, T_future, action_dim) future actions
        num_steps: number of iterative decoding steps
        temperature: sampling temperature
        
    Returns:
        future_tokens: (B, T_future, H, W) predicted tokens
    """
    model.eval()
    
    B, T_past, H, W = past_frames.shape
    T_future = future_actions.shape[1]
    device = past_frames.device
    mask_token_id = model.mask_token_id
    N = H * W
    
    # Initialize future with mask tokens
    future_tokens = torch.full((B, T_future, H, W), mask_token_id, 
                               dtype=torch.long, device=device)
    
    # Create dummy past actions
    past_actions = torch.zeros(B, T_past, future_actions.shape[-1], device=device)
    
    # Combine
    all_tokens = torch.cat([past_frames, future_tokens], dim=1)
    all_actions = torch.cat([past_actions, future_actions], dim=1)
    T_total = T_past + T_future
    
    # Track masking
    is_masked = torch.zeros((B, T_total, H, W), dtype=torch.bool, device=device)
    is_masked[:, T_past:] = True
    
    K = num_steps
    causal_mask = torch.full((T_total, T_total), float("-inf"), device=device)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    
    # Iterative decoding
    for step_idx in range(K):
        paper_k = K - 1 - step_idx
        num_keep = math.ceil(cosine_schedule(paper_k / K) * N)
        
        logits = model(all_tokens, all_actions, causal_mask)
        
        # Only consider future positions
        future_logits = logits[:, T_past:]  # (B, T_future, H*W, V)
        
        # Sample for future
        B_curr, T_f, N_tok, V = future_logits.shape
        future_logits_flat = future_logits.reshape(B_curr * T_f * N_tok, V)
        future_logits_flat[:, mask_token_id] = float('-inf')
        future_logits_flat = future_logits_flat / temperature
        
        probs = F.softmax(future_logits_flat, dim=-1)
        sampled_flat = torch.multinomial(probs, num_samples=1).squeeze(-1)
        sampled = sampled_flat.reshape(B_curr, T_f, N_tok)
        
        # Confidence
        log_probs = F.log_softmax(future_logits, dim=-1)
        selected_log_probs = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        
        gumbel_scale = paper_k / K if K > 0 else 0
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(selected_log_probs) + 1e-10) + 1e-10)
        confidence = selected_log_probs + gumbel_noise * gumbel_scale
        
        confidence[~is_masked[:, T_past:].reshape(B_curr, T_f, N_tok)] = float('inf')
        
        # Update
        future_flat = all_tokens[:, T_past:].reshape(B_curr, T_f, N_tok)
        future_flat = torch.where(is_masked[:, T_past:].reshape(B_curr, T_f, N_tok), sampled, future_flat)
        
        if paper_k > 0:
            conf_flat = confidence.reshape(B_curr, T_f * N_tok)
            _, top_indices = torch.topk(conf_flat, k=num_keep, dim=-1)
            
            new_mask = torch.ones((B_curr, T_f * N_tok), dtype=torch.bool, device=device)
            new_mask.scatter_(1, top_indices, False)
            
            future_flat[new_mask] = mask_token_id
            is_masked[:, T_past:] = new_mask.reshape(B_curr, T_f, N_tok)
        else:
            is_masked[:, T_past:] = False
        
        all_tokens[:, T_past:] = future_flat.reshape(B_curr, T_f, H, W)
    
    return all_tokens[:, T_past:]


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Full-scale training of discrete diffusion on Moving MNIST")
    
    # Data
    parser.add_argument("--data_path", type=str, default="mnist_test_seq.1.npy")
    parser.add_argument("--num_train", type=int, default=8000)
    parser.add_argument("--num_val", type=int, default=2000)
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_token_levels", type=int, default=16)
    parser.add_argument("--frame_size", type=int, default=64)
    
    # Model (larger for high-quality generation)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--use_amp", action="store_true", default=True)
    
    # EMA
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    
    # Masking (EXACT from world model)
    parser.add_argument("--num_past_frames", type=int, default=10)
    parser.add_argument("--noise_eta", type=float, default=5.0)
    parser.add_argument("--prob_future_pred", type=float, default=0.5)
    parser.add_argument("--prob_joint_denoise", type=float, default=0.4)
    
    # Sampling during training
    parser.add_argument("--sample_interval", type=int, default=20)
    parser.add_argument("--num_sampling_steps", type=int, default=20)
    
    # Logging
    parser.add_argument("--output_dir", type=str, default="outputs/mnist_diffusion_full")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    
    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
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
    print("\n" + "="*60)
    print("Creating dataloaders...")
    print(f"  Train: {cfg.num_train} sequences")
    print(f"  Val: {cfg.num_val} sequences")
    print(f"  Frame size: {cfg.frame_size}x{cfg.frame_size}")
    print(f"  Sequence length: {cfg.seq_len}")
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
    print("\n" + "="*60)
    print("Creating model...")
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
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Create EMA
    ema = None
    if cfg.use_ema:
        ema = ModelEMA(model, decay=cfg.ema_decay)
        print(f"  EMA decay: {cfg.ema_decay}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    
    # Create masker (EXACT from world model)
    masker = DiscreteDiffusionMasker(
        vocab_size=cfg.num_token_levels,
        mask_token_id=cfg.num_token_levels,
        num_past_frames=cfg.num_past_frames,
        noise_eta=cfg.noise_eta,
        prob_future_pred=cfg.prob_future_pred,
        prob_joint_denoise=cfg.prob_joint_denoise,
    )
    
    # Create scaler for AMP
    scaler = GradScaler(enabled=cfg.use_amp)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if cfg.resume:
        print(f"\nResuming from {cfg.resume}...")
        checkpoint = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        if ema and "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        print(f"  Resumed from epoch {start_epoch-1}")
    
    # Create TensorBoard writer
    log_dir = output_dir / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir)
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print(f"  Epochs: {cfg.epochs}")
    print(f"  Learning rate: {cfg.lr}")
    print(f"  Gradient clipping: {cfg.grad_clip}")
    print(f"  Mixed precision: {cfg.use_amp}")
    print("="*60)
    
    global_step = 0
    best_val_loss = float("inf")
    
    for epoch in range(start_epoch, cfg.epochs + 1):
        # Train
        train_loss, global_step, obj_counts = train_epoch(
            model, train_loader, masker, optimizer, scaler, device, cfg, epoch, writer, global_step
        )
        
        # Update EMA
        if ema:
            ema.update(model)
        
        print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}")
        print(f"  Objectives: {obj_counts}")
        
        # Validate
        if epoch % 5 == 0 or epoch == cfg.epochs:
            # Validate with regular model
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
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                }
                if ema:
                    checkpoint["ema"] = ema.state_dict()
                torch.save(checkpoint, output_dir / "best_model.pt")
                print(f"  Saved best model (val_loss: {val_loss:.4f})")
        
        # Sample during training
        if epoch % cfg.sample_interval == 0:
            print("\n  Generating samples...")
            sample_model = ema.ema_model if ema else model
            
            # Generate from scratch
            gen_tokens = generate_video(
                sample_model,
                num_frames=cfg.seq_len,
                height=cfg.frame_size,
                width=cfg.frame_size,
                num_steps=cfg.num_sampling_steps,
                device=device,
                temperature=1.0,
            )
            
            # Get a batch for conditioning
            val_batch = next(iter(val_loader))
            past_frames = val_batch["tokens"][:1, :cfg.num_past_frames].to(device)
            future_actions = val_batch["actions"][:1, cfg.num_past_frames:].to(device)
            gt_future = val_batch["tokens"][:1, cfg.num_past_frames:].to(device)
            
            # Predict future
            pred_future = predict_future(
                sample_model,
                past_frames,
                future_actions,
                num_steps=cfg.num_sampling_steps,
                temperature=1.0,
            )
            
            # Save samples
            from tests.mnist_diffusion.analyze_samples import save_comparison_grid, tokens_to_video
            
            # Save generation
            gen_video = tokens_to_video(gen_tokens[0], levels=cfg.num_token_levels)
            save_comparison_grid(
                None, gen_video,
                str(output_dir / f"sample_gen_epoch{epoch}.png"),
                title=f"Generated (Epoch {epoch})"
            )
            
            # Save future prediction
            gt_full = torch.cat([past_frames[0], gt_future[0]], dim=0)
            pred_full = torch.cat([past_frames[0], pred_future[0]], dim=0)
            gt_video = tokens_to_video(gt_full, levels=cfg.num_token_levels)
            pred_video = tokens_to_video(pred_full, levels=cfg.num_token_levels)
            save_comparison_grid(
                gt_video, pred_video,
                str(output_dir / f"sample_pred_epoch{epoch}.png"),
                title=f"Future Prediction (Epoch {epoch})"
            )
            print(f"  Samples saved to {output_dir}")
        
        # Save checkpoint
        if epoch % cfg.save_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
            }
            if ema:
                checkpoint["ema"] = ema.state_dict()
            torch.save(checkpoint, output_dir / f"checkpoint_epoch{epoch}.pt")
            print(f"  Saved checkpoint at epoch {epoch}")
    
    writer.close()
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
