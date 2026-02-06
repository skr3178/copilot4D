"""Discrete diffusion masking: Algorithm 1 + 3 training objectives.

Implements the improved MaskGIT discrete diffusion training from CoPilot4D.
"""

import math
import random
from typing import Tuple, Dict

import torch
import torch.nn.functional as F

from copilot4d.utils.config import WorldModelConfig


def cosine_mask_schedule(u: torch.Tensor) -> torch.Tensor:
    """Mask schedule: gamma(u) = cos(u * pi/2).
    
    Args:
        u: float tensor in [0, 1]
        
    Returns:
        Mask ratio in [0, 1]
    """
    return torch.cos(u * math.pi / 2)


class DiscreteDiffusionMasker:
    """Prepares masked batches for discrete diffusion training.
    
    Implements Algorithm 1 from the paper with three training objectives:
    1. Future prediction (50%): past=GT, future=masked+noised
    2. Joint denoise (40%): all frames partially masked+noised
    3. Individual denoise (10%): each frame independently masked, identity temporal mask
    """

    def __init__(self, cfg: WorldModelConfig):
        self.cfg = cfg
        self.mask_token_id = cfg.mask_token_id
        self.vocab_size = cfg.codebook_size  # 1024, excluding mask

    def sample_objective(self) -> str:
        """Sample training objective according to paper probabilities."""
        r = random.random()
        if r < self.cfg.prob_future_pred:
            return "future_prediction"
        elif r < self.cfg.prob_future_pred + self.cfg.prob_joint_denoise:
            return "joint_denoise"
        else:
            return "individual_denoise"

    def apply_random_masking(
        self,
        tokens: torch.Tensor,
        mask_ratio: float,
        noise_ratio: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random masking and optional noise to tokens.
        
        Algorithm 1 steps 1-2:
        1. Mask ceil(gamma(u)*N) tokens with mask_token_id
        2. Add random noise to floor(eta% * unmasked) tokens
        
        Args:
            tokens: (N,) or (T, H, W) discrete token indices
            mask_ratio: fraction of tokens to mask [0, 1]
            noise_ratio: fraction of unmasked to replace with random noise [0, 1]
            
        Returns:
            masked_tokens: same shape as input with mask_token_id applied
            was_masked: bool tensor same shape, True where mask was applied
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
            Dict with:
                - tokens: (B, T, H, W) masked/noised tokens
                - targets: (B, T, H, W) original tokens (for loss computation)
                - temporal_mask: (T, T) attention mask
                - objective: str name of objective used
                - was_masked: (B, T, H, W) bool tensor of masked positions
        """
        B, T, H, W = tokens.shape
        device = tokens.device
        
        # Sample objective if not specified
        if objective is None:
            objective = self.sample_objective()
        
        num_past = self.cfg.num_past_frames
        
        # Prepare output tensors
        out_tokens = tokens.clone()
        temporal_mask = None
        was_masked = torch.zeros_like(tokens, dtype=torch.bool)
        
        # Sample mask ratio: u0 ~ Uniform(0,1), ratio = gamma(u0)
        u0 = random.random()
        mask_ratio = cosine_mask_schedule(torch.tensor(u0)).item()

        # Sample noise ratio: u1 ~ Uniform(0,1), noise = u1 * eta%
        # (Algorithm 1 steps 4-5)
        u1 = random.random()
        noise_ratio = u1 * self.cfg.noise_eta / 100.0

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
    
    Loss is computed on ALL positions (masked and unmasked) as per the paper.
    
    Args:
        logits: (B, T, H*W, vocab_size) model predictions
        targets: (B, T, H, W) ground truth token indices
        label_smoothing: label smoothing factor
        
    Returns:
        Scalar loss tensor
    """
    B, T, N, V = logits.shape
    
    # Flatten targets
    targets_flat = targets.reshape(B, T, N)
    
    # Reshape logits for F.cross_entropy
    # From (B, T, N, V) to (B*T*N, V)
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets_flat.reshape(-1)
    
    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        label_smoothing=label_smoothing,
    )
    
    return loss
