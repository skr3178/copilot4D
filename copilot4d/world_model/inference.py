"""Discrete diffusion inference: Algorithm 2 with Classifier-Free Guidance.

Implements iterative decoding from CoPilot4D paper (Section 4.4, Appendix A.2.2).
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F

from copilot4d.utils.config import WorldModelConfig
from copilot4d.world_model.world_model import CoPilot4DWorldModel
from copilot4d.world_model.temporal_block import make_causal_mask, make_identity_mask


class WorldModelSampler:
    """Sampler for discrete diffusion inference with CFG.
    
    Implements Algorithm 2 from the paper:
    - Iterative denoising with K steps
    - Top-k sampling for x0 prediction
    - Confidence-based unmasking
    - Classifier-free guidance
    """

    def __init__(self, model: CoPilot4DWorldModel, cfg: WorldModelConfig):
        self.model = model
        self.cfg = cfg
        self.mask_token_id = cfg.mask_token_id
        self.num_steps = cfg.num_sampling_steps
        self.cfg_weight = cfg.cfg_weight
        self.temperature = cfg.choice_temperature
        
    @torch.no_grad()
    def predict_next_frame(
        self,
        past_tokens: torch.Tensor,
        past_actions: torch.Tensor,
        future_action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a single future frame given past context.
        
        Args:
            past_tokens: (B, T_past, H, W) ground truth past tokens
            past_actions: (B, T_past, 16) past actions
            future_action: (B, 1, 16) action for the frame to predict
            
        Returns:
            predicted_tokens: (B, H, W) predicted token indices
        """
        B, T_past, H, W = past_tokens.shape
        device = past_tokens.device
        
        # Combine actions
        actions = torch.cat([past_actions, future_action], dim=1)  # (B, T_past+1, 16)
        T_total = T_past + 1
        
        # Initialize with all mask tokens for future frame
        future_tokens = torch.full(
            (B, 1, H, W), self.mask_token_id, dtype=torch.long, device=device
        )
        tokens = torch.cat([past_tokens, future_tokens], dim=1)  # (B, T_total, H, W)
        
        # Track which positions are still masked
        is_masked = torch.zeros((B, T_total, H, W), dtype=torch.bool, device=device)
        is_masked[:, T_past:] = True  # Future frame starts fully masked
        
        N = H * W  # tokens per frame (per sample)
        K = self.num_steps

        # Iterative decoding: Algorithm 2
        # Paper loops k = K-1, ..., 0. We loop k = 0, ..., K-1 and map.
        for step_idx in range(K):
            # Map to paper's k: first iteration = K-1, last = 0
            paper_k = K - 1 - step_idx

            # M = ceil(gamma(k/K) * N): tokens to keep unmasked after this step
            # At first step (paper_k=K-1): M is small (few unmasked)
            # At last step (paper_k=0): M = N (all unmasked)
            num_keep = math.ceil(cosine_schedule(paper_k / K) * N)
            
            # Get logits with CFG
            logits = self._forward_with_cfg(tokens, actions, T_past)
            # logits: (B, T_total, H*W, vocab_size)
            
            # Focus on future frame logits
            future_logits = logits[:, T_past].reshape(B, N, -1)  # (B, N, vocab_size)

            # Sample x̃_0 from logits (Algorithm 2 step 3)
            sampled_tokens = torch.distributions.Categorical(logits=future_logits).sample()  # (B, N)

            # Compute confidence: l_k = log p(x̃_0 | x_{k+1}) + Gumbel * k/K
            # (Algorithm 2 step 4)
            log_probs = F.log_softmax(future_logits, dim=-1)
            selected_log_probs = log_probs.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)  # (B, N)

            # Add Gumbel noise scaled by paper_k/K
            gumbel_scale = paper_k / K
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(selected_log_probs) + 1e-10) + 1e-10)
            confidence = selected_log_probs + gumbel_noise * gumbel_scale  # (B, N)

            # On non-mask indices of x_{k+1}: l_k = +inf (Algorithm 2 step 5)
            current_mask_flat = is_masked[:, T_past].reshape(B, N)  # (B, N)
            confidence[~current_mask_flat] = float('inf')

            # Keep unmasked where predicted, re-mask rest
            # Replace masked positions with sampled tokens
            future_flat = tokens[:, T_past].reshape(B, N)
            future_flat = torch.where(current_mask_flat, sampled_tokens, future_flat)

            if paper_k > 0:
                # x_k = x̃_0 on top-M indices of l_k (Algorithm 2 step 7)
                # Top-M have highest confidence → keep unmasked; rest → re-mask
                _, top_indices = torch.topk(confidence, k=num_keep, dim=-1)  # (B, num_keep)
                new_mask = torch.ones((B, N), dtype=torch.bool, device=device)
                new_mask.scatter_(1, top_indices, False)

                # Re-mask low confidence positions
                future_flat[new_mask] = self.mask_token_id
                is_masked[:, T_past] = new_mask.reshape(B, H, W)
            else:
                # Last step: keep everything unmasked
                is_masked[:, T_past] = False

            tokens[:, T_past] = future_flat.reshape(B, H, W)
        
        return tokens[:, T_past:].squeeze(1)

    def _forward_with_cfg(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        num_past: int,
    ) -> torch.Tensor:
        """Forward pass with efficient single-pass classifier-free guidance.

        Paper (A.2.2, Figure 10): "CFG can be efficiently implemented with a
        single forward pass at each diffusion step by increasing temporal
        sequence length by 1, and setting the attention mask to be a causal
        mask within the previous sequence length, and an identity mask for the
        last frame."

        We append a duplicate of the future frame (position T) at position T+1,
        build a (T+1, T+1) mask where [0..T-1] is causal and [T] (the appended
        unconditional copy) only attends to itself.

        Args:
            tokens: (B, T, H, W)
            actions: (B, T, 16)
            num_past: number of past frames

        Returns:
            logits: (B, T, H*W, vocab_size) with CFG applied to future frame
        """
        B, T, H, W = tokens.shape
        device = tokens.device

        if self.cfg_weight == 0 or self.cfg_weight == 1.0:
            causal_mask = make_causal_mask(T, device)
            return self.model(tokens, actions, causal_mask)

        # Build extended sequence: [past_0, ..., past_{T-2}, future_cond, future_uncond]
        # The unconditional frame is a copy of the future frame
        future_frame = tokens[:, -1:, :, :]  # (B, 1, H, W)
        tokens_ext = torch.cat([tokens, future_frame], dim=1)  # (B, T+1, H, W)
        future_action = actions[:, -1:, :]  # (B, 1, 16)
        actions_ext = torch.cat([actions, future_action], dim=1)  # (B, T+1, 16)

        # Build (T+1, T+1) mask:
        # First T positions: standard causal mask (lower triangular)
        # Position T (unconditional): only attends to itself (identity row)
        T_ext = T + 1
        cfg_mask = torch.full((T_ext, T_ext), float("-inf"), device=device)
        # Causal block for first T frames
        cfg_mask[:T, :T] = make_causal_mask(T, device)
        # Unconditional frame (last row): only attend to itself
        cfg_mask[T, T] = 0.0

        # Single forward pass
        logits_ext = self.model(tokens_ext, actions_ext, cfg_mask)
        # logits_ext: (B, T+1, H*W, vocab_size)

        # Extract conditional (position T-1) and unconditional (position T) logits
        logits_cond = logits_ext[:, :T]  # (B, T, H*W, V)
        logits_uncond_frame = logits_ext[:, T]  # (B, H*W, V) — unconditional future

        # Apply CFG only to the future frame (last position of cond)
        logits_cond_future = logits_cond[:, -1]  # (B, H*W, V)
        logits_cfg_future = logits_cond_future + self.cfg_weight * (
            logits_cond_future - logits_uncond_frame
        )

        # Replace the future frame logits with CFG logits
        logits_out = logits_cond.clone()
        logits_out[:, -1] = logits_cfg_future

        return logits_out

    @torch.no_grad()
    def autoregressive_predict(
        self,
        past_tokens: torch.Tensor,
        past_actions: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict multiple future frames autoregressively.
        
        Args:
            past_tokens: (B, T_past, H, W)
            past_actions: (B, T_past, 16)
            future_actions: (B, T_future, 16)
            
        Returns:
            future_tokens: (B, T_future, H, W)
        """
        B, T_past, H, W = past_tokens.shape
        T_future = future_actions.shape[1]
        device = past_tokens.device
        
        all_future_tokens = []
        current_past_tokens = past_tokens
        current_past_actions = past_actions
        
        for t in range(T_future):
            # Predict next frame
            next_action = future_actions[:, t:t+1]
            pred_frame = self.predict_next_frame(
                current_past_tokens,
                current_past_actions,
                next_action,
            )
            all_future_tokens.append(pred_frame)
            
            # Update context for next iteration (slide window)
            # Keep last (num_past - 1) frames + new prediction
            if T_past > 1:
                current_past_tokens = torch.cat([
                    current_past_tokens[:, -T_past+1:],
                    pred_frame.unsqueeze(1),
                ], dim=1)
                current_past_actions = torch.cat([
                    current_past_actions[:, -T_past+1:],
                    next_action,
                ], dim=1)
            else:
                current_past_tokens = pred_frame.unsqueeze(1)
                current_past_actions = next_action
        
        return torch.stack(all_future_tokens, dim=1)


def cosine_schedule(t: float) -> float:
    """Cosine mask schedule: gamma(t) = cos(t * pi/2).
    
    Args:
        t: normalized step in [0, 1]
        
    Returns:
        Mask ratio in [0, 1]
    """
    return math.cos(t * math.pi / 2)
