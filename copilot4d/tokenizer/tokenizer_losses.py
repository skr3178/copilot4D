"""Loss functions for the CoPilot4D tokenizer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


def l1_depth_loss(pred_depths: torch.Tensor, gt_depths: torch.Tensor) -> torch.Tensor:
    """L1 depth loss between predicted and ground truth depths.

    Args:
        pred_depths: (B, R) predicted depths
        gt_depths: (B, R) ground truth depths

    Returns:
        loss: scalar
    """
    return torch.abs(pred_depths - gt_depths).mean()


def surface_concentration_loss(
    weights: torch.Tensor,
    sample_depths: torch.Tensor,
    gt_depths: torch.Tensor,
    epsilon: float = 0.4,
) -> torch.Tensor:
    """Surface concentration loss: penalize weights far from surface.

    Paper Equation (5):
        L_conc = (1/(B*R)) * Σᵢ wᵢ² · 𝟙(|hᵢ - D_gt| > ε)

    where 𝟙(·) is the indicator function:
        𝟙(|hᵢ - D_gt| > ε) = 1 if sample i is farther than ε from true surface
        𝟙(|hᵢ - D_gt| > ε) = 0 otherwise (within ε of surface)

    This encourages the model to concentrate weights near the true surface.

    Args:
        weights: (B, R, S) volume rendering weights from NFG
        sample_depths: (B, R, S) depth samples along each ray
        gt_depths: (B, R) ground truth depths
        epsilon: distance threshold in meters (paper: ε = 0.4m)

    Returns:
        loss: scalar
    """
    B, R, S = weights.shape

    # Expand gt_depths to match sample_depths
    gt_depths_expanded = gt_depths.unsqueeze(-1)  # (B, R, 1)

    # Indicator function: 𝟙(|hᵢ - D_gt| > ε)
    # Returns 1.0 if sample is farther than epsilon from surface, 0.0 otherwise
    indicator = (torch.abs(sample_depths - gt_depths_expanded) > epsilon).float()

    # Surface concentration loss: (1/(B*R)) * Σᵢ wᵢ² · 𝟙(|hᵢ - D_gt| > ε)
    loss = (weights ** 2 * indicator).sum() / (B * R)

    return loss


def spatial_skip_bce_loss(
    skip_logits: torch.Tensor,
    gt_occupancy: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy loss for spatial skip occupancy prediction.

    Args:
        skip_logits: (B, H, W, Z) logits (before sigmoid)
        gt_occupancy: (B, H, W, Z) binary occupancy (0 or 1)

    Returns:
        loss: scalar
    """
    # Flatten for BCE
    B, H, W, Z = skip_logits.shape
    logits_flat = skip_logits.reshape(-1)
    gt_flat = gt_occupancy.reshape(-1)

    # BCE with logits (numerically stable)
    loss = F.binary_cross_entropy_with_logits(logits_flat, gt_flat)

    return loss


def tokenizer_total_loss(
    pred_depths: torch.Tensor,
    gt_depths: torch.Tensor,
    weights: torch.Tensor,
    sample_depths: torch.Tensor,
    vq_loss: torch.Tensor,
    skip_logits: torch.Tensor,
    gt_occupancy: torch.Tensor,
    surface_conc_eps: float = 1.0,
    vq_weight: float = 1.0,
    vq_loss_codebook: Optional[torch.Tensor] = None,
    vq_loss_commitment: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute total tokenizer loss as sum of all components.

    L = L_depth_l1 + L_surface_concentration + 1.0 * L_vq + L_skip_bce

    Args:
        pred_depths: (B, R) predicted depths from NFG
        gt_depths: (B, R) ground truth depths
        weights: (B, R, S) volume rendering weights
        sample_depths: (B, R, S) depth samples along rays
        vq_loss: scalar VQ commitment loss
        skip_logits: (B, H, W, Z) spatial skip logits
        gt_occupancy: (B, H, W, Z) ground truth occupancy
        surface_conc_eps: epsilon for surface concentration loss
        vq_weight: weight for VQ loss (default 1.0)
        vq_loss_codebook: Optional scalar codebook loss component (for logging)
        vq_loss_commitment: Optional scalar commitment loss component (for logging)

    Returns:
        dict with 'total', 'depth_l1', 'surface_conc', 'vq', 'skip_bce', 
        and optionally 'vq_codebook', 'vq_commitment'
    """
    # L1 depth loss
    depth_l1 = l1_depth_loss(pred_depths, gt_depths)

    # Surface concentration loss
    surface_conc = surface_concentration_loss(
        weights, sample_depths, gt_depths, epsilon=surface_conc_eps
    )

    # Spatial skip BCE loss
    skip_bce = spatial_skip_bce_loss(skip_logits, gt_occupancy)

    # Total loss
    total = depth_l1 + surface_conc + vq_weight * vq_loss + skip_bce

    result = {
        "total": total,
        "depth_l1": depth_l1,
        "surface_conc": surface_conc,
        "vq": vq_loss,
        "skip_bce": skip_bce,
    }
    
    # Add VQ component losses for detailed monitoring
    if vq_loss_codebook is not None:
        result["vq_codebook"] = vq_loss_codebook
    if vq_loss_commitment is not None:
        result["vq_commitment"] = vq_loss_commitment

    return result


class TokenizerLoss(nn.Module):
    """Convenience module that computes all tokenizer losses."""

    def __init__(self, surface_conc_eps: float = 1.0, vq_weight: float = 1.0):
        super().__init__()
        self.surface_conc_eps = surface_conc_eps
        self.vq_weight = vq_weight

    def forward(
        self,
        pred_depths: torch.Tensor,
        gt_depths: torch.Tensor,
        weights: torch.Tensor,
        sample_depths: torch.Tensor,
        vq_loss: torch.Tensor,
        skip_logits: torch.Tensor,
        gt_occupancy: torch.Tensor,
        vq_loss_codebook: Optional[torch.Tensor] = None,
        vq_loss_commitment: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses.

        Args:
            pred_depths: (B, R) predicted depths
            gt_depths: (B, R) ground truth depths
            weights: (B, R, S) volume rendering weights
            sample_depths: (B, R, S) depth samples
            vq_loss: scalar VQ loss
            skip_logits: (B, H, W, Z) spatial skip logits
            gt_occupancy: (B, H, W, Z) GT occupancy
            vq_loss_codebook: Optional scalar codebook loss component
            vq_loss_commitment: Optional scalar commitment loss component

        Returns:
            dict with loss components
        """
        return tokenizer_total_loss(
            pred_depths,
            gt_depths,
            weights,
            sample_depths,
            vq_loss,
            skip_logits,
            gt_occupancy,
            self.surface_conc_eps,
            self.vq_weight,
            vq_loss_codebook,
            vq_loss_commitment,
        )
