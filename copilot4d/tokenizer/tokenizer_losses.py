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


def depth_weighted_l1_loss(
    pred_depths: torch.Tensor, 
    gt_depths: torch.Tensor, 
    alpha: float = 0.5,
    epsilon: float = 1.0
) -> torch.Tensor:
    """Depth-weighted L1 loss: penalizes errors more for farther depths.
    
    Loss = |pred - gt| * (depth)^alpha
    
    This gives more weight to errors at farther depths when alpha > 0,
    which helps improve far-range accuracy (30-50m) where absolute errors
    are larger but relative errors matter more.
    
    With alpha=0.5:
    - 10m depth: weight = sqrt(10) ≈ 3.16
    - 50m depth: weight = sqrt(50) ≈ 7.07
    - Far field gets ~2.2x more gradient emphasis
    
    Args:
        pred_depths: (B, R) predicted depths
        gt_depths: (B, R) ground truth depths
        alpha: weighting exponent (0=uniform, 0.5=emphasizes far field, 1.0=linear)
        epsilon: small constant for numerical stability

    Returns:
        loss: scalar
    """
    abs_error = torch.abs(pred_depths - gt_depths)
    
    # Direct depth weighting: (depth)^alpha
    # This emphasizes far depths (30-50m) over near depths (0-10m)
    depth_clamped = torch.clamp(gt_depths, min=epsilon)
    weights = torch.pow(depth_clamped, alpha)
    
    # Normalize weights to have mean 1.0 (so magnitude is similar to unweighted)
    weights = weights / weights.mean()
    
    weighted_error = abs_error * weights
    return weighted_error.mean()


def relative_depth_loss(
    pred_depths: torch.Tensor, 
    gt_depths: torch.Tensor,
    epsilon: float = 1.0
) -> torch.Tensor:
    """Relative depth loss: |pred - gt| / gt
    
    This emphasizes percentage error rather than absolute meters.
    Better for far-range accuracy where absolute errors are naturally larger.
    
    Args:
        pred_depths: (B, R) predicted depths
        gt_depths: (B, R) ground truth depths
        epsilon: small constant for numerical stability

    Returns:
        loss: scalar (mean relative error)
    """
    abs_error = torch.abs(pred_depths - gt_depths)
    relative_error = abs_error / torch.clamp(gt_depths, min=epsilon)
    return relative_error.mean()


def combined_depth_loss(
    pred_depths: torch.Tensor,
    gt_depths: torch.Tensor,
    absolute_weight: float = 0.9,
    relative_weight: float = 0.1,
    alpha: float = 0.5,
    epsilon: float = 1.0
) -> Dict[str, torch.Tensor]:
    """Combined absolute and relative depth loss with normalized weights.
    
    L = w_abs * L_abs + w_rel * L_rel
    
    where:
        L_abs = |pred - gt| * (depth)^alpha  (depth-weighted absolute, emphasizes far field)
        L_rel = |pred - gt| / gt             (relative/percentage error)
    
    Weights are normalized to sum to 1.0 to maintain stable total loss magnitude
    during the ramp period (prevents VQ and surface concentration from destabilizing).
    
    Args:
        pred_depths: (B, R) predicted depths
        gt_depths: (B, R) ground truth depths
        absolute_weight: weight for absolute loss component (before normalization)
        relative_weight: weight for relative loss component (before normalization)
        alpha: depth weighting exponent for absolute component (0.5 = sqrt depth)
        epsilon: small constant for numerical stability

    Returns:
        dict with 'total', 'absolute', 'relative' loss components
    """
    # Compute individual losses
    abs_error = torch.abs(pred_depths - gt_depths)
    depth_clamped = torch.clamp(gt_depths, min=epsilon)
    
    # Absolute loss with direct depth weighting (emphasizes far field)
    if alpha > 0:
        weights = torch.pow(depth_clamped, alpha)  # Direct: depth^alpha
        weights = weights / weights.mean()  # Normalize
        abs_loss = (abs_error * weights).mean()
    else:
        abs_loss = abs_error.mean()
    
    # Relative loss (percentage error)
    rel_loss = (abs_error / depth_clamped).mean()
    
    # Normalize weights to sum to 1.0 (maintains stable total loss magnitude)
    total_weight = absolute_weight + relative_weight
    abs_w = absolute_weight / total_weight
    rel_w = relative_weight / total_weight
    
    # Combined loss with normalized weights
    total = abs_w * abs_loss + rel_w * rel_loss
    
    return {
        "total": total,
        "absolute": abs_loss,
        "relative": rel_loss,
        "abs_weight": abs_w,  # Return actual weights used
        "rel_weight": rel_w,
    }


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
    # New depth loss parameters
    depth_loss_type: str = "l1",
    depth_loss_alpha: float = 0.5,
    depth_loss_absolute_weight: float = 0.5,
    depth_loss_relative_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Compute total tokenizer loss as sum of all components.

    L = L_depth + L_surface_concentration + 1.0 * L_vq + L_skip_bce

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
        depth_loss_type: Type of depth loss ('l1', 'weighted', 'relative', 'combined')
        depth_loss_alpha: Depth weighting exponent (0=uniform, 0.5=balanced, 1.0=full)
        depth_loss_absolute_weight: Weight for absolute error component
        depth_loss_relative_weight: Weight for relative error component

    Returns:
        dict with 'total', 'depth_l1', 'depth_abs', 'depth_rel', 'surface_conc', 
        'vq', 'skip_bce', and optionally 'vq_codebook', 'vq_commitment'
    """
    # Compute depth loss based on type
    abs_weight_used = 1.0  # Default for non-combined modes
    rel_weight_used = 0.0
    
    if depth_loss_type == "l1":
        # Standard L1 loss (backward compatible)
        depth_l1 = l1_depth_loss(pred_depths, gt_depths)
        depth_loss_total = depth_l1
        depth_abs = depth_l1
        depth_rel = torch.tensor(0.0, device=pred_depths.device)
        
    elif depth_loss_type == "weighted":
        # Depth-weighted L1 loss
        depth_loss_total = depth_weighted_l1_loss(
            pred_depths, gt_depths, alpha=depth_loss_alpha
        )
        depth_l1 = depth_loss_total  # For backward compatibility
        depth_abs = depth_loss_total
        depth_rel = torch.tensor(0.0, device=pred_depths.device)
        
    elif depth_loss_type == "relative":
        # Relative (percentage) loss only
        depth_loss_total = relative_depth_loss(pred_depths, gt_depths)
        depth_l1 = l1_depth_loss(pred_depths, gt_depths)  # For logging
        depth_abs = depth_l1
        depth_rel = depth_loss_total
        
    elif depth_loss_type == "combined":
        # Combined absolute + relative loss with normalized weights
        combined = combined_depth_loss(
            pred_depths, gt_depths,
            absolute_weight=depth_loss_absolute_weight,
            relative_weight=depth_loss_relative_weight,
            alpha=depth_loss_alpha
        )
        depth_loss_total = combined["total"]
        depth_l1 = combined["absolute"]  # For backward compatibility
        depth_abs = combined["absolute"]
        depth_rel = combined["relative"]
        # Store actual normalized weights for monitoring
        abs_weight_used = combined["abs_weight"]
        rel_weight_used = combined["rel_weight"]
        
    else:
        raise ValueError(f"Unknown depth_loss_type: {depth_loss_type}")

    # Surface concentration loss
    surface_conc = surface_concentration_loss(
        weights, sample_depths, gt_depths, epsilon=surface_conc_eps
    )

    # Spatial skip BCE loss
    skip_bce = spatial_skip_bce_loss(skip_logits, gt_occupancy)

    # Total loss
    total = depth_loss_total + surface_conc + vq_weight * vq_loss + skip_bce

    result = {
        "total": total,
        "depth_l1": depth_l1,        # For backward compatibility
        "depth_abs": depth_abs,       # Absolute component
        "depth_rel": depth_rel,       # Relative component  
        "depth_abs_weight": abs_weight_used,  # Actual weight used (normalized)
        "depth_rel_weight": rel_weight_used,  # Actual weight used (normalized)
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

    def __init__(
        self, 
        surface_conc_eps: float = 1.0, 
        vq_weight: float = 1.0,
        depth_loss_type: str = "l1",
        depth_loss_alpha: float = 0.5,
        depth_loss_absolute_weight: float = 0.5,
        depth_loss_relative_weight: float = 0.5,
    ):
        super().__init__()
        self.surface_conc_eps = surface_conc_eps
        self.vq_weight = vq_weight
        self.depth_loss_type = depth_loss_type
        self.depth_loss_alpha = depth_loss_alpha
        self.depth_loss_absolute_weight = depth_loss_absolute_weight
        self.depth_loss_relative_weight = depth_loss_relative_weight

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
            self.depth_loss_type,
            self.depth_loss_alpha,
            self.depth_loss_absolute_weight,
            self.depth_loss_relative_weight,
        )
