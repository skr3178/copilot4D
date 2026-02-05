"""Evaluation metrics for CoPilot4D tokenizer.

Includes:
- Chamfer Distance (paper Section A.2.1): geometric similarity between point clouds
- Depth L1 error: per-point depth accuracy
- Point cloud reconstruction utilities
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def depths_to_points(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    depths: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert ray depths back to 3D points.
    
    Paper Equation (1): r(h) = o + h * d
    where o is ray origin, d is unit direction, h is depth.
    
    Args:
        ray_origins: (B, R, 3) ray origins (typically sensor at origin)
        ray_directions: (B, R, 3) unit direction vectors
        depths: (B, R) predicted or ground truth depths
        mask: (B, R) optional mask for valid depths
        
    Returns:
        points: (B, N, 3) reconstructed 3D points (N <= R after masking)
    """
    B, R, _ = ray_origins.shape
    
    # r(h) = o + h * d
    points = ray_origins + ray_directions * depths.unsqueeze(-1)  # (B, R, 3)
    
    if mask is not None:
        # Filter by mask
        filtered_points = []
        for b in range(B):
            valid_points = points[b, mask[b]]  # (N_b, 3)
            filtered_points.append(valid_points)
        return filtered_points  # List of (N_b, 3) tensors
    
    return points  # (B, R, 3)


def chamfer_distance(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    bidirectional: bool = True,
    batch_reduction: str = "mean",
) -> torch.Tensor:
    """Compute Chamfer Distance between two point clouds.
    
    Paper Section A.2.1: Chamfer distance is used for validation to measure
    geometric similarity between reconstructed and ground truth point clouds.
    
    Chamfer Distance:
        CD(S1, S2) = (1/|S1|) * Σ_{x∈S1} min_{y∈S2} ||x - y||_2
                   + (1/|S2|) * Σ_{y∈S2} min_{x∈S1} ||y - x||_2
    
    Args:
        pred_points: (B, N1, 3) or (N1, 3) predicted point cloud
        gt_points: (B, N2, 3) or (N2, 3) ground truth point cloud
        bidirectional: If True, compute symmetric CD (both directions)
        batch_reduction: "mean" or "sum" across batch
        
    Returns:
        cd: scalar Chamfer distance (mean per-point L2 distance)
    """
    # Handle single point cloud (no batch dimension)
    single_input = False
    if pred_points.dim() == 2:
        pred_points = pred_points.unsqueeze(0)
        gt_points = gt_points.unsqueeze(0)
        single_input = True
    
    B, N1, _ = pred_points.shape
    _, N2, _ = gt_points.shape
    
    # Compute pairwise distances: (B, N1, N2)
    # Using ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
    pred_sq = (pred_points ** 2).sum(dim=-1, keepdim=True)  # (B, N1, 1)
    gt_sq = (gt_points ** 2).sum(dim=-1, keepdim=True)  # (B, N2, 1)
    
    # (B, N1, 1) + (B, 1, N2) - 2 * (B, N1, N2)
    dist_matrix = pred_sq + gt_sq.transpose(1, 2) - 2 * torch.bmm(pred_points, gt_points.transpose(1, 2))
    dist_matrix = torch.clamp(dist_matrix, min=0.0)  # Avoid numerical errors
    
    # Forward direction: for each pred point, find nearest GT
    min_pred_to_gt, _ = torch.sqrt(dist_matrix).min(dim=2)  # (B, N1)
    forward_cd = min_pred_to_gt.mean(dim=1)  # (B,)
    
    if bidirectional:
        # Backward direction: for each GT point, find nearest pred
        min_gt_to_pred, _ = torch.sqrt(dist_matrix).min(dim=1)  # (B, N2)
        backward_cd = min_gt_to_pred.mean(dim=1)  # (B,)
        cd = (forward_cd + backward_cd) / 2  # (B,)
    else:
        cd = forward_cd  # (B,)
    
    # Reduce across batch
    if single_input:
        return cd[0]
    
    if batch_reduction == "mean":
        return cd.mean()
    elif batch_reduction == "sum":
        return cd.sum()
    else:
        return cd  # (B,)


def chamfer_distance_from_depths(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    pred_depths: torch.Tensor,
    gt_depths: torch.Tensor,
    depth_threshold: float = 100.0,
) -> Tuple[torch.Tensor, dict]:
    """Compute Chamfer Distance directly from depth predictions.
    
    Convenience function that:
    1. Converts depths to 3D points
    2. Filters out invalid depths (e.g., -1 for skipped rays)
    3. Computes Chamfer distance
    
    Args:
        ray_origins: (B, R, 3) ray origins
        ray_directions: (B, R, 3) ray directions
        pred_depths: (B, R) predicted depths
        gt_depths: (B, R) ground truth depths
        depth_threshold: Maximum valid depth value
        
    Returns:
        cd: scalar Chamfer distance
        info: dict with additional metrics (forward_cd, backward_cd, coverage)
    """
    B, R = pred_depths.shape
    
    # Create validity masks
    pred_valid = (pred_depths > 0) & (pred_depths < depth_threshold)
    gt_valid = (gt_depths > 0) & (gt_depths < depth_threshold)
    
    # Convert to points
    pred_points = ray_origins + ray_directions * pred_depths.unsqueeze(-1)  # (B, R, 3)
    gt_points = ray_origins + ray_directions * gt_depths.unsqueeze(-1)  # (B, R, 3)
    
    # Compute Chamfer distance per batch
    cds = []
    forward_cds = []
    backward_cds = []
    valid_batches = 0
    
    for b in range(B):
        pred_valid_b = pred_valid[b]
        gt_valid_b = gt_valid[b]
        
        # Skip if no valid points
        if pred_valid_b.sum() == 0 or gt_valid_b.sum() == 0:
            continue
        
        pred_pts = pred_points[b, pred_valid_b]  # (N1, 3)
        gt_pts = gt_points[b, gt_valid_b]  # (N2, 3)
        
        # Compute distances
        N1, N2 = pred_pts.shape[0], gt_pts.shape[0]
        
        # Pairwise distance matrix
        dist_matrix = torch.cdist(pred_pts, gt_pts)  # (N1, N2)
        
        # Forward: pred -> GT
        min_pred_to_gt = dist_matrix.min(dim=1)[0]  # (N1,)
        forward_cd = min_pred_to_gt.mean()
        
        # Backward: GT -> pred
        min_gt_to_pred = dist_matrix.min(dim=0)[0]  # (N2,)
        backward_cd = min_gt_to_pred.mean()
        
        # Symmetric CD
        cd = (forward_cd + backward_cd) / 2
        
        cds.append(cd)
        forward_cds.append(forward_cd)
        backward_cds.append(backward_cd)
        valid_batches += 1
    
    if valid_batches == 0:
        # Return dummy values if no valid data
        return torch.tensor(0.0, device=pred_depths.device), {
            "chamfer_forward": 0.0,
            "chamfer_backward": 0.0,
            "coverage_pred": 0.0,
            "coverage_gt": 0.0,
        }
    
    # Average across valid batches
    avg_cd = torch.stack(cds).mean()
    avg_forward = torch.stack(forward_cds).mean()
    avg_backward = torch.stack(backward_cds).mean()
    
    # Coverage metrics: what % of rays have valid predictions
    coverage_pred = pred_valid.float().mean().item()
    coverage_gt = gt_valid.float().mean().item()
    
    info = {
        "chamfer_forward": avg_forward.item(),
        "chamfer_backward": avg_backward.item(),
        "coverage_pred": coverage_pred,
        "coverage_gt": coverage_gt,
    }
    
    return avg_cd, info


def compute_depth_metrics(
    pred_depths: torch.Tensor,
    gt_depths: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> dict:
    """Compute comprehensive depth prediction metrics.
    
    Args:
        pred_depths: (B, R) predicted depths
        gt_depths: (B, R) ground truth depths
        valid_mask: (B, R) optional mask for valid depth pairs
        
    Returns:
        dict with metrics:
            - l1_error: mean absolute error (paper target: < 0.1m)
            - l2_error: root mean squared error
            - rel_error: mean relative error
            - delta_1: % of predictions with max(pred/gt, gt/pred) < 1.25
            - delta_2: % < 1.25^2
            - delta_3: % < 1.25^3
    """
    if valid_mask is None:
        # Use valid GT depths as mask
        valid_mask = gt_depths > 0
    
    # Mask out invalid depths
    pred_valid = pred_depths[valid_mask]
    gt_valid = gt_depths[valid_mask]
    
    if len(gt_valid) == 0:
        return {
            "depth_l1": 0.0,
            "depth_l2": 0.0,
            "depth_rel": 0.0,
            "depth_delta_1": 0.0,
            "depth_delta_2": 0.0,
            "depth_delta_3": 0.0,
            "depth_abs_rel": 0.0,
            "depth_rmse": 0.0,
            "depth_rmse_log": 0.0,
        }
    
    # Threshold for stability
    pred_valid = torch.clamp(pred_valid, min=1e-3)
    gt_valid = torch.clamp(gt_valid, min=1e-3)
    
    # Absolute error
    abs_diff = torch.abs(pred_valid - gt_valid)
    l1_error = abs_diff.mean()
    
    # RMSE
    sq_diff = (pred_valid - gt_valid) ** 2
    rmse = torch.sqrt(sq_diff.mean())
    
    # Relative error
    rel_error = (abs_diff / gt_valid).mean()
    
    # Log RMSE
    log_sq_diff = (torch.log(pred_valid) - torch.log(gt_valid)) ** 2
    rmse_log = torch.sqrt(log_sq_diff.mean())
    
    # Thresholded accuracy (delta metrics from Eigen et al.)
    ratio = torch.max(pred_valid / gt_valid, gt_valid / pred_valid)
    delta_1 = (ratio < 1.25).float().mean()
    delta_2 = (ratio < 1.25 ** 2).float().mean()
    delta_3 = (ratio < 1.25 ** 3).float().mean()
    
    return {
        "depth_l1": l1_error.item(),
        "depth_l2": torch.sqrt(sq_diff).mean().item(),
        "depth_rel": rel_error.item(),
        "depth_delta_1": delta_1.item(),
        "depth_delta_2": delta_2.item(),
        "depth_delta_3": delta_3.item(),
        "depth_abs_rel": rel_error.item(),
        "depth_rmse": rmse.item(),
        "depth_rmse_log": rmse_log.item(),
    }


class TokenizerMetrics:
    """Convenience class for computing all tokenizer metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.metrics = {
            "depth_l1": [],
            "depth_l2": [],
            "chamfer": [],
            "vq_perplexity": [],
            "vq_active_pct": [],
        }
    
    def update(
        self,
        pred_depths: torch.Tensor,
        gt_depths: torch.Tensor,
        vq_metrics: Optional[dict] = None,
        ray_origins: Optional[torch.Tensor] = None,
        ray_directions: Optional[torch.Tensor] = None,
    ):
        """Update metrics with a batch of predictions.
        
        Args:
            pred_depths: (B, R) predicted depths
            gt_depths: (B, R) ground truth depths
            vq_metrics: Optional dict with VQ metrics (perplexity, active_pct, etc.)
            ray_origins: (B, R, 3) required for Chamfer distance
            ray_directions: (B, R, 3) required for Chamfer distance
        """
        # Depth metrics
        depth_metrics = compute_depth_metrics(pred_depths, gt_depths)
        self.metrics["depth_l1"].append(depth_metrics["depth_l1"])
        self.metrics["depth_l2"].append(depth_metrics["depth_l2"])
        
        # Chamfer distance (if ray info provided)
        if ray_origins is not None and ray_directions is not None:
            cd, _ = chamfer_distance_from_depths(
                ray_origins, ray_directions, pred_depths, gt_depths
            )
            self.metrics["chamfer"].append(cd.item())
        
        # VQ metrics
        if vq_metrics is not None:
            if "vq_perplexity" in vq_metrics:
                self.metrics["vq_perplexity"].append(vq_metrics["vq_perplexity"])
            if "vq_active_pct" in vq_metrics:
                self.metrics["vq_active_pct"].append(vq_metrics["vq_active_pct"])
    
    def compute(self) -> dict:
        """Compute mean of all accumulated metrics."""
        result = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                result[key] = sum(values) / len(values)
            else:
                result[key] = 0.0
        return result
