# World Model Point Cloud Evaluation - Implementation Plan

## Overview
Implement full paper-matching evaluation metrics:
- **Chamfer Distance** (primary metric)
- **L1 Med/Mean** depth error
- **AbsRel** (absolute relative depth error)

---

## Phase 1: Token → Point Cloud Decoding

### 1.1 Token to BEV Features
```python
def tokens_to_nfg(tokenizer, tokens):
    """
    Input: tokens (H, W) with values 0-1023
    
    Process:
    1. Flatten tokens to (1, H*W)
    2. Lookup codebook: embed[tokens] → (1, N, codebook_dim)
    3. Post-project: (1, N, encoder_dim)
    4. Decode through tokenizer.decoder → (1, dec_grid^2, dec_output_dim)
    5. Build NFG → (1, F, Z, H, W)
    
    Output: NFG tensor (1, F, Z, H, W)
    """
```

### 1.2 NFG to Point Cloud via Raycasting
```python
def nfg_to_pointcloud(nfg, pose, num_rays=1024, max_depth=20.0):
    """
    Input: 
    - NFG: (1, F, Z, H, W) neural feature grid
    - pose: (4, 4) camera/ego pose
    - num_rays: number of rays to cast
    - max_depth: maximum depth (20m for our tokenizer)
    
    Process:
    1. Generate ray origins (all at ego center)
    2. Generate ray directions (evenly spaced in 360° or front-facing)
    3. Query NFG via volume rendering → depths (1, num_rays)
    4. Convert depth + direction to 3D points:
       point = origin + direction * depth
    5. Transform to world coordinates using pose
    
    Output: points (num_rays, 3) in world coordinates
    """
```

**Key Implementation Details:**
- Use `tokenizer.nfg.query_rays()` for volume rendering
- Ray directions: can use spherical coordinates or uniform sampling
- Filter out invalid depths (max_depth or beyond)

---

## Phase 2: Chamfer Distance

### 2.1 Bidirectional Nearest Neighbor
```python
def chamfer_distance(pred_points, gt_points):
    """
    Input:
    - pred_points: (N, 3) predicted point cloud
    - gt_points: (M, 3) ground truth point cloud
    
    Process:
    1. pred_to_gt: for each pred point, find nearest GT point
       distances_pred = min_dist(pred[i], gt) for all i
    2. gt_to_pred: for each GT point, find nearest pred point
       distances_gt = min_dist(gt[j], pred) for all j
    3. Chamfer = mean(distances_pred) + mean(distances_gt)
    
    Output: scalar Chamfer Distance (meters)
    """
```

**Implementation Notes:**
- Use `scipy.spatial.cKDTree` for efficient nearest neighbor
- Handle empty point clouds (return large distance)
- Crop to ROI before computing (for fair comparison)

---

## Phase 3: Depth Metrics

### 3.1 Ray-based Depth Comparison
```python
def compute_depth_metrics(pred_points, gt_points, ray_origins, ray_directions):
    """
    Input:
    - pred_points: (N, 3) predicted
    - gt_points: (M, 3) ground truth
    - ray_origins: (R, 3) rays to cast
    - ray_directions: (R, 3) ray directions
    
    Process:
    1. For each ray:
       - Project pred_points onto ray → pred_depth
       - Project gt_points onto ray → gt_depth
       - Compute L1 error = |pred_depth - gt_depth|
    2. Aggregate statistics:
       - L1 Med = median(errors)
       - L1 Mean = mean(errors)
       - AbsRel Med = median(errors / gt_depth) * 100
       - AbsRel Mean = mean(errors / gt_depth) * 100
    
    Output: dict with metrics
    """
```

**Alternative (simpler):**
Since we already have ray depths from tokenizer rendering, we can compare directly:
```python
# If using tokenizer's volume rendering
pred_depths = tokenizer.decode(tokens, ray_origins, ray_directions)["pred_depths"]
gt_depths = tokenizer.decode(gt_tokens, ray_origins, ray_directions)["pred_depths"]

# Compute metrics directly on depths
l1_errors = torch.abs(pred_depths - gt_depths)
l1_med = torch.median(l1_errors)
l1_mean = torch.mean(l1_errors)
absrel = l1_errors / gt_depths * 100
```

---

## Phase 4: ROI Handling

### 4.1 Crop to Region of Interest
```python
def crop_to_roi(points, roi_bounds):
    """
    Default ROI (paper specification):
    - x: [-70m, +70m]
    - y: [-70m, +70m]
    - z: [-4.5m, +4.5m]
    
    Our tokenizer (20m range):
    - x: [-20m, +20m]
    - y: [-20m, +20m]
    - z: [-4.5m, +4.5m]
    
    Process: Filter points within bounds
    """
    mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
           (points[:, 1] >= y_min) & (points[:, 1] <= y_max) & \
           (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    return points[mask]
```

---

## Phase 5: Unified Evaluation Script

### 5.1 Main Evaluation Flow
```python
def evaluate_world_model(checkpoint, sequence, start_frame, num_future):
    # 1. Load models
    world_model, tokenizer = load_models(checkpoint)
    
    # 2. Load data
    past_tokens, gt_future_tokens, poses = load_data(sequence, start_frame)
    
    # 3. Predict future
    pred_tokens = predict_future(world_model, past_tokens, num_future)
    
    # 4. Decode to point clouds
    all_metrics = []
    for i in range(num_future):
        # Decode tokens to point clouds
        pred_pc = decode_tokens_to_pointcloud(tokenizer, pred_tokens[i], poses[i])
        gt_pc = decode_tokens_to_pointcloud(tokenizer, gt_future_tokens[i], poses[i])
        
        # Crop to ROI
        pred_pc = crop_to_roi(pred_pc, ROI_BOUNDS)
        gt_pc = crop_to_roi(gt_pc, ROI_BOUNDS)
        
        # Compute metrics
        cd = chamfer_distance(pred_pc, gt_pc)
        depth_metrics = compute_depth_metrics(pred_pc, gt_pc, poses[i])
        
        all_metrics.append({
            'chamfer_distance': cd,
            'l1_med': depth_metrics['l1_med'],
            'l1_mean': depth_metrics['l1_mean'],
            'absrel_med': depth_metrics['absrel_med'],
            'absrel_mean': depth_metrics['absrel_mean'],
        })
    
    return all_metrics
```

---

## Implementation Details

### Key Challenges:

1. **Ray Generation**: Need to match the ray pattern used during tokenizer training
   - Check tokenizer config for `num_depth_samples`, ray sampling strategy
   - May need to load calibration from KITTI dataset

2. **Coordinate Systems**: 
   - Tokens are in BEV grid coordinates
   - NFG is in voxel coordinates
   - Point clouds need to be in world coordinates
   - Need proper pose transformations

3. **Efficiency**:
   - Chamfer Distance with k-d trees: O(N log M)
   - For 1000 points each: very fast
   - For 10000+ points: may need downsampling

### Libraries Needed:
```python
import torch
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
```

### Files to Modify/Create:

| File | Purpose |
|------|---------|
| `scripts/evaluate_world_model_pc.py` | New evaluation script |
| `copilot4d/utils/point_cloud_utils.py` | Chamfer Distance, point cloud ops |
| `copilot4d/utils/ray_utils.py` | Ray generation, depth computation |

---

## Expected Outputs

### Metrics Format:
```yaml
Frame 0:
  chamfer_distance: 0.35m
  l1_med: 0.12m
  l1_mean: 0.85m
  absrel_med: 1.1%
  absrel_mean: 9.2%
  
Frame 1:
  chamfer_distance: 0.42m
  ...

Average:
  chamfer_distance: 0.38m
  ...
```

### Visualizations:
1. **Point cloud overlay**: Pred (red) vs GT (green) 3D scatter
2. **Error heatmap**: Per-point Chamfer Distance color-coded
3. **Depth error plot**: Predicted vs GT depth scatter

---

## Timeline Estimate

| Phase | Time |
|-------|------|
| Phase 1: Token→Point Cloud | 2-3 hours |
| Phase 2: Chamfer Distance | 1 hour |
| Phase 3: Depth Metrics | 1-2 hours |
| Phase 4: ROI & Integration | 1 hour |
| Phase 5: Testing & Validation | 2 hours |
| **Total** | **7-9 hours** |

---

## Open Questions

1. **Ray sampling**: Should we use the same ray pattern as tokenizer training?
2. **Point density**: How many rays to cast? (Paper may use 1024 or 2048)
3. **Ground truth**: Should we use raw LiDAR or tokenizer-reconstructed GT?
   - Raw LiDAR: true ground truth
   - Tokenizer-reconstructed: matches what model can achieve

**Recommendation**: Compare against both for completeness.
