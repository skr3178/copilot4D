## Plan

cat ~/.claude/plans/giggly-hugging-cat.md 

## DEBUG NOTES:
Key Findings

  1. ROI Filtering (Fig 1)

  • Original: 124,668 points extending to ±80m
  • Filtered: 123,230 points within ±51.2m (x,y) and ±3m (z)
  • The red dashed box shows the ROI boundaries

  2. Voxelization (Fig 2)

  • Only 18,968 voxels occupied out of 512×512×32 = 8,388,608 possible
  • This extreme sparsity (0.23%) is why pillar-based BEV representation is efficient
  • You can see the road structure, buildings, and trees in the voxel grid

  3. Height Distribution (Fig 3)

  • Clear peak at -1.71m = ground plane
  • This matches KITTI's Velodyne mount height (~1.7m above ground)
  • Points below -1m: 68.4% (mostly ground)
  • Points above -1m: 31.6% (objects, vegetation)

  4. Points per Voxel (Fig 4)

  • Mean: 5.79 points per voxel
  • Most voxels (5,200+) have only 1-2 points
  • 455 voxels are "full" with 35 points (max capacity)\


 Updated Architecture:

  [dx, dy, dz, reflectance] (4-dim)
      ↓
  Linear(4, 16) → LayerNorm(16) → ReLU
      ↓
  Linear(16, 16)  ← No ReLU (can output negative values)
      ↓
  Sum pooling
      ↓
  LayerNorm(16)
      ↓
  Output: (V, 16)