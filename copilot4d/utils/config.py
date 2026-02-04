"""TokenizerConfig: all spatial dimensions and hyperparameters for the VQVAE tokenizer.

This configuration follows the CoPilot4D paper (Section 4.1, Appendix A.2.1):
- ROI: [-80m, 80m] x [-80m, 80m] x [-4.5m, 4.5m]
- Voxel size: 15.625cm x 15.625cm x 14.0625cm
- Grid: 1024 x 1024 x 64 voxels
- Feature dim: 64 (after PointNet)
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TokenizerConfig:
    # --- ROI bounds (metres, Velodyne frame) ---
    # Paper: model 3D world in [-80m, 80m] x [-80m, 80m] x [-4.5m, 4.5m]
    x_min: float = -80.0
    x_max: float = 80.0
    y_min: float = -80.0
    y_max: float = 80.0
    z_min: float = -4.5
    z_max: float = 4.5

    # --- Voxel grid ---
    # Paper: voxel size 15.625cm x 15.625cm x 14.0625cm
    # 160m / 0.15625m = 1024 (x, y)
    # 9m / 0.140625m = 64 (z)
    voxel_grid_xy: int = 1024     # H = W of 3D voxel grid
    voxel_grid_z: int = 64        # number of height bins (z)
    max_points_per_voxel: int = 35
    
    # Voxel sizes (meters) - Paper: 15.625cm x 15.625cm x 14.0625cm
    # Computed from ROI / grid, but stored for reference
    # voxel_size_x = 160m / 1024 = 0.15625m (15.625cm)
    # voxel_size_y = 160m / 1024 = 0.15625m (15.625cm)
    # voxel_size_z = 9m / 64 = 0.140625m (14.0625cm)

    # --- Encoder ---
    # Paper: PointNet outputs 16-dim features per voxel
    # After BEV pooling with z-embedding, projected to 64-dim
    voxel_feat_dim: int = 16       # VoxelPointNet output dim (paper: 16)
    bev_feat_dim: int = 64         # BEV pillar feature dim (after z-pooling + projection)
    patch_size: int = 4            # PatchEmbed patch size
    enc_embed_dim: int = 128       # stage-1 Swin dim
    enc_stage1_depth: int = 2
    enc_stage1_heads: int = 8
    enc_stage2_dim: int = 256      # after PatchMerging
    enc_stage2_depth: int = 6
    enc_stage2_heads: int = 16  # Paper: 16 heads (text overrides diagram)
    window_size: int = 8
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.1
    use_checkpoint: bool = False

    # --- VQ ---
    vq_dim: int = 256              # dim going into / out of VQ (encoder output dim)
    vq_codebook_size: int = 1024
    vq_codebook_dim: int = 1024    # internal codebook vector dim
    vq_commitment_cost: float = 0.25  # Paper: lambda_1 = 0.25
    vq_codebook_cost: float = 1.0     # Paper: lambda_2 = 1.0
    vq_kmeans_iters: int = 10
    vq_dead_threshold: int = 256      # iterations before code is "dead"
    vq_dead_percentage: float = 0.03  # 3% threshold for re-init
    vq_min_iterations: int = 200      # min iterations before re-init


    # --- Decoder ---
    dec_stage1_depth: int = 6
    dec_stage1_heads: int = 8
    dec_stage2_depth: int = 2
    dec_stage2_heads: int = 8
    dec_output_dim: int = 128      # after PatchUpsample, stage-2 dim

    # --- NFG (Neural Feature Grid) ---
    nfg_feat_dim: int = 16
    nfg_upsample_factor: int = 2   # decoder output -> NFG spatial factor per dim
    nfg_z_bins: int = 64           # same as voxel_grid_z
    nfg_mlp_hidden: int = 32  # Paper: hidden dimension is 32
    num_depth_samples: int = 64    # samples per ray for volume rendering

    # --- Rays ---
    rays_per_frame: int = 2048
    ray_chunk_size: int = 256

    # --- Spatial skip ---
    skip_upsample_factor: int = 4  # decoder output -> full BEV resolution

    # --- Loss ---
    surface_conc_eps: float = 0.4  # Paper: margin epsilon = 0.4 meters

    # --- Data ---
    kitti_root: str = "data/kitti/pykitti"
    train_sequences: List[str] = field(
        default_factory=lambda: [f"{i:02d}" for i in range(9)]
    )
    val_sequences: List[str] = field(default_factory=lambda: ["09"])
    test_sequences: List[str] = field(default_factory=lambda: ["10"])

    # --- Training ---
    batch_size: int = 4
    grad_accum_steps: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    amp: bool = True

    # --- Checkpointing ---
    save_every_steps: int = 5000
    eval_every_steps: int = 1000
    num_eval_batches: int = 10
    output_dir: str = "outputs/tokenizer"

    # --- Derived properties ---
    @property
    def voxel_size_x(self) -> float:
        """Voxel size in x dimension (15.625cm)."""
        return (self.x_max - self.x_min) / self.voxel_grid_xy  # 0.15625m

    @property
    def voxel_size_y(self) -> float:
        """Voxel size in y dimension (15.625cm)."""
        return (self.y_max - self.y_min) / self.voxel_grid_xy  # 0.15625m

    @property
    def voxel_size_z(self) -> float:
        """Voxel size in z dimension (14.0625cm)."""
        return (self.z_max - self.z_min) / self.voxel_grid_z  # 0.140625m

    @property
    def voxel_size_xy(self) -> float:
        """Voxel size in x and y dimensions (15.625cm)."""
        return self.voxel_size_x

    @property
    def token_grid_size(self) -> int:
        """Spatial size of token grid (after PatchEmbed + PatchMerging)."""
        # After PatchEmbed (patch_size=4): 1024/4 = 256
        # After PatchMerging (2x down): 256/2 = 128
        return self.voxel_grid_xy // (self.patch_size * 2)

    @property
    def decoder_output_grid_size(self) -> int:
        """Spatial size of decoder output (after PatchUpsample)."""
        return self.token_grid_size * 2

    @property
    def nfg_spatial_size(self) -> int:
        """Spatial H=W of NFG."""
        return self.decoder_output_grid_size * self.nfg_upsample_factor

    @property
    def num_tokens(self) -> int:
        return self.token_grid_size ** 2

    @property
    def skip_grid_xy(self) -> int:
        """Full BEV resolution for spatial skip."""
        return self.decoder_output_grid_size * self.skip_upsample_factor

    @property
    def ray_depth_min(self) -> float:
        return 1.0

    @property
    def ray_depth_max(self) -> float:
        return ((self.x_max - self.x_min) ** 2 + (self.y_max - self.y_min) ** 2) ** 0.5 / 2.0
