"""TokenizerConfig: all spatial dimensions and hyperparameters for the VQVAE tokenizer."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TokenizerConfig:
    # --- ROI bounds (metres, Velodyne frame) ---
    x_min: float = -51.2
    x_max: float = 51.2
    y_min: float = -51.2
    y_max: float = 51.2
    z_min: float = -3.0
    z_max: float = 3.0

    # --- Voxel grid ---
    voxel_grid_xy: int = 512       # H = W of BEV grid
    voxel_grid_z: int = 32         # number of height bins
    max_points_per_voxel: int = 35

    # --- Encoder ---
    voxel_feat_dim: int = 16       # VoxelPointNet output dim
    bev_feat_dim: int = 64         # BEV pillar feature dim
    patch_size: int = 4            # PatchEmbed patch size
    enc_embed_dim: int = 128       # stage-1 Swin dim
    enc_stage1_depth: int = 2
    enc_stage1_heads: int = 8
    enc_stage2_dim: int = 256      # after PatchMerging
    enc_stage2_depth: int = 6
    enc_stage2_heads: int = 16
    window_size: int = 8
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.1
    use_checkpoint: bool = False

    # --- VQ ---
    vq_dim: int = 256              # dim going into / out of VQ (encoder output dim)
    vq_codebook_size: int = 1024
    vq_codebook_dim: int = 1024    # internal codebook vector dim
    vq_commitment_cost: float = 0.25
    vq_decay: float = 0.99
    vq_kmeans_init: bool = True
    vq_kmeans_iters: int = 10
    vq_threshold_ema_dead_code: int = 2

    # --- Decoder ---
    dec_stage1_depth: int = 6
    dec_stage1_heads: int = 16
    dec_stage2_depth: int = 2
    dec_stage2_heads: int = 8
    dec_output_dim: int = 128      # after PatchUpsample, stage-2 dim

    # --- NFG (Neural Feature Grid) ---
    nfg_feat_dim: int = 16
    nfg_upsample_factor: int = 2   # decoder output -> NFG spatial factor per dim
    nfg_z_bins: int = 32           # same as voxel_grid_z
    nfg_mlp_hidden: int = 32
    num_depth_samples: int = 64    # samples per ray for volume rendering

    # --- Rays ---
    rays_per_frame: int = 2048
    ray_chunk_size: int = 256

    # --- Spatial skip ---
    skip_upsample_factor: int = 4  # decoder output -> full BEV resolution

    # --- Loss ---
    surface_conc_eps: float = 1.0  # epsilon for surface concentration loss (metres)

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
    def voxel_size_xy(self) -> float:
        return (self.x_max - self.x_min) / self.voxel_grid_xy

    @property
    def voxel_size_z(self) -> float:
        return (self.z_max - self.z_min) / self.voxel_grid_z

    @property
    def token_grid_size(self) -> int:
        """Spatial size of token grid (after PatchEmbed + PatchMerging)."""
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
