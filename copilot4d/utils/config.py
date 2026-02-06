"""Configuration dataclasses for CoPilot4D.

TokenizerConfig: VQVAE tokenizer (Section 4.1, Appendix A.2.1)
WorldModelConfig: U-Net Spatio-Temporal Transformer world model (Section 4.4, Appendix A.2.2)
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
    # Paper: L_vq = lambda_1 * ||sg[E(o)] - z_hat||^2 + lambda_2 * ||sg[z_hat] - E(o)||^2
    #   lambda_1 = 0.25 (codebook update) - codebook_cost
    #   lambda_2 = 1.0 (commitment) - commitment_cost
    vq_commitment_cost: float = 1.0   # lambda_2: encoder commits to codebook
    vq_codebook_cost: float = 0.25    # lambda_1: codebook moves to encoder outputs
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
    save_every_steps: int = 2000
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


@dataclass
class WorldModelConfig:
    """World model config (Section 4.4, Appendix A.2.2).

    U-Net Spatio-Temporal Transformer operating on discrete BEV tokens
    from the frozen tokenizer. Uses discrete diffusion (improved MaskGIT).
    """

    # --- Tokenizer interface ---
    codebook_size: int = 1024
    token_grid_h: int = 128
    token_grid_w: int = 128

    # --- Sequence ---
    num_frames: int = 6              # 3 past + 3 future for initial dev
    num_past_frames: int = 3

    # --- U-Net architecture ---
    level_dims: Tuple[int, ...] = (256, 384, 512)
    level_heads: Tuple[int, ...] = (8, 12, 16)
    level_windows: Tuple[int, ...] = (8, 8, 16)
    head_dim: int = 32
    enc_st_blocks: Tuple[int, ...] = (2, 2, 1)   # ST-blocks per encoder level
    dec_st_blocks: Tuple[int, ...] = (1, 2)       # ST-blocks per decoder level
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.0

    # --- Action conditioning ---
    action_dim: int = 16             # 4x4 SE(3) flattened

    # --- Discrete diffusion ---
    mask_schedule: str = "cosine"    # gamma(u) = cos(u * pi/2)
    noise_eta: float = 20.0          # eta% uniform noise
    label_smoothing: float = 0.1
    prob_future_pred: float = 0.5
    prob_joint_denoise: float = 0.4
    prob_individual_denoise: float = 0.1

    # --- Inference ---
    num_sampling_steps: int = 10
    cfg_weight: float = 2.0
    choice_temperature: float = 4.5

    # --- Optimization (paper A.3) ---
    lr: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.0001
    warmup_steps: int = 2000
    max_steps: int = 750000
    cosine_min_ratio: float = 0.1
    batch_size: int = 8
    grad_clip: float = 5.0
    amp: bool = True

    # --- Data ---
    kitti_root: str = "data/kitti/pykitti"
    tokenizer_checkpoint: str = ""
    token_dir: str = ""
    train_sequences: List[str] = field(
        default_factory=lambda: [f"{i:02d}" for i in range(9)]
    )
    val_sequences: List[str] = field(default_factory=lambda: ["09"])
    test_sequences: List[str] = field(default_factory=lambda: ["10"])

    # --- Checkpointing ---
    save_every_steps: int = 5000
    eval_every_steps: int = 2500
    log_every_steps: int = 100
    output_dir: str = "outputs/world_model"

    # --- Derived properties ---
    @property
    def mask_token_id(self) -> int:
        return self.codebook_size     # 1024

    @property
    def vocab_size(self) -> int:
        return self.codebook_size + 1  # 1025 (includes mask token)

    @property
    def num_tokens_per_frame(self) -> int:
        return self.token_grid_h * self.token_grid_w  # 16384
