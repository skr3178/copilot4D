 CoPilot4D World Model Implementation Plan                                                          │
     │                                                                                                    │
     │ Overview                                                                                           │
     │                                                                                                    │
     │ Implement the U-Net based Spatio-Temporal Transformer world model (Figure 7, Section 4.4, Appendix │
     │ A.2.2) that operates on discrete BEV tokens from the frozen tokenizer. The world model uses        │
     │ discrete diffusion (improved MaskGIT) to predict future observations conditioned on past           │
     │ observations and ego-vehicle actions.                                                              │
     │                                                                                                    │
     │ Model size target: ~39M parameters.                                                                │
     │                                                                                                    │
     │ ---                                                                                                │
     │ Architecture Summary (from paper)                                                                  │
     │                                                                                                    │
     │ Input: (B, T, 128, 128) discrete token indices + (B, T, 16) actions                                │
     │                           ↓                                                                        │
     │               Token Embed (1025, 256) + Linear→LN→Linear                                           │
     │               + Spatial Pos (128×128, 256)                                                         │
     │               + Temporal Pos (T, 256)                                                              │
     │               + Action Proj Linear→LN→Linear                                                       │
     │                           ↓                                                                        │
     │ ENCODER Level 1: 128×128, dim=256, 8 heads, win=8                                                  │
     │   2 ST-Blocks (each = 2 Swin + 1 Temporal)                                                         │
     │               → Patch Merging (256→384) ↓                                                          │
     │ ENCODER Level 2: 64×64, dim=384, 12 heads, win=8                                                   │
     │   2 ST-Blocks                                                                                      │
     │               → Patch Merging (384→512) ↓                                                          │
     │ BOTTLENECK Level 3: 32×32, dim=512, 16 heads, win=16                                               │
     │   1 ST-Block                                                                                       │
     │               → Level Merging (512→384) ↑ + skip from Enc L2                                       │
     │ DECODER Level 2: 64×64, dim=384                                                                    │
     │   1 ST-Block                                                                                       │
     │               → Level Merging (384→256) ↑ + skip from Enc L1                                       │
     │ DECODER Level 1: 128×128, dim=256                                                                  │
     │   2 ST-Blocks                                                                                      │
     │                           ↓                                                                        │
     │               LayerNorm → Weight-tied Linear → (B, T, 128×128, 1024) logits                        │
     │                                                                                                    │
     │ ---                                                                                                │
     │ File Structure                                                                                     │
     │                                                                                                    │
     │ copilot4d/                                                                                         │
     │ ├── world_model/                          # NEW package                                            │
     │ │   ├── __init__.py                                                                                │
     │ │   ├── world_model.py                    # CoPilot4DWorldModel (main module)                      │
     │ │   ├── temporal_block.py                 # GPT-2 style causal temporal attention                  │
     │ │   ├── spatio_temporal_block.py          # 2 Swin blocks + 1 temporal block                       │
     │ │   ├── patch_merging.py                  # PatchMerging with custom target_dim                    │
     │ │   ├── level_merging.py                  # Upsample + skip connection (decoder)                   │
     │ │   ├── input_embeddings.py               # Token/spatial/temporal/action embeddings               │
     │ │   ├── masking.py                        # Discrete diffusion masking + 3 objectives              │
     │ │   └── inference.py                      # Iterative decoding with CFG                            │
     │ ├── data/                                                                                          │
     │ │   └── kitti_sequence_dataset.py         # NEW: sequential frame loading                          │
     │ ├── utils/                                                                                         │
     │ │   └── config.py                         # MODIFY: add WorldModelConfig                           │
     │ scripts/                                                                                           │
     │ ├── pretokenize_kitti.py                  # NEW: pre-compute tokens to disk                        │
     │ ├── train_world_model.py                  # NEW: training script                                   │
     │ configs/                                                                                           │
     │ ├── world_model.yaml                      # NEW: full config                                       │
     │ └── world_model_debug.yaml                # NEW: small debug config                                │
     │                                                                                                    │
     │ ---                                                                                                │
     │ Implementation Steps                                                                               │
     │                                                                                                    │
     │ Step 1: Add WorldModelConfig to copilot4d/utils/config.py                                          │
     │                                                                                                    │
     │ Add a new dataclass after TokenizerConfig. Key fields:                                             │
     │                                                                                                    │
     │ @dataclass                                                                                         │
     │ class WorldModelConfig:                                                                            │
     │     # Tokenizer interface                                                                          │
     │     codebook_size: int = 1024        # Must match tokenizer                                        │
     │     token_grid_h: int = 128                                                                        │
     │     token_grid_w: int = 128                                                                        │
     │                                                                                                    │
     │     # Sequence (3+3 for initial dev, switch to 5+5 later)                                          │
     │     num_frames: int = 6              # 3 past + 3 future for faster iteration                      │
     │                                                                                                    │
     │     # U-Net levels: dims (256, 384, 512), heads (8, 12, 16), windows (8, 8, 16)                    │
     │     level_dims: tuple = (256, 384, 512)                                                            │
     │     level_heads: tuple = (8, 12, 16)                                                               │
     │     level_windows: tuple = (8, 8, 16)                                                              │
     │     head_dim: int = 32               # Fixed per-head dim across all levels                        │
     │     # ST-blocks per level: enc (2,2,1), dec (1,2) — matching Figure 7                              │
     │     enc_st_blocks: tuple = (2, 2, 1)                                                               │
     │     dec_st_blocks: tuple = (1, 2)                                                                  │
     │     mlp_ratio: float = 4.0                                                                         │
     │                                                                                                    │
     │     # Action conditioning                                                                          │
     │     action_dim: int = 16             # 4×4 SE(3) flattened                                         │
     │                                                                                                    │
     │     # Discrete diffusion                                                                           │
     │     mask_schedule: str = "cosine"    # γ(u) = cos(u·π/2)                                           │
     │     noise_eta: float = 20.0          # η% uniform noise                                            │
     │     label_smoothing: float = 0.1                                                                   │
     │     # Training objective mix                                                                       │
     │     prob_future_pred: float = 0.5                                                                  │
     │     prob_joint_denoise: float = 0.4                                                                │
     │     prob_individual_denoise: float = 0.1                                                           │
     │     num_past_frames: int = 3         # 3 past for initial dev                                      │
     │                                                                                                    │
     │     # Inference                                                                                    │
     │     num_sampling_steps: int = 10     # K diffusion steps                                           │
     │     cfg_weight: float = 2.0          # Guidance scale w                                            │
     │     choice_temperature: float = 4.5                                                                │
     │                                                                                                    │
     │     # Optimization (from paper A.3)                                                                │
     │     lr: float = 0.001                                                                              │
     │     beta2: float = 0.95                                                                            │
     │     weight_decay: float = 0.0001                                                                   │
     │     warmup_steps: int = 2000                                                                       │
     │     max_steps: int = 750000                                                                        │
     │     cosine_min_ratio: float = 0.1                                                                  │
     │     batch_size: int = 8                                                                            │
     │     grad_clip: float = 5.0                                                                         │
     │     amp: bool = True                                                                               │
     │                                                                                                    │
     │     # Data                                                                                         │
     │     kitti_root: str = "data/kitti/pykitti"                                                         │
     │     tokenizer_checkpoint: str = ""                                                                 │
     │     token_dir: str = ""              # Pre-computed tokens directory                               │
     │                                                                                                    │
     │     # Derived                                                                                      │
     │     @property                                                                                      │
     │     def mask_token_id(self) -> int:                                                                │
     │         return self.codebook_size    # 1024                                                        │
     │                                                                                                    │
     │     @property                                                                                      │
     │     def vocab_size(self) -> int:                                                                   │
     │         return self.codebook_size + 1  # 1025 (includes mask token)                                │
     │                                                                                                    │
     │     @property                                                                                      │
     │     def num_tokens_per_frame(self) -> int:                                                         │
     │         return self.token_grid_h * self.token_grid_w  # 16384                                      │
     │                                                                                                    │
     │ Step 2: copilot4d/world_model/input_embeddings.py                                                  │
     │                                                                                                    │
     │ WorldModelInputEmbedding — Converts discrete tokens + positions + actions into continuous          │
     │ embeddings.                                                                                        │
     │                                                                                                    │
     │ Paper (A.2.2): "After the embedding layer, we additionally apply Linear → LayerNorm → Linear." All │
     │ conditioning goes through Linear → LN → Linear before addition. Bias removed in all Linear layers  │
     │ (except QKV in Swin).                                                                              │
     │                                                                                                    │
     │ forward(token_indices: (B,T,H*W), actions: (B,T,16)) → (B,T,H*W,256)                               │
     │                                                                                                    │
     │ Components:                                                                                        │
     │ - token_embedding: nn.Embedding(1025, 256) — weight-tied with output projection                    │
     │ - embed_proj: Linear(256,256,bias=False) → LN(256) → Linear(256,256,bias=False)                    │
     │ - spatial_pos: nn.Parameter(1, H*W, 256) — ViT-style, same for all frames                          │
     │ - temporal_pos: nn.Parameter(1, T_max, 1, 256) — learnable, broadcast to spatial                   │
     │ - action_proj: Linear(16,256,bias=False) → LN(256) → Linear(256,256,bias=False)                    │
     │                                                                                                    │
     │ Step 3: copilot4d/world_model/temporal_block.py                                                    │
     │                                                                                                    │
     │ TemporalBlock — GPT-2 style causal attention across time at each spatial location.                 │
     │                                                                                                    │
     │ forward(x: (B,T,N,C), temporal_mask: (T,T)) → (B,T,N,C)                                            │
     │                                                                                                    │
     │ Internally reshapes to (B*N, T, C) to apply temporal attention across T for each spatial location  │
     │ independently. Pre-norm architecture (LN before attention and MLP).                                │
     │                                                                                                    │
     │ Components:                                                                                        │
     │ - norm1: LayerNorm(C)                                                                              │
     │ - attn: nn.MultiheadAttention(C, num_heads, batch_first=True, bias=True for QKV)                   │
     │ - norm2: LayerNorm(C)                                                                              │
     │ - mlp: Linear(C, 4C, bias=False) → GELU → Linear(4C, C, bias=False)                                │
     │ - drop_path: DropPath (from swin_transformer.py)                                                   │
     │                                                                                                    │
     │ Mask helpers (module-level functions):                                                             │
     │ - make_causal_mask(T) → lower-triangular (T,T) float mask (0 = attend, -inf = block)               │
     │ - make_identity_mask(T) → diagonal-only (T,T) float mask                                           │
     │                                                                                                    │
     │ Step 4: copilot4d/world_model/spatio_temporal_block.py                                             │
     │                                                                                                    │
     │ SpatioTemporalBlock — 2 Swin blocks (spatial) + 1 temporal block.                                  │
     │                                                                                                    │
     │ forward(x: (B,T,N,C), temporal_mask: (T,T)) → (B,T,N,C)                                            │
     │                                                                                                    │
     │ Reuses SwinTransformerBlock from copilot4d/tokenizer/swin_transformer.py:                          │
     │ - Import directly: from copilot4d.tokenizer.swin_transformer import SwinTransformerBlock           │
     │ - Swin blocks operate per-frame: reshape (B,T,N,C) → (B*T,N,C) for spatial, then back              │
     │ - Shift sizes alternate: 0 for first block, window_size//2 for second block                        │
     │ - Temporal block operates across time at each spatial location                                     │
     │                                                                                                    │
     │ Step 5: copilot4d/world_model/patch_merging.py                                                     │
     │                                                                                                    │
     │ WorldModelPatchMerging — Same spatial grouping as tokenizer's PatchMerging but with custom target  │
     │ dim.                                                                                               │
     │                                                                                                    │
     │ The tokenizer's PatchMerging does 4*dim → 2*dim. The world model needs 4*256→384 and 4*384→512.    │
     │                                                                                                    │
     │ forward(x: (B, H*W, C)) → (B, H/2*W/2, target_dim)                                                 │
     │                                                                                                    │
     │ Groups 2×2 patches → concat (4*C) → LayerNorm → Linear(4*C, target_dim, bias=False).               │
     │                                                                                                    │
     │ Step 6: copilot4d/world_model/level_merging.py                                                     │
     │                                                                                                    │
     │ LevelMerging — Decoder upsampling with skip connection.                                            │
     │                                                                                                    │
     │ Paper (A.2.2): "first we use a linear layer to output the 2× upsampled feature map (similar to a   │
     │ deconvolution layer), concatenate with the lower-level feature map, applies LayerNorm on every     │
     │ feature, and uses a linear projection to reduce the feature dimension. A residual connection is    │
     │ then applied."                                                                                     │
     │                                                                                                    │
     │ forward(x_up: (B,T,N_up,C_up), x_skip: (B,T,N_skip,C_skip)) → (B,T,N_skip,C_skip)                  │
     │                                                                                                    │
     │ Components:                                                                                        │
     │ - upsample: ConvTranspose2d(C_up, C_skip, kernel=2, stride=2, bias=False) — 2× spatial             │
     │ - norm: LayerNorm(2 * C_skip)                                                                      │
     │ - proj: Linear(2 * C_skip, C_skip, bias=False)                                                     │
     │ - Residual: output + x_skip                                                                        │
     │                                                                                                    │
     │ Step 7: copilot4d/world_model/world_model.py                                                       │
     │                                                                                                    │
     │ CoPilot4DWorldModel — The complete U-Net Transformer.                                              │
     │                                                                                                    │
     │ forward(token_indices: (B,T,H,W), actions: (B,T,16), temporal_mask: (T,T))                         │
     │     → logits: (B,T,H*W,1025)                                                                       │
     │                                                                                                    │
     │ Full forward pass with tensor shapes:                                                              │
     │ Input tokens: (B, T, 128, 128) → flatten → (B, T, 16384)                                           │
     │ Embed: (B, T, 16384, 256)                                                                          │
     │                                                                                                    │
     │ Encoder L1: 2 ST-blocks → (B, T, 16384, 256), save skip_L1                                         │
     │   PatchMerge: (B*T, 16384, 256) → (B*T, 4096, 384)                                                 │
     │ Encoder L2: 2 ST-blocks → (B, T, 4096, 384), save skip_L2                                          │
     │   PatchMerge: (B*T, 4096, 384) → (B*T, 1024, 512)                                                  │
     │ Bottleneck L3: 1 ST-block → (B, T, 1024, 512)                                                      │
     │                                                                                                    │
     │ LevelMerge 3→2: upsample + skip_L2 → (B, T, 4096, 384)                                             │
     │ Decoder L2: 1 ST-block → (B, T, 4096, 384)                                                         │
     │ LevelMerge 2→1: upsample + skip_L1 → (B, T, 16384, 256)                                            │
     │ Decoder L1: 2 ST-blocks → (B, T, 16384, 256)                                                       │
     │                                                                                                    │
     │ Output: LayerNorm → F.linear(x, embed.weight, bias) → (B, T, 16384, 1025)                          │
     │                                                                                                    │
     │ Weight tying: output projection reuses input_embed.token_embedding.weight (1025×256).              │
     │                                                                                                    │
     │ Initialization (paper A.3):                                                                        │
     │ - Fan-in: all weights init with Normal(0, sqrt(1/(3*H))) where H = input dim                       │
     │ - Residual scaling: per feature level, count L = num_transformer_blocks × 2, scale residual output │
     │ projections by sqrt(1/L)                                                                           │
     │ - Bias = False everywhere except QKV in Swin WindowAttention                                       │
     │                                                                                                    │
     │ Step 8: copilot4d/world_model/masking.py                                                           │
     │                                                                                                    │
     │ DiscreteDiffusionMasker — Implements Algorithm 1 + 3 training objectives.                          │
     │                                                                                                    │
     │ prepare_batch(tokens: (B,T,H,W))                                                                   │
     │     → masked_tokens: (B,T,H,W), targets: (B,T,H,W), temporal_mask: (T,T), objective: str           │
     │                                                                                                    │
     │ Algorithm 1 (per frame):                                                                           │
     │ 1. Sample u₀ ~ Uniform(0,1), mask ⌈γ(u₀)·N⌉ tokens → replace with mask_token_id (1024)             │
     │ 2. Sample u₁ ~ Uniform(0,1), noise ⌊u₁·η%⌋ of remaining tokens → random codebook indices           │
     │ 3. Loss = CE(logits, original) on ALL positions (masked + unmasked)                                │
     │                                                                                                    │
     │ Three objectives (randomly sampled per iteration):                                                 │
     │                                                                                                    │
     │                                                                                                    │
     │                                                                                                    │
     │ Objective: Future prediction                                                                       │
     │ Probability: 50%                                                                                   │
     │ Past frames: Unmasked (ground truth)                                                               │
     │ Future frames: Fully masked+noised                                                                 │
     │ Temporal mask: Causal                                                                              │
     │ ────────────────────────────────────────                                                           │
     │ Objective: Joint denoise                                                                           │
     │ Probability: 40%                                                                                   │
     │ Past frames: Partially masked+noised                                                               │
     │ Future frames: Partially masked+noised                                                             │
     │ Temporal mask: Causal                                                                              │
     │ ────────────────────────────────────────                                                           │
     │ Objective: Individual denoise                                                                      │
     │ Probability: 10%                                                                                   │
     │ Past frames: Independently masked+noised                                                           │
     │ Future frames: Independently masked+noised                                                         │
     │ Temporal mask: Identity                                                                            │
     │ Step 9: copilot4d/world_model/inference.py                                                         │
     │                                                                                                    │
     │ WorldModelSampler — Algorithm 2 with classifier-free guidance.                                     │
     │                                                                                                    │
     │ predict_next_frame(past_tokens: (B,T_past,H,W), past_actions: (B,T_past,16),                       │
     │                    future_action: (B,1,16)) → predicted_tokens: (B,H,W)                            │
     │                                                                                                    │
     │ Algorithm 2 (K steps):                                                                             │
     │ 1. x_K = all mask tokens                                                                           │
     │ 2. For k = K-1, ..., 0:                                                                            │
     │   - Get logits from model (conditional pass with causal mask)                                      │
     │   - Get logits from model (unconditional pass with identity mask for last frame)                   │
     │   - CFG: logits = cond + w · (cond - uncond)                                                       │
     │   - Sample x̃₀ ~ Categorical(logits                                                                 │
     │   - l_k = log p(x̃₀ | x_{k+1}) + Gumbel(0,1) · k/                                                   │
     │   - Set l_k = +∞ on non-mask indices of x_{k+1}                                                    │
     │   - M = ⌈γ(k/K) · N⌉                                                                               │
     │   - Keep top-M of l_k as unmasked, re-mask rest                                                    │
     │ 3. Return x₀                                                                                       │
     │                                                                                                    │
     │ CFG efficiency (paper A.2.2, Figure 10): "CFG can be efficiently implemented with a single forward │
     │ pass by increasing temporal sequence length by 1, setting the attention mask to be a causal mask   │
     │ within the previous sequence length, and an identity mask for the last frame."                     │
     │                                                                                                    │
     │ Step 10: copilot4d/data/kitti_sequence_dataset.py                                                  │
     │                                                                                                    │
     │ KITTISequenceDataset — Loads sequences of T consecutive tokenized frames + relative poses.         │
     │                                                                                                    │
     │ __getitem__(idx) → {"tokens": (T,H,W) long, "actions": (T,16) float}                               │
     │                                                                                                    │
     │ - Uses pre-computed token files from token_dir (preferred) or online tokenization                  │
     │ - Computes relative SE(3) poses: T_next @ inv(T_curr) → flatten to 16-dim                          │
     │ - Builds sample list of valid windows (sequence, start_frame) where [start, start+T) are valid     │
     │ - Simple collate: stack tokens and actions                                                         │
     │                                                                                                    │
     │ Step 11: scripts/pretokenize_kitti.py                                                              │
     │                                                                                                    │
     │ Pre-compute tokens for all KITTI frames using frozen tokenizer. Does NOT modify any existing       │
     │ tokenizer code — only imports and uses CoPilot4DTokenizer.get_tokens().                            │
     │                                                                                                    │
     │ - Load tokenizer checkpoint (read-only)                                                            │
     │ - For each sequence, each frame: tokenizer.get_tokens() → save (128,128) tensor as .pt             │
     │ - Also save poses for the sequence (for action computation)                                        │
     │ - Output: outputs/tokens/{seq}/{frame:06d}.pt                                                      │
     │ - ~32KB per frame × ~23K frames ≈ 750MB total                                                      │
     │ - CLI args: --tokenizer_config, --tokenizer_checkpoint, --output_dir, --sequences                  │
     │                                                                                                    │
     │ Step 12: scripts/train_world_model.py                                                              │
     │                                                                                                    │
     │ Following train_tokenizer.py patterns:                                                             │
     │ - YAML config loading → WorldModelConfig                                                           │
     │ - Dataset + DataLoader with sequence_collate_fn                                                    │
     │ - AdamW: lr=0.001, β₂=0.95, weight_decay=0.0001 (exclude bias/norm/embed)                          │
     │ - Schedule: 2000-step linear warmup + cosine decay to 10% over 750K steps                          │
     │ - AMP with GradScaler                                                                              │
     │ - Gradient clipping: max_norm=5.0                                                                  │
     │ - Loss: F.cross_entropy(logits.reshape(-1, 1025), targets.reshape(-1), label_smoothing=0.1)        │
     │ - JSONL metrics logging                                                                            │
     │ - Checkpoint saving                                                                                │
     │                                                                                                    │
     │ Step 13: Config files                                                                              │
     │                                                                                                    │
     │ configs/world_model.yaml — Full config matching paper.                                             │
     │ configs/world_model_debug.yaml — Small config for testing (reduced dims, fewer blocks, smaller     │
     │ grid).                                                                                             │
     │                                                                                                    │
     │ ---                                                                                                │
     │ Key Reused Components                                                                              │
     │                                                                                                    │
     │ Component: SwinTransformerBlock                                                                    │
     │ Source file: copilot4d/tokenizer/swin_transformer.py:170                                           │
     │ Usage: Spatial attention in each ST-block                                                          │
     │ ────────────────────────────────────────                                                           │
     │ Component: WindowAttention                                                                         │
     │ Source file: copilot4d/tokenizer/swin_transformer.py:94                                            │
     │ Usage: Used internally by SwinTransformerBlock                                                     │
     │ ────────────────────────────────────────                                                           │
     │ Component: DropPath                                                                                │
     │ Source file: copilot4d/tokenizer/swin_transformer.py:24                                            │
     │ Usage: Stochastic depth in temporal blocks                                                         │
     │ ────────────────────────────────────────                                                           │
     │ Component: Mlp                                                                                     │
     │ Source file: copilot4d/tokenizer/swin_transformer.py:41                                            │
     │ Usage: MLP in temporal blocks                                                                      │
     │ ────────────────────────────────────────                                                           │
     │ Component: window_partition/reverse                                                                │
     │ Source file: copilot4d/tokenizer/swin_transformer.py:61,77                                         │
     │ Usage: Used internally by Swin                                                                     │
     │ ────────────────────────────────────────                                                           │
     │ Component: CoPilot4DTokenizer.get_tokens()                                                         │
     │ Source file: copilot4d/tokenizer/tokenizer_model.py:279                                            │
     │ Usage: Pre-tokenization script                                                                     │
     │ ────────────────────────────────────────                                                           │
     │ Component: CoPilot4DTokenizer.render_from_tokens()                                                 │
     │ Source file: copilot4d/tokenizer/tokenizer_model.py:300                                            │
     │ Usage: Evaluation (tokens→point cloud)                                                             │
     │ ────────────────────────────────────────                                                           │
     │ Component: Training patterns                                                                       │
     │ Source file: scripts/train_tokenizer.py                                                            │
     │ Usage: Optimizer, scheduler, checkpointing, logging                                                │
     │ ────────────────────────────────────────                                                           │
     │ Component: Dataset patterns                                                                        │
     │ Source file: copilot4d/data/kitti_dataset.py                                                       │
     │ Usage: PyKITTI loading, collate function                                                           │
     │ New vs reused: The only truly new neural network component is TemporalBlock. Everything else either│
     │  reuses existing components or composes them.                                                      │
     │                                                                                                    │
     │ ---                                                                                                │
     │ Verification Plan                                                                                  │
     │                                                                                                    │
     │ 1. Unit test each module: Create a small debug config, instantiate each component, pass random     │
     │ tensors through, verify output shapes                                                              │
     │ 2. Parameter count: Verify ~39M total parameters with full config                                  │
     │ 3. Forward pass: Full model forward with random (B=2, T=4, 128, 128) tokens → (B, T, 16384, 1025)  │
     │ logits                                                                                             │
     │ 4. Masking: Verify each of the 3 objectives produces correct mask patterns and temporal attention  │
     │ masks                                                                                              │
     │ 5. Training loop: Run a few steps on debug config with synthetic data, verify loss decreases       │
     │ 6. Pre-tokenization: Run on a few KITTI frames, verify saved token shapes                          │
     │ 7. Inference: Run iterative decoding on debug config, verify output token shapes                   │
     ╰───────────────────────────────────────────────────────────────────────────────────────