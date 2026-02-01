"""Shape verification tests for all tokenizer modules."""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from copilot4d.utils.config import TokenizerConfig
from copilot4d.tokenizer.voxel_encoder import VoxelPointNet
from copilot4d.tokenizer.bev_pooling import BEVPillarPooling
from copilot4d.tokenizer.swin_transformer import (
    SwinEncoder, SwinDecoder, PatchEmbed, PatchMerging, PatchUpsample
)
from copilot4d.tokenizer.vector_quantizer import VectorQuantizer
from copilot4d.tokenizer.neural_feature_grid import NeuralFeatureGrid
from copilot4d.tokenizer.spatial_skipping import SpatialSkipBranch
from copilot4d.tokenizer.tokenizer_model import CoPilot4DTokenizer


def get_debug_config():
    """Get debug config for testing."""
    return TokenizerConfig(
        voxel_grid_xy=256,
        voxel_grid_z=16,
        rays_per_frame=128,
        ray_chunk_size=64,
        batch_size=2,
    )


def get_prototype_config():
    """Get prototype config for testing."""
    return TokenizerConfig(
        voxel_grid_xy=512,
        voxel_grid_z=32,
        rays_per_frame=256,
        ray_chunk_size=128,
        batch_size=2,
        use_checkpoint=True,
    )


class TestVoxelEncoder:
    """Test VoxelPointNet shapes."""

    def test_forward(self):
        model = VoxelPointNet(in_dim=4, hidden_dim=16, out_dim=16)
        V, P = 100, 35
        features = torch.randn(V, P, 4)
        num_points = torch.randint(1, P, (V,))

        out = model(features, num_points)
        assert out.shape == (V, 16)


class TestBEVPillarPooling:
    """Test BEVPillarPooling shapes."""

    def test_forward_debug(self):
        cfg = get_debug_config()
        model = BEVPillarPooling(
            voxel_dim=cfg.voxel_feat_dim,
            z_bins=cfg.voxel_grid_z,
            bev_dim=cfg.bev_feat_dim,
        )

        V = 500
        voxel_features = torch.randn(V, cfg.voxel_feat_dim)
        coords = torch.randint(0, cfg.voxel_grid_xy, (V, 3))
        coords[:, 0] = coords[:, 0] % cfg.batch_size  # batch_idx

        bev = model(
            voxel_features,
            coords,
            cfg.batch_size,
            cfg.voxel_grid_xy,
            cfg.voxel_grid_xy,
        )
        assert bev.shape == (cfg.batch_size, cfg.voxel_grid_xy, cfg.voxel_grid_xy, cfg.bev_feat_dim)


class TestSwinTransformer:
    """Test Swin encoder/decoder shapes."""

    def test_patch_embed_debug(self):
        cfg = get_debug_config()
        patch_embed = PatchEmbed(
            img_size=cfg.voxel_grid_xy,
            patch_size=cfg.patch_size,
            in_chans=cfg.bev_feat_dim,
            embed_dim=cfg.enc_embed_dim,
        )

        x = torch.randn(cfg.batch_size, cfg.bev_feat_dim, cfg.voxel_grid_xy, cfg.voxel_grid_xy)
        out = patch_embed(x)

        expected_patches = (cfg.voxel_grid_xy // cfg.patch_size) ** 2
        assert out.shape == (cfg.batch_size, expected_patches, cfg.enc_embed_dim)

    def test_encoder_debug(self):
        cfg = get_debug_config()
        encoder = SwinEncoder(cfg)

        bev = torch.randn(cfg.batch_size, cfg.bev_feat_dim, cfg.voxel_grid_xy, cfg.voxel_grid_xy)
        out = encoder(bev)

        expected_tokens = cfg.token_grid_size ** 2
        assert out.shape == (cfg.batch_size, expected_tokens, cfg.enc_stage2_dim)

    def test_encoder_prototype(self):
        cfg = get_prototype_config()
        encoder = SwinEncoder(cfg)

        bev = torch.randn(cfg.batch_size, cfg.bev_feat_dim, cfg.voxel_grid_xy, cfg.voxel_grid_xy)
        out = encoder(bev)

        expected_tokens = cfg.token_grid_size ** 2
        assert out.shape == (cfg.batch_size, expected_tokens, cfg.enc_stage2_dim)

    def test_decoder_debug(self):
        cfg = get_debug_config()
        decoder = SwinDecoder(cfg)

        num_tokens = cfg.token_grid_size ** 2
        x = torch.randn(cfg.batch_size, num_tokens, cfg.vq_dim)
        out = decoder(x)

        expected_positions = cfg.decoder_output_grid_size ** 2
        assert out.shape == (cfg.batch_size, expected_positions, cfg.dec_output_dim)

    def test_patch_merging(self):
        B, H, W, C = 2, 64, 64, 128
        patch_merge = PatchMerging(input_resolution=(H, W), dim=C)

        x = torch.randn(B, H * W, C)
        out = patch_merge(x)

        assert out.shape == (B, (H // 2) * (W // 2), 2 * C)

    def test_patch_upsample(self):
        B, H, W, C = 2, 32, 32, 256
        patch_upsample = PatchUpsample(input_resolution=(H, W), dim=C)

        x = torch.randn(B, H * W, C)
        out = patch_upsample(x)

        assert out.shape == (B, (2 * H) * (2 * W), C // 2)


class TestVectorQuantizer:
    """Test VectorQuantizer shapes."""

    def test_forward(self):
        vq = VectorQuantizer(
            dim=256,
            codebook_size=1024,
            codebook_dim=1024,
        )

        B, N = 2, 64
        x = torch.randn(B, N, 256)
        quantized, indices, loss = vq(x)

        assert quantized.shape == (B, N, 256)
        assert indices.shape == (B, N)
        assert loss.ndim == 0  # scalar

    def test_get_codebook_entry(self):
        vq = VectorQuantizer(
            dim=256,
            codebook_size=1024,
            codebook_dim=1024,
        )

        B, H, W = 2, 8, 8
        indices = torch.randint(0, 1024, (B, H, W))
        entries = vq.get_codebook_entry(indices)

        assert entries.shape == (B, H, W, 256)


class TestNeuralFeatureGrid:
    """Test NeuralFeatureGrid shapes."""

    def test_build_nfg_debug(self):
        cfg = get_debug_config()
        nfg = NeuralFeatureGrid(cfg)

        dec_grid = cfg.decoder_output_grid_size
        decoder_output = torch.randn(cfg.batch_size, dec_grid ** 2, cfg.dec_output_dim)

        grid = nfg.build_nfg(decoder_output)
        # Expected: (B, F, Z, H, W)
        assert grid.shape == (cfg.batch_size, cfg.nfg_feat_dim, cfg.nfg_z_bins, cfg.nfg_spatial_size, cfg.nfg_spatial_size)

    def test_query_rays_debug(self):
        cfg = get_debug_config()
        nfg = NeuralFeatureGrid(cfg)

        dec_grid = cfg.decoder_output_grid_size
        decoder_output = torch.randn(cfg.batch_size, dec_grid ** 2, cfg.dec_output_dim)
        ray_origins = torch.randn(cfg.batch_size, cfg.rays_per_frame, 3)
        ray_directions = torch.randn(cfg.batch_size, cfg.rays_per_frame, 3)
        ray_directions = ray_directions / ray_directions.norm(dim=-1, keepdim=True)

        pred_depths, weights, grid = nfg(decoder_output, ray_origins, ray_directions)

        assert pred_depths.shape == (cfg.batch_size, cfg.rays_per_frame)
        assert weights.shape == (cfg.batch_size, cfg.rays_per_frame, cfg.num_depth_samples)
        assert grid.shape == (cfg.batch_size, cfg.nfg_feat_dim, cfg.nfg_z_bins, cfg.nfg_spatial_size, cfg.nfg_spatial_size)


class TestSpatialSkipBranch:
    """Test SpatialSkipBranch shapes."""

    def test_forward_debug(self):
        cfg = get_debug_config()
        skip = SpatialSkipBranch(cfg)

        dec_grid = cfg.decoder_output_grid_size
        decoder_output = torch.randn(cfg.batch_size, dec_grid ** 2, cfg.dec_output_dim)

        skip_logits = skip(decoder_output)
        assert skip_logits.shape == (cfg.batch_size, cfg.skip_grid_xy, cfg.skip_grid_xy, cfg.voxel_grid_z)


class TestFullTokenizer:
    """Test full CoPilot4DTokenizer."""

    def test_encode_debug(self):
        cfg = get_debug_config()
        model = CoPilot4DTokenizer(cfg)

        V = 500
        features = torch.randn(V, cfg.max_points_per_voxel, 4)
        num_points = torch.randint(1, cfg.max_points_per_voxel, (V,))
        coords = torch.randint(0, cfg.voxel_grid_xy, (V, 3))
        coords[:, 0] = coords[:, 0] % cfg.batch_size

        quantized, indices = model.encode(features, num_points, coords, cfg.batch_size)

        expected_tokens = cfg.token_grid_size ** 2
        assert quantized.shape == (cfg.batch_size, expected_tokens, cfg.vq_dim)
        assert indices.shape == (cfg.batch_size, cfg.token_grid_size, cfg.token_grid_size)

    def test_forward_debug(self):
        cfg = get_debug_config()
        model = CoPilot4DTokenizer(cfg)

        V = 500
        features = torch.randn(V, cfg.max_points_per_voxel, 4)
        num_points = torch.randint(1, cfg.max_points_per_voxel, (V,))
        coords = torch.randint(0, cfg.voxel_grid_xy, (V, 3))
        coords[:, 0] = coords[:, 0] % cfg.batch_size

        ray_origins = torch.randn(cfg.batch_size, cfg.rays_per_frame, 3)
        ray_directions = torch.randn(cfg.batch_size, cfg.rays_per_frame, 3)
        ray_directions = ray_directions / ray_directions.norm(dim=-1, keepdim=True)
        ray_depths = torch.rand(cfg.batch_size, cfg.rays_per_frame) * 50 + 5

        gt_occupancy = torch.rand(cfg.batch_size, cfg.voxel_grid_xy, cfg.voxel_grid_xy, cfg.voxel_grid_z)
        gt_occupancy = (gt_occupancy > 0.5).float()

        outputs = model(
            features=features,
            num_points=num_points,
            coords=coords,
            batch_size=cfg.batch_size,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            gt_depths=ray_depths,
            gt_occupancy=gt_occupancy,
        )

        # Check outputs
        assert "losses" in outputs
        assert "pred_depths" in outputs
        assert outputs["pred_depths"].shape == (cfg.batch_size, cfg.rays_per_frame)
        assert "indices" in outputs
        assert outputs["indices"].shape == (cfg.batch_size, cfg.token_grid_size, cfg.token_grid_size)
        assert "skip_logits" in outputs
        assert outputs["skip_logits"].shape == (cfg.batch_size, cfg.skip_grid_xy, cfg.skip_grid_xy, cfg.voxel_grid_z)

    def test_get_tokens_debug(self):
        cfg = get_debug_config()
        model = CoPilot4DTokenizer(cfg)

        V = 500
        features = torch.randn(V, cfg.max_points_per_voxel, 4)
        num_points = torch.randint(1, cfg.max_points_per_voxel, (V,))
        coords = torch.randint(0, cfg.voxel_grid_xy, (V, 3))
        coords[:, 0] = coords[:, 0] % cfg.batch_size

        indices = model.get_tokens(features, num_points, coords, cfg.batch_size)
        assert indices.shape == (cfg.batch_size, cfg.token_grid_size, cfg.token_grid_size)

    def test_render_from_tokens_debug(self):
        cfg = get_debug_config()
        model = CoPilot4DTokenizer(cfg)

        indices = torch.randint(0, cfg.vq_codebook_size, (cfg.batch_size, cfg.token_grid_size, cfg.token_grid_size))
        ray_origins = torch.randn(cfg.batch_size, cfg.rays_per_frame, 3)
        ray_directions = torch.randn(cfg.batch_size, cfg.rays_per_frame, 3)
        ray_directions = ray_directions / ray_directions.norm(dim=-1, keepdim=True)

        outputs = model.render_from_tokens(indices, ray_origins, ray_directions)
        assert outputs["pred_depths"].shape == (cfg.batch_size, cfg.rays_per_frame)


class TestConfigDerivedProperties:
    """Test TokenizerConfig derived properties."""

    def test_debug_config(self):
        cfg = get_debug_config()

        assert cfg.voxel_size_xy == (cfg.x_max - cfg.x_min) / cfg.voxel_grid_xy
        assert cfg.token_grid_size == cfg.voxel_grid_xy // (cfg.patch_size * 2)
        assert cfg.decoder_output_grid_size == cfg.token_grid_size * 2
        assert cfg.nfg_spatial_size == cfg.decoder_output_grid_size * cfg.nfg_upsample_factor
        assert cfg.num_tokens == cfg.token_grid_size ** 2

    def test_prototype_config(self):
        cfg = get_prototype_config()

        assert cfg.token_grid_size == 64  # 512 // (4 * 2)
        assert cfg.decoder_output_grid_size == 128  # 64 * 2
        assert cfg.nfg_spatial_size == 256  # 128 * 2


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
