"""Swin Transformer blocks adapted from Microsoft's reference implementation.

Includes: WindowAttention, SwinTransformerBlock, PatchEmbed, PatchMerging,
PatchUpsample (new), BasicLayer, SwinEncoder, SwinDecoder.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from typing import Tuple, Optional


def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def trunc_normal_(tensor, std=0.02):
    nn.init.trunc_normal_(tensor, std=std)


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularization."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition into non-overlapping windows.

    Args:
        x: (B, H, W, C)
        window_size: int

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window_partition.

    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: int
        H, W: original spatial dims

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based multi-head self attention with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # Compute relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (num_windows*B, N, C)
            mask: (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with optional shifted window attention."""

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0)
            )
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            x_windows = window_partition(shifted_x, self.window_size)
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding using Conv2d."""

    def __init__(self, img_size=512, patch_size=4, in_chans=64, embed_dim=128, norm_layer=nn.LayerNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            (B, num_patches, embed_dim)
        """
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, Ph*Pw, C)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    """Patch Merging: downsample spatial by 2x, double channels."""

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        Args:
            x: (B, H*W, C)

        Returns:
            (B, H/2*W/2, 2*C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W
        assert H % 2 == 0 and W % 2 == 0

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # (B, H/2, W/2, 4*C)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchUpsample(nn.Module):
    """Patch Upsample: inverse of PatchMerging. Upsample spatial by 2x, halve channels."""

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        # Project from dim to 2 * (dim//2) = dim for 4 sub-pixels, keeping dim//2 channels
        self.expand = nn.Linear(dim, 4 * (dim // 2), bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        """
        Args:
            x: (B, H*W, C)

        Returns:
            (B, 4*H*W, C//2) = (B, (2H)*(2W), C//2)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W

        x = self.expand(x)  # (B, H*W, 4*(C//2))
        x = x.view(B, H, W, 4, C // 2)

        # Rearrange: place 4 sub-pixels into 2x2 spatial grid
        # sub-pixel order: [top-left, bottom-left, top-right, bottom-right]
        x = x.permute(0, 1, 3, 2, 4)  # (B, H, 4, W, C//2) -- intermediate
        x = x.reshape(B, H, 2, 2, W, C // 2)
        x = x.permute(0, 1, 2, 4, 3, 5)  # (B, H, 2, W, 2, C//2)
        x = x.reshape(B, 2 * H, 2 * W, C // 2)
        x = x.view(B, -1, C // 2)

        x = self.norm(x)
        return x


class BasicLayer(nn.Module):
    """A Swin Transformer stage: multiple blocks + optional down/upsample."""

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        upsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample else None
        self.upsample = upsample(input_resolution, dim=dim, norm_layer=norm_layer) if upsample else None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class SwinEncoder(nn.Module):
    """Two-stage Swin encoder: stage1 -> PatchMerging -> stage2 -> LN.

    Input: (B, C_in, H, W) -- BEV feature map
    Output: (B, num_tokens, enc_dim)
    """

    def __init__(self, cfg):
        super().__init__()
        from copilot4d.utils.config import TokenizerConfig

        img_size = cfg.voxel_grid_xy
        patch_res = img_size // cfg.patch_size  # after PatchEmbed

        # PatchEmbed
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.bev_feat_dim,
            embed_dim=cfg.enc_embed_dim,
            norm_layer=nn.LayerNorm,
        )

        # Stochastic depth
        total_depth = cfg.enc_stage1_depth + cfg.enc_stage2_depth
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, total_depth)]

        # Stage 1: Swin blocks at patch_res resolution
        self.stage1 = BasicLayer(
            dim=cfg.enc_embed_dim,
            input_resolution=(patch_res, patch_res),
            depth=cfg.enc_stage1_depth,
            num_heads=cfg.enc_stage1_heads,
            window_size=cfg.window_size,
            mlp_ratio=cfg.mlp_ratio,
            drop_path=dpr[:cfg.enc_stage1_depth],
            downsample=PatchMerging,  # spatial /2, channels *2
            use_checkpoint=cfg.use_checkpoint,
        )

        # After PatchMerging: resolution = patch_res//2, dim = enc_embed_dim*2 = enc_stage2_dim
        merged_res = patch_res // 2

        # Stage 2: Swin blocks at merged resolution
        self.stage2 = BasicLayer(
            dim=cfg.enc_stage2_dim,
            input_resolution=(merged_res, merged_res),
            depth=cfg.enc_stage2_depth,
            num_heads=cfg.enc_stage2_heads,
            window_size=cfg.window_size,
            mlp_ratio=cfg.mlp_ratio,
            drop_path=dpr[cfg.enc_stage1_depth:],
            use_checkpoint=cfg.use_checkpoint,
        )

        self.norm = nn.LayerNorm(cfg.enc_stage2_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) BEV features

        Returns:
            (B, num_tokens, enc_stage2_dim)
        """
        x = self.patch_embed(x)  # (B, patch_res^2, enc_embed_dim)
        x = self.stage1(x)       # (B, (patch_res/2)^2, enc_stage2_dim)
        x = self.stage2(x)       # (B, (patch_res/2)^2, enc_stage2_dim)
        x = self.norm(x)
        return x


class SwinDecoder(nn.Module):
    """Two-stage Swin decoder: stage1 -> PatchUpsample -> stage2 -> LN.

    Input: (B, num_tokens, dec_dim)
    Output: (B, 4*num_tokens, dec_output_dim)
    """

    def __init__(self, cfg):
        super().__init__()
        from copilot4d.utils.config import TokenizerConfig

        token_grid = cfg.token_grid_size  # e.g. 64

        total_depth = cfg.dec_stage1_depth + cfg.dec_stage2_depth
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, total_depth)]

        # Stage 1: Swin blocks at token_grid resolution
        self.stage1 = BasicLayer(
            dim=cfg.enc_stage2_dim,  # same as encoder output dim (= vq_dim)
            input_resolution=(token_grid, token_grid),
            depth=cfg.dec_stage1_depth,
            num_heads=cfg.dec_stage1_heads,
            window_size=cfg.window_size,
            mlp_ratio=cfg.mlp_ratio,
            drop_path=dpr[:cfg.dec_stage1_depth],
            upsample=PatchUpsample,  # spatial *2, channels /2
            use_checkpoint=cfg.use_checkpoint,
        )

        # After PatchUpsample: resolution = 2*token_grid, dim = enc_stage2_dim//2 = dec_output_dim
        upsampled_res = token_grid * 2

        # Stage 2: Swin blocks at upsampled resolution
        self.stage2 = BasicLayer(
            dim=cfg.dec_output_dim,
            input_resolution=(upsampled_res, upsampled_res),
            depth=cfg.dec_stage2_depth,
            num_heads=cfg.dec_stage2_heads,
            window_size=cfg.window_size,
            mlp_ratio=cfg.mlp_ratio,
            drop_path=dpr[cfg.dec_stage1_depth:],
            use_checkpoint=cfg.use_checkpoint,
        )

        self.norm = nn.LayerNorm(cfg.dec_output_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_tokens, vq_dim/enc_stage2_dim)

        Returns:
            (B, decoder_output_grid^2, dec_output_dim)
        """
        x = self.stage1(x)  # (B, 4*num_tokens, dec_output_dim)
        x = self.stage2(x)  # (B, 4*num_tokens, dec_output_dim)
        x = self.norm(x)
        return x
