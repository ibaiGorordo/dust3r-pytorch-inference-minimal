# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# Main encoder
# --------------------------------------------------------
# References:
# timm
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/patch_embed.py

from functools import partial

import torch
import torch.nn as nn

from .third_party import RoPE2D
from .blocks import DropPath, Mlp, Attention


class Block(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 rope: RoPE2D,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer: nn.GELU = nn.GELU,
                 norm_layer: partial = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                              proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ just adding _init_weights + position getter compared to timm.models.layers.patch_embed.PatchEmbed"""

    def __init__(self,
                 img_size: tuple[int, int] = (512, 512),
                 patch_size: tuple[int, int] = (16, 16),
                 in_chans: int = 3,
                 embed_dim: int = 768,
                 norm_layer: nn.Module = None):

        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

    def _init_weights(self):
        w = self.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

class Dust3rEncoder(nn.Module):
    def __init__(self,
                 ckpt_dict: dict,
                 batch: int = 2,
                 width: int = 512,
                 height: int = 512,
                 patch_size: int = 16,
                 enc_embed_dim: int = 1024,
                 enc_num_heads: int = 16,
                 enc_depth: int = 24,
                 mlp_ratio: float = 4.,
                 norm_layer: partial = partial(nn.LayerNorm, eps=1e-6),
                 device: torch.device = torch.device('cuda')
                 ):
        super().__init__()
        self.patch_embed = PatchEmbed((height, width), (patch_size,patch_size), 3, enc_embed_dim)
        self.rope = RoPE2D(batch, width, height, patch_size, base=100.0, device=device)
        self.enc_blocks = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, self.rope, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(enc_depth)])
        self.enc_norm = norm_layer(enc_embed_dim)

        self._load_checkpoint(ckpt_dict)
        self.to(device)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x)

        return self.enc_norm(x)

    def _load_checkpoint(self, ckpt_dict: dict):
        enc_state_dict = {
            k: v for k, v in ckpt_dict['model'].items()
            if k.startswith("patch_embed") or k.startswith("enc_blocks") or k.startswith("enc_norm")
        }
        self.load_state_dict(enc_state_dict, strict=True)