from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn

from .common import DropPath, Mlp, Attention, CrossAttention
from .third_party import RoPE2D


class DecoderBlock(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer: nn.GELU = nn.GELU,
                 norm_layer: partial = nn.LayerNorm,
                 norm_mem: bool = True,
                 rope: RoPE2D = None,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                              proj_drop=drop)
        self.cross_attn = CrossAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                         proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        y_ = self.norm_y(y)
        x = x + self.drop_path(self.cross_attn(self.norm2(x), y_, y_))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x, y



class Dust3rDecoder(nn.Module):
    def __init__(self,
                 ckpt_dict: dict,
                 batch: int = 1,
                 width: int = 512,
                 height: int = 512,
                 patch_size: int = 16,
                 enc_embed_dim: int = 1024,
                 dec_embed_dim: int = 768,
                 dec_num_heads: int = 12,
                 dec_depth: int = 12,
                 mlp_ratio: float = 4.,
                 norm_im2_in_dec: bool = True, # whether to apply normalization of the 'memory' = (second image) in the decoder
                 norm_layer: partial = partial(nn.LayerNorm, eps=1e-6),
                 device: torch.device = torch.device('cuda'),
                 ):
        super().__init__()


        self.rope = RoPE2D(batch, width, height, patch_size, base=100.0, device=device)

        # transfer from encoder to decoder
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                         norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)])

        self.dec_blocks2 = deepcopy(self.dec_blocks)

        # final norm layer
        self.dec_norm = norm_layer(dec_embed_dim)

        self._load_checkpoint(ckpt_dict)
        self.to(device)

    @torch.inference_mode()
    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                                   torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f1_0 = f1_6 = f1_9 = f1
        f2_0 = f2_6 = f2_9 = f2

        # Project to decoder dimension
        f1_prev, f2_prev = self.decoder_embed(f1), self.decoder_embed(f2)


        for i, (blk1, blk2) in enumerate(zip(self.dec_blocks, self.dec_blocks2), start=1):
            # img1 side
            f1, _ = blk1(f1_prev, f2_prev)

            # img2 side
            f2, _ = blk2(f2_prev, f1_prev)

            # Store the result
            f1_prev, f2_prev = f1, f2

            if i == 6:
                f1_6, f2_6 = f1, f2
            elif i == 9:
                f1_9, f2_9 = f1, f2

        f1_12, f2_12 = self.dec_norm(f1), self.dec_norm(f2)

        return f1_0, f1_6, f1_9, f1_12, f2_0, f2_6, f2_9, f2_12

    def _load_checkpoint(self, ckpt_dict: dict):
        dec_state_dict = {
            k: v for k, v in ckpt_dict['model'].items()
            if k.startswith("decoder_embed") or k.startswith("dec_blocks") or k.startswith("dec_norm")
        }
        self.load_state_dict(dec_state_dict, strict=True)
