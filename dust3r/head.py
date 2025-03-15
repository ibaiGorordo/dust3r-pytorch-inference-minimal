# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# post process function for all heads: extract 3D points/confidence from output
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from .third_party import DPTHead

def postprocess(out):
    """
    extract 3D points/confidence from prediction head output
    """
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,3
    depth = reg_dense_depth(fmap[:, :, :, 0:3])
    conf = reg_dense_conf(fmap[:, :, :, 3])
    return depth, conf


def reg_dense_depth(xyz):
    """
    extract 3D points from prediction head output
    """

    # distance to origin
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)

    return xyz * (torch.exp(d) - 1)

def reg_dense_conf(x, vmin=1, vmax=torch.inf):
    """
    extract confidence from prediction head output
    """
    return vmin + x.exp().clip(max=vmax-vmin)


class LinearPts3d (nn.Module):
    """
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self,
                 width=512,
                 height=512,
                 patch_size=16,
                 dec_embed_dim=768,
                 has_conf=True):
        super().__init__()
        self.patch_size = patch_size
        self.has_conf = has_conf
        self.num_h = height // patch_size
        self.num_w = width // patch_size

        self.proj = nn.Linear(dec_embed_dim, (3 + has_conf)*self.patch_size**2)

    def setup(self, croconet):
        pass

    def forward(self, tokens_0, tokens_6, tokens_9, tokens_12):
        B, S, D = tokens_12.shape

        # extract 3D points
        feat = self.proj(tokens_12)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, self.num_h, self.num_w)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W

        # permute + norm depth
        return feat


class Dust3rHead(nn.Module):
    def __init__(self,
                 ckpt_dict,
                 width=512,
                 height=512,
                 device=torch.device('cuda'),
                 ):
        super().__init__()

        self.downstream_head1 = DPTHead(width, height) if self._is_dpt(ckpt_dict) else LinearPts3d(width, height)
        self.downstream_head2 = DPTHead(width, height) if self._is_dpt(ckpt_dict) else LinearPts3d(width, height)

        self._load_checkpoint(ckpt_dict)
        self.to(device)


    @torch.inference_mode()
    def forward(self, d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12):
        out1 = self.downstream_head1(d1_0, d1_6, d1_9, d1_12)
        out2 = self.downstream_head2(d2_0, d2_6, d2_9, d2_12)

        # Postprocess
        pts3d1, conf1 = postprocess(out1)
        pts3d2, conf2 = postprocess(out2)

        return pts3d1, conf1, pts3d2, conf2

    def _load_checkpoint(self, ckpt_dict):
        head_state_dict = {
            k.replace(".dpt", ""): v
            for k, v in ckpt_dict['model'].items()
            if "head" in k
        }
        self.load_state_dict(head_state_dict, strict=True)

    def _is_dpt(self, ckpt_dict):
        return any("dpt" in k for k in ckpt_dict['model'].keys())