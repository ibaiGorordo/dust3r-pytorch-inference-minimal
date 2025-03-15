# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# Main encoder/decoder blocks
# --------------------------------------------------------
# References:
# timm
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/helpers.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/mlp.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/patch_embed.py

from dataclasses import dataclass
import os
import requests
from tqdm import tqdm

from huggingface_hub import hf_hub_url
import numpy as np
import torch
import torch.nn as nn

model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

def download(url: str, filename: str):
    with open(filename, 'wb') as f:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))

            # tqdm has many interesting parameters. Feel free to experiment!
            tqdm_params = {
                'total': total,
                'miniters': 1,
                'unit': 'B',
                'unit_scale': True,
                'unit_divisor': 1024,
            }
            with tqdm(**tqdm_params) as pb:
                for chunk in r.iter_content(chunk_size=8192):
                    pb.update(len(chunk))
                    f.write(chunk)


def download_hf_model(filename: str, model_dir: str = model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    filename = filename.split('.')[0] + '.pth'
    path = model_dir + "/" + filename
    if os.path.exists(path):
        return path

    print(f"Model {filename} not found, downloading from Hugging Face Hub...")

    repo_id = "camenduru/dust3r"

    url = hf_hub_url(repo_id=repo_id, filename=filename)
    download(url, path)
    print("Model downloaded successfully to", path)

    return path


def get_device() -> torch.device:
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')

    return device

def calculate_img_size(img_size: tuple[int, int],
                       input_size: int,
                       pad_size: int = 16) -> tuple[int, int]:
    # Calculate the new image size to keep the aspect ratio but being multiple of pad_size
    w, h = img_size
    # Checl the aspect ratio
    if h > w:
        new_h = input_size
        new_w = int(w * new_h / h)
    else:
        new_w = input_size
        new_h = int(h * new_w / w)
    # Make the new image size multiple of pad_size
    new_h = int(np.ceil(new_h / pad_size) * pad_size)
    new_w = int(np.ceil(new_w / pad_size) * pad_size)
    return new_w, new_h


@dataclass
class Output:
    input: np.ndarray
    pts3d: np.ndarray
    colors: np.ndarray
    conf_map: np.ndarray
    depth_map: np.ndarray
    intrinsic: np.ndarray
    pose: np.ndarray
    width: int
    height: int


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, rope, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]
        # q,k,v = qkv.unbind(2)  # make torchscript happy (cannot use tensor as tuple)

        q = self.rope(q)
        k = self.rope(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):

    def __init__(self, dim, rope, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = rope

    def forward(self, query, key, value):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]

        q = self.projq(query).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.projk(key).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.projv(value).reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




