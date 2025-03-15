# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch


def get_positions(b, h, w, device):
    x = torch.arange(w, device=device)
    y = torch.arange(h, device=device)
    positions = torch.cartesian_prod(y, x)
    return positions.view(1, h * w, 2).expand(b, -1, 2)

def get_cos_sin(base, D, seq_len, device, dtype):
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2).float().to(device) / D))
    t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
    freqs = torch.cat((freqs, freqs), dim=-1)
    return freqs.cos(), freqs.sin()

class RoPE2D(torch.nn.Module):

    def __init__(self, batch=2, width=512, height=288, patch_size=16, base=100.0, D=32, device=torch.device('cpu'), dtype=torch.float32):
        super().__init__()

        pos = get_positions(batch, height // patch_size, width // patch_size, device)
        pos_x, pos_y = pos[:, :, 1], pos[:, :, 0]
        cos, sin = get_cos_sin(base, D, int(pos.max()) + 1, device, dtype)

        self.cos_x = torch.nn.functional.embedding(pos_x, cos)[:, None, :, :]
        self.sin_x = torch.nn.functional.embedding(pos_x, sin)[:, None, :, :]
        self.cos_y = torch.nn.functional.embedding(pos_y, cos)[:, None, :, :]
        self.sin_y = torch.nn.functional.embedding(pos_y, sin)[:, None, :, :]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, cos, sin):
        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(self, tokens):
        # split features into two along the feature dimension, and apply rope1d on each half
        y, x = tokens.chunk(2, dim=-1)
        x = self.apply_rope1d(x, self.cos_x, self.sin_x)
        y = self.apply_rope1d(y, self.cos_y, self.sin_y)
        tokens = torch.cat((y, x), dim=-1)
        return tokens


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    width = 512
    height = 288
    patch_size = 16
    batch_size, seq_len, n_heads, dim = 1, 16, 576, 64
    D = dim // 2
    base = 100.0

    model = RoPE2D(width=width, height=height, patch_size=patch_size, base=base, D=D, device=device)

    tokens = torch.randn(batch_size, seq_len, n_heads, dim, dtype=torch.float32, device=device)
    model(tokens)

    # torch.onnx.export(
    #     model,
    #     (tokens,),
    #     "models/rope2d.onnx",
    #     input_names=["tokens"],
    #     output_names=["rotated_tokens"],
    #     opset_version=13,  # or whichever opset you need
    # )

if __name__ == '__main__':
    main()
