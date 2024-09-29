"""
Rotary Position Embedding adapted from
https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
and
https://github.com/BaratiLab/OFormer/tree/main/airfoil
"""
import torch
from torch import nn
from einops import rearrange


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, min_freq=1/64, scale=1.):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.interpolation_factor = self.scale / self.min_freq
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, coordinates, device):
        t = coordinates.to(device).type_as(self.inv_freq)  # [b, n]
        t = t * self.interpolation_factor  # rescaling of the encoding was shown to be beneficial
        freqs = torch.einsum('... i , j -> ... i j', t, self.inv_freq)  # [b, n, d//2]
        return torch.cat((freqs, freqs), dim=-1)  # [b, n, d]


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_1d(t, freqs):
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
    # split t into first half and second half
    # t: [b, h, n, d]
    # freq_x/y: [b, n, d]
    d = t.shape[-1]
    t_x, t_y = t[..., :d//2], t[..., d//2:]

    return torch.cat((apply_rotary_pos_emb_1d(t_x, freqs_x),
                      apply_rotary_pos_emb_1d(t_y, freqs_y)), dim=-1)


def apply_rotary_pos_emb_multi(t: torch.Tensor, freqs: list):
    space_dim = len(freqs)
    d = t.shape[-1]
    d1 = d // space_dim

    t_emb = []
    for i, freq in enumerate(freqs):
        s = i*d1
        e = (i+1)*d1 if i < space_dim-1 else d   # use all the remaining dimensions
        t_part = t[..., s:e]
        t_emb_i = apply_rotary_pos_emb_1d(t_part, freq)
        t_emb.append(t_emb_i)

    t_emb = torch.cat(t_emb, dim=-1)
    return t_emb
