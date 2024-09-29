"""
The file is adapted from https://github.com/BaratiLab/OFormer/tree/main/airfoil/nn_module
encoder and decoder modules correspondingly as well as attention module
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, orthogonal_
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.optim.lr_scheduler import OneCycleLR
import pytorch_lightning as pl

from models.encoding import RotaryEmbedding, apply_rotary_pos_emb_multi
from models.losses import MultiLoss, CorrelationLoss, ScaledMaeLoss, DownsampledLoss, MaskedLoss
from models.normalizer import Normalizer
from models.loss_helper import get_pde_loss_function


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class GeGELU(nn.Module):
    """https: // paperswithcode.com / method / geglu"""

    def __init__(self):
        super().__init__()
        self.fn = nn.GELU()

    def forward(self, x):
        c = x.shape[-1]  # channel last arrangement
        return self.fn(x[..., :int(c // 2)]) * x[..., int(c // 2):]


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            GeGELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ReLUFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class StandardAttention(nn.Module):
    """Standard scaled dot product attention"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., causal=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.causal = causal  # simple autogressive attention with upper triangular part being masked zero

    def forward(self, x, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            if not self.causal:
                raise Exception('Passing in mask while attention is not causal')
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)  # similarity score

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def masked_instance_norm(x, mask, eps=1e-5):
    """
    x of shape: [batch_size (N), num_objects (L), features(C)]
    mask of shape: [batch_size (N), num_objects (L), 1]
    """
    mask = mask.float()  # (N,L,1)
    mean = (torch.sum(x * mask, 1) / torch.sum(mask, 1))  # (N,C)
    mean = mean.detach()
    var_term = ((x - mean.unsqueeze(1).expand_as(x)) * mask) ** 2  # (N,L,C)
    var = (torch.sum(var_term, 1) / torch.sum(mask, 1))  # (N,C)
    var = var.detach()
    mean_reshaped = mean.unsqueeze(1).expand_as(x)  # (N, L, C)
    var_reshaped = var.unsqueeze(1).expand_as(x)  # (N, L, C)
    ins_norm = (x - mean_reshaped) / torch.sqrt(var_reshaped + eps)  # (N, L, C)
    return ins_norm


class LinearAttention(nn.Module):
    """
    Contains following two types of attention, as discussed in "Choose a Transformer: Fourier or Galerkin"

    Galerkin type attention, with instance normalization on Key and Value
    Fourier type attention, with instance normalization on Query and Key
    """

    def __init__(self,
                 dim,
                 attn_type,  # ['fourier', 'galerkin']
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 init_params=True,
                 relative_emb=False,
                 scale=1.,
                 init_method='orthogonal',  # ['xavier', 'orthogonal']
                 init_gain=None,
                 relative_emb_dim=2,
                 min_freq=1 / 64,  # 1/64 is for 64 x 64 ns2d,
                 cat_pos=False,
                 pos_dim=2,
                 use_ln=False
                 ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.attn_type = attn_type
        self.use_ln = use_ln

        self.heads = heads
        self.dim_head = dim_head

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if attn_type == 'galerkin':
            if not self.use_ln:
                self.k_norm = nn.InstanceNorm1d(dim_head)  # affine=True
                self.v_norm = nn.InstanceNorm1d(dim_head)  # affine=True
            else:
                self.k_norm = nn.LayerNorm(dim_head)
                self.v_norm = nn.LayerNorm(dim_head)

        elif attn_type == 'fourier':
            if not self.use_ln:
                self.q_norm = nn.InstanceNorm1d(dim_head)
                self.k_norm = nn.InstanceNorm1d(dim_head)
            else:
                self.q_norm = nn.LayerNorm(dim_head)
                self.k_norm = nn.LayerNorm(dim_head)

        else:
            raise Exception(f'Unknown attention type {attn_type}')

        if not cat_pos:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim + pos_dim * heads, dim),
                nn.Dropout(dropout)
            )

        if init_gain is None:
            self.init_gain = 1. / dim_head
            self.diagonal_weight = 1. / dim_head
        else:
            self.init_gain = init_gain
            self.diagonal_weight = init_gain

        self.init_method = init_method
        if init_params:
            self._init_params()

        self.cat_pos = cat_pos
        self.pos_dim = pos_dim

        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        if relative_emb:
            assert not cat_pos
            self.emb_module = RotaryEmbedding(dim_head // self.relative_emb_dim, min_freq=min_freq, scale=scale)

    def _init_params(self):
        if self.init_method == 'xavier':
            init_fn = xavier_uniform_
        elif self.init_method == 'orthogonal':
            init_fn = orthogonal_
        else:
            raise Exception('Unknown initialization')

        for param in self.to_qkv.parameters():
            if param.ndim > 1:
                for h in range(self.heads):
                    if self.attn_type == 'fourier':
                        # for v
                        init_fn(param[(self.heads * 2 + h) * self.dim_head:(self.heads * 2 + h + 1) * self.dim_head, :],
                                gain=self.init_gain)
                        param.data[(self.heads * 2 + h) * self.dim_head:(self.heads * 2 + h + 1) * self.dim_head,
                        :] += self.diagonal_weight * \
                              torch.diag(torch.ones(
                                  param.size(-1),
                                  dtype=torch.float32))
                    else:  # for galerkin
                        # for q
                        init_fn(param[h * self.dim_head:(h + 1) * self.dim_head, :], gain=self.init_gain)
                        #
                        param.data[h * self.dim_head:(h + 1) * self.dim_head, :] += self.diagonal_weight * \
                                                                                    torch.diag(torch.ones(
                                                                                        param.size(-1),
                                                                                        dtype=torch.float32))

    def norm_wrt_domain(self, x, norm_fn):
        b = x.shape[0]
        return rearrange(
            norm_fn(rearrange(x, 'b h n d -> (b h) n d')),
            '(b h) n d -> b h n d', b=b)

    def forward(self, x, pos=None, not_assoc=False, padding_mask=None):
        # padding mask will be in shape [b, n, 1], it will indicates which point are padded and should be ignored
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        if pos is None and self.relative_emb:
            raise Exception('Must pass in coordinates when under relative position embedding mode')

        if padding_mask is None:
            if self.attn_type == 'galerkin':
                k = self.norm_wrt_domain(k, self.k_norm)
                v = self.norm_wrt_domain(v, self.v_norm)
            else:  # fourier
                q = self.norm_wrt_domain(q, self.q_norm)
                k = self.norm_wrt_domain(k, self.k_norm)
        else:
            grid_size = torch.sum(padding_mask, dim=[-1, -2]).view(-1, 1, 1, 1)  # [b, 1, 1]
            padding_mask = repeat(padding_mask, 'b n d -> (b h) n d', h=self.heads)  # [b, n, 1]

            if self.use_ln:
                if self.attn_type == 'galerkin':
                    k = self.k_norm(k)
                    v = self.v_norm(v)
                else:  # fourier
                    q = self.q_norm(q)
                    k = self.k_norm(k)
            else:

                if self.attn_type == 'galerkin':
                    k = rearrange(k, 'b h n d -> (b h) n d')
                    v = rearrange(v, 'b h n d -> (b h) n d')

                    k = masked_instance_norm(k, padding_mask)
                    v = masked_instance_norm(v, padding_mask)

                    k = rearrange(k, '(b h) n d -> b h n d', h=self.heads)
                    v = rearrange(v, '(b h) n d -> b h n d', h=self.heads)
                else:  # fourier
                    q = rearrange(q, 'b h n d -> (b h) n d')
                    k = rearrange(k, 'b h n d -> (b h) n d')

                    q = masked_instance_norm(q, padding_mask)
                    k = masked_instance_norm(k, padding_mask)

                    q = rearrange(q, '(b h) n d -> b h n d', h=self.heads)
                    k = rearrange(k, '(b h) n d -> b h n d', h=self.heads)

            padding_mask = rearrange(padding_mask, '(b h) n d -> b h n d', h=self.heads)  # [b, h, n, 1]

        q = self.apply_rotary_emb(q, pos)
        k = self.apply_rotary_emb(k, pos)

        if self.cat_pos and not self.relative_emb:
            assert pos.size(-1) == self.pos_dim
            pos = pos.unsqueeze(1)
            pos = pos.repeat([1, self.heads, 1, 1])
            q, k, v = [torch.cat([pos, x], dim=-1) for x in (q, k, v)]

        if not_assoc:
            # this is more efficient when n<<c
            score = torch.matmul(q, k.transpose(-1, -2))
            if padding_mask is not None:
                padding_mask = ~padding_mask
                padding_mask_arr = torch.matmul(padding_mask, padding_mask.transpose(-1, -2))  # [b, h, n, n]
                mask_value = 0.
                score = score.masked_fill(padding_mask_arr, mask_value)
                out = torch.matmul(score, v) * (1. / grid_size)
            else:
                out = torch.matmul(score, v) * (1. / q.shape[2])
        else:
            if padding_mask is not None:
                q = q.masked_fill(~padding_mask, 0)
                k = k.masked_fill(~padding_mask, 0)
                v = v.masked_fill(~padding_mask, 0)
                dots = torch.matmul(k.transpose(-1, -2), v)
                out = torch.matmul(q, dots) * (1. / grid_size)
            else:
                dots = torch.matmul(k.transpose(-1, -2), v)
                out = torch.matmul(q, dots) * (1. / q.shape[2])
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def apply_rotary_emb(self, qk, pos_x):
        if self.relative_emb:
            freqs = []
            for i in range(self.relative_emb_dim):
                freq_x_i = self.emb_module(pos_x[:, :, i], qk.device)
                freq_x_i = repeat(freq_x_i, 'b t d -> b h t d', h=self.heads)
                freqs.append(freq_x_i)

            qk = apply_rotary_pos_emb_multi(qk, freqs)
        return qk


class CrossLinearAttention(nn.Module):
    def __init__(self,
                 dim,
                 attn_type,  # ['fourier', 'galerkin']
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 init_params=True,
                 relative_emb=False,
                 scale=1.,
                 init_method='orthogonal',  # ['xavier', 'orthogonal']
                 init_gain=None,
                 relative_emb_dim=2,
                 min_freq=1 / 64,  # 1/64 is for 64 x 64 ns2d,
                 cat_pos=False,
                 pos_dim=2,
                 use_ln=False,
                 ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.attn_type = attn_type
        self.use_ln = use_ln

        self.heads = heads
        self.dim_head = dim_head

        # query is the classification token
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        if attn_type == 'galerkin':
            if not self.use_ln:
                self.k_norm = nn.InstanceNorm1d(dim_head)
                self.v_norm = nn.InstanceNorm1d(dim_head)
            else:
                self.k_norm = nn.LayerNorm(dim_head)
                self.v_norm = nn.LayerNorm(dim_head)

        elif attn_type == 'fourier':
            if not self.use_ln:
                self.q_norm = nn.InstanceNorm1d(dim_head)
                self.k_norm = nn.InstanceNorm1d(dim_head)
            else:
                self.q_norm = nn.LayerNorm(dim_head)
                self.k_norm = nn.LayerNorm(dim_head)

        else:
            raise Exception(f'Unknown attention type {attn_type}')

        if not cat_pos:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim + pos_dim * heads, dim),
                nn.Dropout(dropout)
            )

        if init_gain is None:
            self.init_gain = 1. / dim_head
            self.diagonal_weight = 1. / dim_head
        else:
            self.init_gain = init_gain
            self.diagonal_weight = init_gain
        self.init_method = init_method
        if init_params:
            self._init_params()

        self.cat_pos = cat_pos
        self.pos_dim = pos_dim

        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        if relative_emb:
            self.emb_module = RotaryEmbedding(dim_head // self.relative_emb_dim, min_freq=min_freq, scale=scale)

    def _init_params(self):
        if self.init_method == 'xavier':
            init_fn = xavier_uniform_
        elif self.init_method == 'orthogonal':
            init_fn = orthogonal_
        else:
            raise Exception('Unknown initialization')

        for param in self.to_kv.parameters():
            if param.ndim > 1:
                for h in range(self.heads):
                    # for k
                    init_fn(param[h * self.dim_head:(h + 1) * self.dim_head, :], gain=self.init_gain)
                    param.data[h * self.dim_head:(h + 1) * self.dim_head, :] += self.diagonal_weight * \
                                                                                torch.diag(torch.ones(
                                                                                    param.size(-1),
                                                                                    dtype=torch.float32))

                    # for v
                    init_fn(param[(self.heads + h) * self.dim_head:(self.heads + h + 1) * self.dim_head, :],
                            gain=self.init_gain)
                    param.data[(self.heads + h) * self.dim_head:(self.heads + h + 1) * self.dim_head,
                    :] += self.diagonal_weight * \
                          torch.diag(torch.ones(
                              param.size(-1), dtype=torch.float32))

        for param in self.to_q.parameters():
            if param.ndim > 1:
                for h in range(self.heads):
                    # for q
                    init_fn(param[h * self.dim_head:(h + 1) * self.dim_head, :], gain=self.init_gain)
                    param.data[h * self.dim_head:(h + 1) * self.dim_head, :] += self.diagonal_weight * \
                                                                                torch.diag(torch.ones(
                                                                                    param.size(-1),
                                                                                    dtype=torch.float32))

    def norm_wrt_domain(self, x, norm_fn):
        b = x.shape[0]
        return rearrange(
            norm_fn(rearrange(x, 'b h n d -> (b h) n d')),
            '(b h) n d -> b h n d', b=b)

    def forward(self, x, z, x_pos=None, z_pos=None, padding_mask=None):
        # x (z^T z)
        # x [b, n1, d]
        # z [b, n2, d]
        n1 = x.shape[1]  # x [b, n1, d]
        n2 = z.shape[1]  # z [b, n2, d]
        if padding_mask is not None:
            grid_size = torch.sum(padding_mask, dim=1).view(-1, 1, 1, 1)

        q = self.to_q(x)

        kv = self.to_kv(z).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)

        if (x_pos is None or z_pos is None) and self.relative_emb:
            raise Exception('Must pass in coordinates when under relative position embedding mode')
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        if padding_mask is None:
            if self.attn_type == 'galerkin':
                k = self.norm_wrt_domain(k, self.k_norm)
                v = self.norm_wrt_domain(v, self.v_norm)
            else:  # fourier
                q = self.norm_wrt_domain(q, self.q_norm)
                k = self.norm_wrt_domain(k, self.k_norm)
        else:
            padding_mask = repeat(padding_mask, 'b n d -> (b h) n d', h=self.heads)  # [b, n, 1]
            if self.use_ln:
                if self.attn_type == 'galerkin':
                    k = self.k_norm(k)
                    v = self.v_norm(v)
                else:  # fourier
                    q = self.q_norm(q)
                    k = self.k_norm(k)
            else:

                if self.attn_type == 'galerkin':
                    k = rearrange(k, 'b h n d -> (b h) n d')
                    v = rearrange(v, 'b h n d -> (b h) n d')

                    k = masked_instance_norm(k, padding_mask)
                    v = masked_instance_norm(v, padding_mask)

                    k = rearrange(k, '(b h) n d -> b h n d', h=self.heads)
                    v = rearrange(v, '(b h) n d -> b h n d', h=self.heads)
                else:  # fourier
                    q = rearrange(q, 'b h n d -> (b h) n d')
                    k = rearrange(k, 'b h n d -> (b h) n d')

                    q = masked_instance_norm(q, padding_mask)
                    k = masked_instance_norm(k, padding_mask)

                    q = rearrange(q, '(b h) n d -> b h n d', h=self.heads)
                    k = rearrange(k, '(b h) n d -> b h n d', h=self.heads)

            padding_mask = rearrange(padding_mask, '(b h) n d -> b h n d', h=self.heads)  # [b, h, n, 1]

        q = self.apply_rotary_emb(q, x_pos)
        k = self.apply_rotary_emb(k, z_pos)

        if self.cat_pos and not self.relative_emb:
            assert x_pos.size(-1) == self.pos_dim and z_pos.size(-1) == self.pos_dim
            x_pos = x_pos.unsqueeze(1)
            x_pos = x_pos.repeat([1, self.heads, 1, 1])
            q = torch.cat([x_pos, q], dim=-1)

            z_pos = z_pos.unsqueeze(1)
            z_pos = z_pos.repeat([1, self.heads, 1, 1])
            k = torch.cat([z_pos, k], dim=-1)
            v = torch.cat([z_pos, v], dim=-1)

        if padding_mask is not None:
            q = q.masked_fill(~padding_mask, 0)
            k = k.masked_fill(~padding_mask, 0)
            v = v.masked_fill(~padding_mask, 0)
            dots = torch.matmul(k.transpose(-1, -2), v)
            out = torch.matmul(q, dots) * (1. / grid_size)
        else:
            dots = torch.matmul(k.transpose(-1, -2), v)
            out = torch.matmul(q, dots) * (1. / n2)

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

    def apply_rotary_emb(self, qk, pos_x):
        if self.relative_emb:
            freqs = []
            for i in range(self.relative_emb_dim):
                freq_x_i = self.emb_module(pos_x[:, :, i], qk.device)
                freq_x_i = repeat(freq_x_i, 'b t d -> b h t d', h=self.heads)
                freqs.append(freq_x_i)

            qk = apply_rotary_pos_emb_multi(qk, freqs)
        return qk


class TransformerCatNoCls(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 attn_type,  # ['standard', 'galerkin', 'fourier']
                 use_ln=False,
                 scale=16,  # can be list, or an int
                 dropout=0.,
                 relative_emb_dim=2,
                 min_freq=1 / 64,
                 attention_init='orthogonal',
                 init_gain=None,
                 use_relu=False,
                 cat_pos=False,
                 ):
        super().__init__()
        assert attn_type in ['standard', 'galerkin', 'fourier']

        if isinstance(scale, int):
            scale = [scale] * depth
        assert len(scale) == depth

        self.layers = nn.ModuleList([])
        self.attn_type = attn_type
        self.use_ln = use_ln

        if attn_type == 'standard':
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList([
                        PreNorm(dim, StandardAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)
                        if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout))]),
                )
        else:
            for d in range(depth):
                if scale[d] != -1 or not cat_pos:
                    attn_module = LinearAttention(dim, attn_type,
                                                  heads=heads, dim_head=dim_head, dropout=dropout,
                                                  relative_emb=True, scale=scale[d],
                                                  relative_emb_dim=relative_emb_dim,
                                                  min_freq=min_freq,
                                                  init_method=attention_init,
                                                  init_gain=init_gain,
                                                  use_ln=False,
                                                  )
                else:
                    attn_module = LinearAttention(dim, attn_type,
                                                  heads=heads, dim_head=dim_head, dropout=dropout,
                                                  cat_pos=True,
                                                  pos_dim=relative_emb_dim,
                                                  relative_emb=False,
                                                  init_method=attention_init,
                                                  init_gain=init_gain
                                                  )
                if not use_ln:
                    self.layers.append(
                        nn.ModuleList([attn_module,
                                       FeedForward(dim, mlp_dim, dropout=dropout)
                                       if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout)
                                       ]),
                    )
                else:
                    self.layers.append(
                        nn.ModuleList([
                            nn.LayerNorm(dim),
                            attn_module,
                            nn.LayerNorm(dim),
                            FeedForward(dim, mlp_dim, dropout=dropout)
                            if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout),
                        ]),
                    )

    def forward(self, x, pos_embedding):
        # x in [b n c], pos_embedding in [b n 2]

        for layer_no, attn_layer in enumerate(self.layers):
            if not self.use_ln:
                [attn, ffn] = attn_layer
                x = attn(x, pos_embedding) + x
                x = ffn(x) + x
            else:
                [ln1, attn, ln2, ffn] = attn_layer
                x = ln1(x)
                x = attn(x, pos_embedding) + x
                x = ln2(x)
                x = ffn(x) + x
        return x


class IrregSTEncoder(torch.nn.Module):
    # for time dependent airfoil
    def __init__(self, hparams):
        super().__init__()
        self.tw = hparams.time_window
        # here, assume the input is in the shape [b, t, n, c_in]
        self.to_embedding = nn.Sequential(
            Rearrange('b t n c -> b c t n'),
            nn.Conv2d(hparams.input_channels, hparams.in_emb_dim, kernel_size=(self.tw, 1), stride=(self.tw, 1),
                      padding=(0, 0), bias=False),
            nn.GELU(),
            nn.Conv2d(hparams.in_emb_dim, hparams.in_emb_dim, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0), bias=False),
            Rearrange('b c 1 n -> b n c'),  # [b, n, c_out]
        )

        self.node_embedding = nn.Embedding(hparams.max_node_type, hparams.in_emb_dim)

        self.combine_embedding = nn.Linear(hparams.in_emb_dim * 2, hparams.in_emb_dim, bias=False)

        self.dropout = nn.Dropout(hparams.emb_dropout)

        if hparams.depth > 4:
            self.s_transformer = TransformerCatNoCls(hparams.in_emb_dim,
                                                     hparams.depth,
                                                     hparams.heads,
                                                     hparams.in_emb_dim,
                                                     hparams.in_emb_dim,
                                                     'galerkin', hparams.use_ln,
                                                     scale=[32, 16, 8, 8] + [1] * (hparams.depth - 4),
                                                     relative_emb_dim=hparams.relative_emb_dim,
                                                     min_freq=1 / hparams.res,
                                                     attention_init='orthogonal')
        else:
            self.s_transformer = TransformerCatNoCls(hparams.in_emb_dim,
                                                     hparams.depth,
                                                     hparams.heads,
                                                     hparams.in_emb_dim,
                                                     hparams.in_emb_dim,
                                                     'galerkin',
                                                     hparams.use_ln,
                                                     scale=[32] + [16] * (hparams.depth - 2) + [1],
                                                     relative_emb_dim=hparams.relative_emb_dim,
                                                     min_freq=1 / hparams.res,
                                                     attention_init='orthogonal')

        self.ln = nn.LayerNorm(hparams.in_emb_dim)

        self.to_out = nn.Sequential(
            nn.Linear(hparams.in_emb_dim, hparams.in_emb_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hparams.in_emb_dim, hparams.out_channels, bias=False),
        )

    def forward(self,
                x,  # [b, t, n, c]
                node_type,  # [b, n, 1]
                input_pos,  # [b, n, 2]
                ):
        x = self.to_embedding(x)
        x_node = self.node_embedding(node_type.squeeze(-1))
        x = self.combine_embedding(torch.cat([x, x_node], dim=-1))
        x_skip = x

        x = self.dropout(x)
        x = self.s_transformer.forward(x, input_pos)
        x = self.ln(x + x_skip)
        x = self.to_out(x)

        return x


# code copied from: https://github.com/ndahlquist/pytorch-fourier-feature-networks
# author: Nic Dahlquist
class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, n, num_input_channels],
     returns a tensor of size [batches, n, mapping_size*2].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale, requires_grad=False)

    def forward(self, x):
        batches, num_of_points, channels = x.shape

        # Make shape compatible for matmul with _B.
        # From [B, N, C] to [(B*N), C].
        x = rearrange(x, 'b n c -> (b n) c')

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = rearrange(x, '(b n) c -> b n c', b=batches)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class CrossFormer(nn.Module):
    def __init__(self,
                 dim,
                 attn_type,
                 heads,
                 dim_head,
                 mlp_dim,
                 residual=True,
                 use_ffn=True,
                 use_ln=False,
                 relative_emb=False,
                 scale=1.,
                 relative_emb_dim=2,
                 min_freq=1 / 64,
                 dropout=0.,
                 cat_pos=False,
                 ):
        super().__init__()

        self.cross_attn_module = CrossLinearAttention(dim, attn_type,
                                                      heads=heads, dim_head=dim_head, dropout=dropout,
                                                      relative_emb=relative_emb,
                                                      scale=scale,
                                                      relative_emb_dim=relative_emb_dim,
                                                      min_freq=min_freq,
                                                      init_method='orthogonal',
                                                      cat_pos=cat_pos,
                                                      pos_dim=relative_emb_dim,
                                                      use_ln=False
                                                      )
        self.use_ln = use_ln
        self.residual = residual
        self.use_ffn = use_ffn

        if self.use_ln:
            self.ln1 = nn.LayerNorm(dim)
            self.ln2 = nn.LayerNorm(dim)

        if self.use_ffn:
            self.ffn = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, z, x_pos=None, z_pos=None):
        # x in [b n1 c]
        # b, n1, c = x.shape   # coordinate encoding
        # b, n2, c = z.shape   # system encoding
        if self.use_ln:
            z = self.ln1(z)
            if self.residual:
                x = self.ln2(self.cross_attn_module(x, z, x_pos, z_pos)) + x
            else:
                x = self.ln2(self.cross_attn_module(x, z, x_pos, z_pos))
        else:
            if self.residual:
                x = self.cross_attn_module(x, z, x_pos, z_pos) + x
            else:
                x = self.cross_attn_module(x, z, x_pos, z_pos)

        if self.use_ffn:
            x = self.ffn(x) + x

        return x


class IrregSTDecoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.out_channels = hparams.out_channels
        self.latent_channels = hparams.latent_channels

        self.node_type_embedding = nn.Embedding(hparams.max_node_type, hparams.latent_channels)
        space_dim = hparams.relative_emb_dim

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(space_dim, self.latent_channels // 2, scale=hparams.scale),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
        )

        self.combine_layer = nn.Linear(self.latent_channels * 2, self.latent_channels, bias=False)

        self.input_dropout = nn.Dropout(hparams.dropout)

        self.decoding_transformer = CrossFormer(self.latent_channels, 'galerkin', 4,
                                                self.latent_channels, self.latent_channels,
                                                relative_emb=True,
                                                scale=32.,
                                                relative_emb_dim=space_dim,
                                                min_freq=1 / hparams.res)

        self.mix_layer = LinearAttention(self.latent_channels, 'galerkin',
                                         heads=1, dim_head=self.latent_channels,
                                         relative_emb=True,
                                         scale=32,
                                         relative_emb_dim=space_dim,
                                         min_freq=1 / hparams.res,
                                         use_ln=False)

        self.expand_layer = nn.Linear(self.latent_channels, self.latent_channels * 2, bias=False)

        self.propagator = nn.ModuleList([
            nn.ModuleList([nn.LayerNorm(self.latent_channels * 2),
                           nn.Sequential(
                               nn.Linear(self.latent_channels * 3 + space_dim, self.latent_channels * 2, bias=False),
                               nn.GELU(),
                               nn.Linear(self.latent_channels * 2, self.latent_channels * 2, bias=False),
                               nn.GELU(),
                               nn.Linear(self.latent_channels * 2, self.latent_channels * 2, bias=False),
                               nn.GELU(),
                               nn.Linear(self.latent_channels * 2, self.latent_channels * 2, bias=False)
                           )])
        ])

        self.out_norm = nn.LayerNorm(self.latent_channels * 2)
        self.to_out = nn.Sequential(
            nn.Linear(self.latent_channels * 3, self.latent_channels * 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_channels * 2, self.latent_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_channels, self.out_channels, bias=True))
        self.decoder_norm = None

    def propagate(self, z, z_node, prop_pos):
        for layer in self.propagator:
            norm_fn, ffn = layer
            z = ffn(torch.cat((norm_fn(z), z_node, prop_pos), dim=-1)) + z
        return z

    def decode(self, z, z_node):
        z = self.out_norm(z)
        z = self.to_out(torch.cat((z, z_node), dim=-1))
        return z

    def forward(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 2]
                prop_node_type,  # [b, n, 1]
                forward_steps,
                input_pos):
        history = []
        x_node = self.node_type_embedding(prop_node_type.squeeze(-1))
        x = self.coordinate_projection.forward(propagate_pos)
        x = self.combine_layer(torch.cat((x, x_node), dim=-1))

        z = self.input_dropout(z)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z = self.mix_layer.forward(z, propagate_pos) + z
        z = self.expand_layer(z)

        # forward the dynamics in the latent space
        for step in range(forward_steps):
            z = self.propagate(z, x_node, propagate_pos)
            u = self.decode(z, x_node)
            history.append(u)

        history = torch.stack(history, dim=1)  # concatenate in temporal dimension

        if self.decoder_norm is not None:
            history = self.decoder_norm(history)

        return history

    def set_norm_decoder(self, decoder_norm):
        self.decoder_norm = decoder_norm  # unnormalize the output of the model if needed


class PlOformer(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = IrregSTEncoder(hparams.encoder)
        self.decoder = IrregSTDecoder(hparams.decoder)

        self.time_history = hparams.get('time_history', 128)

        self.loss = hparams.loss
        self.criterion = MultiLoss()  # MSE loss
        self.mae_criterion = DownsampledLoss(type='l1')  # nn.L1Loss()

        # normalization parameters to be loaded with model weights
        self.normalization = 'gauss'
        self.norm_input = True  # the flags correspond to the normalization of inputs and targets in dataloader
        self.norm_target = True
        self.normalizer_input = Normalizer(stats_shape=tuple(hparams.norm_shape))
        self.normalizer_target = Normalizer(stats_shape=tuple(hparams.norm_shape))

        self.correlation = CorrelationLoss()
        self.scaled_mae = ScaledMaeLoss()

        pde_loss_f, pde_loss_sim_f = get_pde_loss_function(system='swe', flip_xy=False)  # the loss is overriden by set_pde_loss
        self.pde_loss = pde_loss_f
        self.pde_loss_simulator = pde_loss_sim_f

        # Optimization parameters
        self.lr = hparams.lr
        self.weight_decay = hparams.weight_decay

        self.curriculum_steps = hparams.curriculum_steps
        self.curriculum_ratio = hparams.curriculum_ratio

    def set_pde_loss_function(self, system, flip_xy):
        Tn_mult = self.time_history / 128
        pde_loss_f, pde_loss_sim_f = get_pde_loss_function(system, flip_xy, Tn_mult=Tn_mult)

        self.pde_loss = pde_loss_f
        self.pde_loss_simulator = pde_loss_sim_f

    def setup(self, stage: str = None) -> None:
        def remove_dim(t):
            if len(t.shape) == 1 and t.shape[0] == 1:
                return t.squeeze(0)
            else:
                return t

        stats = self.trainer.datamodule.get_norm_stats()
        self.norm_input = stats.norm_input
        self.norm_target = stats.norm_target
        # if stage == "fit":
        # set the data normalization statistics
        if self.normalization == "min_max":
            self.normalizer_input.set_stats(remove_dim(stats["input_min"]), remove_dim(stats["input_min_max"]))
            self.normalizer_target.set_stats(remove_dim(stats["target_min"]), remove_dim(stats["target_min_max"]))
        else:
            self.normalizer_input.set_stats(remove_dim(stats["input_mean"]), remove_dim(stats["input_std"]))
            self.normalizer_target.set_stats(remove_dim(stats["target_mean"]), remove_dim(stats["target_std"]))

        return

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                      lr=self.lr,
                                      weight_decay=self.weight_decay)
        scheduler = OneCycleLR(optimizer, max_lr=self.lr,
                               div_factor=1e4,
                               pct_start=0.3,
                               final_div_factor=1e4,
                               total_steps=self.trainer.estimated_stepping_batches
                               )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            },
        }

    def get_unnorm_target(self, s):
        if self.norm_target:
            s_unnorm = self.normalizer_target(s, inverse=True)
        else:
            s_unnorm = s
            s = self.normalizer_target(s, inverse=False)
        return s, s_unnorm

    def forward(self, x, node_type, input_pos, prop_pos, forward_steps):
        z = self.encoder.forward(x, node_type, input_pos)
        pred = self.decoder.forward(z, prop_pos, node_type, forward_steps, input_pos)
        return pred

    def truncate_by_t_history(self, x, y, node_type, pos, n_time):
        # truncate the history of the airfoil data
        if 0 < self.time_history < n_time:
            x = x.reshape(x.shape[0], n_time, -1, x.shape[-1])
            x = x[:, :self.time_history]
            x = x.reshape(x.shape[0], 1, -1, x.shape[-1])

            y = y.reshape(y.shape[0], n_time, -1, y.shape[-1])
            y = y[:, :self.time_history]
            y = y.reshape(y.shape[0], 1, -1, y.shape[-1])

            node_type = node_type.reshape(node_type.shape[0], n_time, -1, node_type.shape[-1])
            node_type = node_type[:, :self.time_history]
            node_type = node_type.reshape(node_type.shape[0], -1, node_type.shape[-1])

            pos = pos.reshape(pos.shape[0], n_time, -1, pos.shape[-1])
            pos = pos[:, :self.time_history]
            pos = pos.reshape(pos.shape[0], -1, pos.shape[-1])

            n_time = self.time_history
        else:
            n_time = n_time[0].item()

        return x, y, node_type, pos, n_time

    def training_step(self, train_batch, batch_idx):
        x, y, node_type, pos, n_time = train_batch
        forward_steps = y.shape[1]
        if forward_steps == 1:
            x, y, node_type, pos, n_time = self.truncate_by_t_history(x, y, node_type, pos, n_time)

        y, y_unnorm = self.get_unnorm_target(y)
        input_pos = prop_pos = pos

        curriculum_limit = int(self.curriculum_ratio * self.trainer.estimated_stepping_batches)
        n_iter = self.trainer.global_step
        if self.curriculum_steps > 0 and n_iter < curriculum_limit:
            progress = (n_iter * 2) / curriculum_limit
            c_steps = self.curriculum_steps + \
                       int(max(0, progress - 1.) * ((forward_steps - self.curriculum_steps) / 2.)) * 2

            y = y[:, :c_steps]  # [b t n]
            forward_steps = c_steps

        y_pred = self.forward(x, node_type, input_pos, prop_pos, forward_steps)  # [b t n c]
        loss = self.criterion(y_pred, y, prop_pos) if self.loss == 'airfoil' else self.criterion(y_pred, y)

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, val_batch, batch_idx):
        # cells can be returned by dataloader in val_batch and
        # they are used for plotting only, I would need them to implement custom callback for visualization
        x, y, node_type, pos, n_time = val_batch
        forward_steps = y.shape[1]
        if forward_steps == 1:
            x, y, node_type, pos, n_time = self.truncate_by_t_history(x, y, node_type, pos, n_time)
        y, y_unnorm = self.get_unnorm_target(y)

        input_pos = prop_pos = pos
        y_pred = self.forward(x, node_type, input_pos, prop_pos, forward_steps)  # [b t n c]

        loss = self.criterion(y_pred, y, prop_pos) if self.loss == 'airfoil' else self.criterion(y_pred, y)
        mae_loss = self.mae_criterion(y_pred, y)

        y_pred_unnorm = self.normalizer_target(y_pred, inverse=True)
        mae_un_loss = self.mae_criterion(y_pred_unnorm, y_unnorm)
        corr = self.correlation(y_pred, y)
        corr = torch.mean(corr)

        # naming is the same as for other models
        loss_u_scaled = self.scaled_mae(y_pred, y)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_mae_u', mae_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_mae_u_un', mae_un_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_corr', corr, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_mae_u_scaled', loss_u_scaled, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        if self.loss == 'airfoil':
            return {'loss': loss}
        else:
            if forward_steps == 1:
                # [b, 1, n_time x n_x, c] -> [b, n_time, n_x, c]
                y_pred = y_pred.reshape(y_pred.shape[0], n_time, -1, y_pred.shape[-1])
                y = y.reshape(y_pred.shape[0], n_time, -1, y_pred.shape[-1])

                x_in = x.reshape(x.shape[0], n_time, -1, x.shape[-1])
                x_in = x_in[..., 0:-2]   # do not take x and t added to the input

                pde_loss = self.get_pde_loss(x_in, y_pred, clamp_loss=False, reduce=True)
                pde_loss_gt = self.get_pde_loss(x_in, y, clamp_loss=False, reduce=True)

                self.log('val_pde_loss', pde_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
                self.log('val_pde_loss_gt', pde_loss_gt, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            return {'pred': y_pred, 'target': y, 'loss': loss}

    def test_step(self, test_batch, batch_idx):
        x, y, node_type, pos, n_time = test_batch
        forward_steps = y.shape[1]
        if forward_steps == 1:
            x, y, node_type, pos, n_time = self.truncate_by_t_history(x, y, node_type, pos, n_time)

        y, y_unnorm = self.get_unnorm_target(y)
        down_factor = self.trainer.datamodule.down_factor if self.trainer.datamodule.down_interp else 1

        input_pos = prop_pos = pos
        y_pred = self.forward(x, node_type, input_pos, prop_pos, forward_steps)  # [b t n c]

        loss = self.criterion(y_pred, y, prop_pos) if self.loss == 'airfoil' else self.criterion(y_pred, y)
        mae_loss = self.mae_criterion(y_pred, y, down_factor)

        y_pred_unnorm = self.normalizer_target(y_pred, inverse=True)
        mae_un_loss = self.mae_criterion(y_pred_unnorm, y_unnorm, down_factor)
        corr = self.correlation(y_pred, y)
        corr = torch.mean(corr)

        # naming is the same as for other models
        loss_u_scaled = self.scaled_mae(y_pred, y)

        self.log('test_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('test_mae_u', mae_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('test_mae_u_un', mae_un_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('test_corr', corr, prog_bar=True, on_epoch=True, on_step=False)
        self.log('test_mae_u_scaled', loss_u_scaled, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        if self.loss == 'airfoil':
            return {'loss': loss}
        else:
            if forward_steps == 1:
                # [b, 1, n_time x n_x, c] -> [b, n_time, n_x, c]
                y_pred = y_pred.reshape(y_pred.shape[0], n_time, -1, y_pred.shape[-1])
                y = y.reshape(y_pred.shape[0], n_time, -1, y_pred.shape[-1])

                x_in = x.reshape(x.shape[0], n_time, -1, x.shape[-1])
                x_in = x_in[..., 0:-2]   # do not take x and t added to the input

                pde_loss = self.get_pde_loss(x_in, y_pred, clamp_loss=False, reduce=True)
                pde_loss_gt = self.get_pde_loss(x_in, y, clamp_loss=False, reduce=True)

                self.log('test_pde_loss', pde_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
                self.log('test_pde_loss_gt', pde_loss_gt, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

            return {'pred': y_pred, 'target': y, 'loss': loss}

    def get_pde_loss(self, cond, pred, x_gt_unnorm=None, clamp_loss=False, reduce=True):
        cond_unnorm = self.normalizer_input(cond, inverse=True)
        pred_unnorm = self.normalizer_target(pred, inverse=True)
        x_unnorm = torch.cat([cond_unnorm, pred_unnorm], dim=-1)

        if x_gt_unnorm is None:
            x_gt_unnorm = x_unnorm

        pde_error_dx_matrix = self.pde_loss(x_unnorm, x_gt_unnorm, self.normalizer_input, self.normalizer_target,
                                            return_d=False, calc_prob=False, clamp_loss=clamp_loss)

        if reduce:
            n_batch = cond_unnorm.shape[0]
            pde_loss = torch.sum(pde_error_dx_matrix) / n_batch
        else:
            pde_loss = pde_error_dx_matrix

        return pde_loss


class PlOformerTimePred(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = IrregSTEncoder(hparams.encoder)
        self.decoder = IrregSTDecoder(hparams.decoder)

        self.loss = hparams.loss
        self.criterion = MultiLoss()  # MSE loss
        self.mae_criterion = DownsampledLoss(type='l1')  # nn.L1Loss()

        # normalization parameters to be loaded with model weights
        self.normalization = 'gauss'
        self.norm_input = True  # the flags correspond to the normalization of inputs and targets in dataloader
        self.norm_target = True
        self.normalizer_input = Normalizer(stats_shape=tuple(hparams.norm_shape))
        self.normalizer_target = Normalizer(stats_shape=tuple(hparams.norm_shape))

        self.normalizer_state1 = Normalizer(stats_shape=tuple(hparams.norm_shape))
        self.normalizer_state2 = Normalizer(stats_shape=tuple(hparams.norm_shape))

        self.correlation = CorrelationLoss()
        self.scaled_mae = ScaledMaeLoss()

        pde_loss_f, pde_loss_sim_f = get_pde_loss_function(system='swe',
                                                           flip_xy=False)  # the loss is overriden by set_pde_loss
        self.pde_loss = pde_loss_f
        self.pde_loss_simulator = pde_loss_sim_f

        # Optimization parameters
        self.lr = hparams.lr
        self.weight_decay = hparams.weight_decay

        self.curriculum_steps = hparams.curriculum_steps
        self.curriculum_ratio = hparams.curriculum_ratio

    def set_pde_loss_function(self, system, flip_xy):
        pde_loss_f, pde_loss_sim_f = get_pde_loss_function(system, flip_xy)

        self.pde_loss = pde_loss_f
        self.pde_loss_simulator = pde_loss_sim_f

    def setup(self, stage: str = None) -> None:
        def unsqueeze_zero(x):
            if len(x.size()) == 0:
                return x.unsqueeze(0)
            return x

        stats = self.trainer.datamodule.get_norm_stats()
        self.norm_input = stats.norm_input
        self.norm_target = stats.norm_target
        # if stage == "fit":
        # set the data normalization statistics
        # since inputs and targets are combined together here, we need to set the normalization stats for both
        if self.normalization == "min_max":
            min_val = torch.cat([unsqueeze_zero(stats["input_min"]),
                                 unsqueeze_zero(stats["target_min"])], dim=-1)
            min_max_val = torch.cat([unsqueeze_zero(stats["input_min_max"]),
                                     unsqueeze_zero(stats["target_min_max"])], dim=-1)
            self.normalizer_input.set_stats(min_val, min_max_val)
            self.normalizer_target.set_stats(min_val, min_max_val)

            self.normalizer_state1.set_stats(stats["input_min"], stats["input_min_max"])
            self.normalizer_state2.set_stats(stats["target_min"], stats["target_min_max"])
        else:
            mean = torch.cat([unsqueeze_zero(stats["input_mean"]),
                              unsqueeze_zero(stats["target_mean"])], dim=-1)
            std = torch.cat([unsqueeze_zero(stats["input_std"]),
                             unsqueeze_zero(stats["target_std"])], dim=-1)
            self.normalizer_input.set_stats(mean, std)
            self.normalizer_target.set_stats(mean, std)

            self.normalizer_state1.set_stats(stats["input_mean"], stats["input_std"])
            self.normalizer_state2.set_stats(stats["target_mean"], stats["target_std"])

        return

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                      lr=self.lr,
                                      weight_decay=self.weight_decay)
        scheduler = OneCycleLR(optimizer, max_lr=self.lr,
                               div_factor=1e4,
                               pct_start=0.3,
                               final_div_factor=1e4,
                               total_steps=self.trainer.estimated_stepping_batches
                               )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            },
        }

    def get_unnorm_target(self, s):
        if self.norm_target:
            s_unnorm = self.normalizer_target(s, inverse=True)
        else:
            s_unnorm = s
            s = self.normalizer_target(s, inverse=False)
        return s, s_unnorm

    def forward(self, x, node_type_inp, node_type_prop, input_pos, prop_pos, forward_steps):

        z = self.encoder.forward(x, node_type_inp, input_pos)
        pred = self.decoder.forward(z, prop_pos, node_type_prop, forward_steps, input_pos)
        return pred

    def training_step(self, train_batch, batch_idx):
        x, y, node_type_inp, node_type_prop, input_pos, prop_pos, n_time = train_batch

        y, y_unnorm = self.get_unnorm_target(y)
        forward_steps = y.shape[1]

        curriculum_limit = int(self.curriculum_ratio * self.trainer.estimated_stepping_batches)
        n_iter = self.trainer.global_step
        if self.curriculum_steps > 0 and n_iter < curriculum_limit:
            progress = (n_iter * 2) / curriculum_limit
            c_steps = self.curriculum_steps + \
                      int(max(0, progress - 1.) * ((forward_steps - self.curriculum_steps) / 2.)) * 2

            y = y[:, :c_steps]  # [b t n]
            forward_steps = c_steps

        y_pred = self.forward(x, node_type_inp, node_type_prop, input_pos, prop_pos, forward_steps)  # [b t n c]
        loss = self.criterion(y_pred, y, prop_pos) if self.loss == 'airfoil' else self.criterion(y_pred, y)

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, val_batch, batch_idx):
        # cells can be returned by dataloader in val_batch and
        # they are used for plotting only, I would need them to implement custom callback for visualization
        x, y, node_type_inp, node_type_prop, input_pos, prop_pos, n_time = val_batch
        y, y_unnorm = self.get_unnorm_target(y)

        forward_steps = y.shape[1]
        y_pred = self.forward(x, node_type_inp, node_type_prop, input_pos, prop_pos, forward_steps)  # [b t n c]

        loss = self.criterion(y_pred, y, prop_pos) if self.loss == 'airfoil' else self.criterion(y_pred, y)
        mae_loss = self.mae_criterion(y_pred, y)

        y_pred_unnorm = self.normalizer_target(y_pred, inverse=True)
        mae_un_loss = self.mae_criterion(y_pred_unnorm, y_unnorm)
        corr = self.correlation(y_pred, y)
        corr = torch.mean(corr)

        # naming is the same as for other models
        loss_u_scaled = self.scaled_mae(y_pred, y)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_mae_u', mae_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_mae_u_un', mae_un_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_corr', corr, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_mae_u_scaled', loss_u_scaled, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        if self.loss == 'airfoil':
            return {'loss': loss}
        else:
            if forward_steps == 1:
                # [b, 1, n_time x n_x, c] -> [b, n_time, n_x, c]
                n_time = n_time[0].item()
                y_pred = y_pred.reshape(y_pred.shape[0], n_time, -1, y_pred.shape[-1])
                y = y.reshape(y_pred.shape[0], n_time, -1, y_pred.shape[-1])

                x_in = x.reshape(x.shape[0], n_time, -1, x.shape[-1])
                x_in = x_in[..., 0:-2]  # do not take x and t added to the input

                # concatenate input and target via time dimension to calculate the pde loss
                state_full_pred = torch.cat([x_in, y_pred], dim=1)
                state_full_gt = torch.cat([x_in, y], dim=1)

                pde_loss = self.get_pde_loss(state_full_pred, clamp_loss=False, reduce=True)
                pde_loss_gt = self.get_pde_loss(state_full_gt, clamp_loss=False, reduce=True)

                self.log('val_pde_loss', pde_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
                self.log('val_pde_loss_gt', pde_loss_gt, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

                y_pred = state_full_pred
                y = state_full_gt

            return {'pred': y_pred, 'target': y, 'loss': loss}

    def test_step(self, test_batch, batch_idx):
        x, y, node_type_inp, node_type_prop, input_pos, prop_pos, n_time = test_batch
        y, y_unnorm = self.get_unnorm_target(y)
        down_factor = self.trainer.datamodule.down_factor if self.trainer.datamodule.down_interp else 1

        forward_steps = y.shape[1]
        y_pred = self.forward(x, node_type_inp, node_type_prop, input_pos, prop_pos, forward_steps)  # [b t n c]

        loss = self.criterion(y_pred, y, prop_pos) if self.loss == 'airfoil' else self.criterion(y_pred, y)
        mae_loss = self.mae_criterion(y_pred, y, down_factor)

        y_pred_unnorm = self.normalizer_target(y_pred, inverse=True)
        mae_un_loss = self.mae_criterion(y_pred_unnorm, y_unnorm, down_factor)
        corr = self.correlation(y_pred, y)
        corr = torch.mean(corr)

        # naming is the same as for other models
        loss_u_scaled = self.scaled_mae(y_pred, y)

        self.log('test_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('test_mae_u', mae_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('test_mae_u_un', mae_un_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('test_corr', corr, prog_bar=True, on_epoch=True, on_step=False)
        self.log('test_mae_u_scaled', loss_u_scaled, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        if self.loss == 'airfoil':
            return {'loss': loss}
        else:
            if forward_steps == 1:
                # [b, 1, n_time x n_x, c] -> [b, n_time, n_x, c]
                n_time = n_time[0].item()
                y_pred = y_pred.reshape(y_pred.shape[0], n_time, -1, y_pred.shape[-1])
                y = y.reshape(y_pred.shape[0], n_time, -1, y_pred.shape[-1])

                x_in = x.reshape(x.shape[0], n_time, -1, x.shape[-1])
                x_in = x_in[..., 0:-2]  # do not take x and t added to the input

                # concatenate input and target via time dimension to calculate the pde loss
                state_full_pred = torch.cat([x_in, y_pred], dim=1)
                state_full_gt = torch.cat([x_in, y], dim=1)

                pde_loss = self.get_pde_loss(state_full_pred, clamp_loss=False, reduce=True)
                pde_loss_gt = self.get_pde_loss(state_full_gt, clamp_loss=False, reduce=True)

                self.log('test_pde_loss', pde_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
                self.log('test_pde_loss_gt', pde_loss_gt, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

                y_pred = state_full_pred
                y = state_full_gt

            return {'pred': y_pred, 'target': y, 'loss': loss}

    def get_pde_loss(self, state, x_gt_unnorm=None, clamp_loss=False, reduce=True):
        x_unnorm = self.normalizer_target(state, inverse=True)  # norm target has states for the full state

        if x_gt_unnorm is None:
            x_gt_unnorm = x_unnorm

        pde_error_dx_matrix = self.pde_loss(x_unnorm, x_gt_unnorm, self.normalizer_state1, self.normalizer_state2,
                                            return_d=False, calc_prob=False, clamp_loss=clamp_loss)

        if reduce:
            n_batch = state.shape[0]
            pde_loss = torch.sum(pde_error_dx_matrix) / n_batch
        else:
            pde_loss = pde_error_dx_matrix

        return pde_loss


class PlOformerStateTimePred(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()

        self.model_state = PlOformer(hparams.hparams_state)
        self.model_time = PlOformerTimePred(hparams.hparams_time)

        self.flip_xy = False

        self.mae_criterion = DownsampledLoss(type='l1')  # nn.L1Loss()
        self.mae_full_criterion = MaskedLoss()

        # normalization parameters to be loaded with model weights
        self.normalization = 'gauss'
        self.norm_input = True  # the flags correspond to the normalization of inputs and targets in dataloader
        self.norm_target = True
        self.normalizer_input = Normalizer(stats_shape=tuple(hparams.norm_shape))
        self.normalizer_target = Normalizer(stats_shape=tuple(hparams.norm_shape))

        self.normalizer_state1 = Normalizer(stats_shape=tuple(hparams.norm_shape))
        self.normalizer_state2 = Normalizer(stats_shape=tuple(hparams.norm_shape))

        pde_loss_f, pde_loss_sim_f = get_pde_loss_function(system='swe',
                                                           flip_xy=False)  # the loss is overriden by set_pde_loss
        self.pde_loss = pde_loss_f
        self.pde_loss_simulator = pde_loss_sim_f

    def set_pde_loss_function(self, system, flip_xy):
        self.flip_xy = flip_xy
        flip_xy = False  # because the predictions will be manually flipped before the future prediction
        pde_loss_f, pde_loss_sim_f = get_pde_loss_function(system, flip_xy)

        self.pde_loss = pde_loss_f
        self.pde_loss_simulator = pde_loss_sim_f

    def setup(self, stage: str = None) -> None:
        def unsqueeze_zero(x):
            if len(x.size()) == 0:
                return x.unsqueeze(0)
            return x

        stats = self.trainer.datamodule.get_norm_stats()
        self.norm_input = stats.norm_input
        self.norm_target = stats.norm_target
        # if stage == "fit":
        # set the data normalization statistics
        # since inputs and targets are combined together here, we need to set the normalization stats for both
        if self.normalization == "min_max":
            min_val = torch.cat([unsqueeze_zero(stats["input_min"]),
                                 unsqueeze_zero(stats["target_min"])], dim=-1)
            min_max_val = torch.cat([unsqueeze_zero(stats["input_min_max"]),
                                     unsqueeze_zero(stats["target_min_max"])], dim=-1)
            self.normalizer_input.set_stats(min_val, min_max_val)
            self.normalizer_target.set_stats(min_val, min_max_val)

            self.normalizer_state1.set_stats(stats["input_min"], stats["input_min_max"])
            self.normalizer_state2.set_stats(stats["target_min"], stats["target_min_max"])
        else:
            mean = torch.cat([unsqueeze_zero(stats["input_mean"]),
                              unsqueeze_zero(stats["target_mean"])], dim=-1)
            std = torch.cat([unsqueeze_zero(stats["input_std"]),
                             unsqueeze_zero(stats["target_std"])], dim=-1)
            self.normalizer_input.set_stats(mean, std)
            self.normalizer_target.set_stats(mean, std)

            self.normalizer_state1.set_stats(unsqueeze_zero(stats["input_mean"]), unsqueeze_zero(stats["input_std"]))
            self.normalizer_state2.set_stats(unsqueeze_zero(stats["target_mean"]), unsqueeze_zero(stats["target_std"]))

        return

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                      lr=self.lr,
                                      weight_decay=self.weight_decay)
        scheduler = OneCycleLR(optimizer, max_lr=self.lr,
                               div_factor=1e4,
                               pct_start=0.3,
                               final_div_factor=1e4,
                               total_steps=self.trainer.estimated_stepping_batches
                               )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            },
        }

    def get_unnorm_target(self, s):
        if self.norm_target:
            s_unnorm = self.normalizer_target(s, inverse=True)
        else:
            s_unnorm = s
            s = self.normalizer_target(s, inverse=False)
        return s, s_unnorm

    def training_step(self, train_batch, batch_idx):
        pass

    def validation_step(self, val_batch, batch_idx):
        pass

    def test_step(self, test_batch, batch_idx):
        x, y, node_type_inp, node_type_prop, input_pos, prop_pos, n_time = test_batch
        y, y_unnorm = self.get_unnorm_target(y)
        down_factor = self.trainer.datamodule.down_factor if self.trainer.datamodule.down_interp else 1
        forward_steps = y.shape[1]

        # state reconstruction first
        h_ch = self.normalizer_state1.subtract.shape[0]
        u_ch = self.normalizer_state2.subtract.shape[0]

        x_obs = x[..., 0:h_ch]  # do not take x and t added to the input
        x_obs_unnorm = self.normalizer_state1(x_obs, inverse=True)
        x_obs_inp = torch.cat([x_obs, x[..., h_ch+u_ch:]], dim=-1)
        x_unobs = x[..., h_ch:h_ch+u_ch]
        x_unobs_unnorm = self.normalizer_state2(x_unobs, inverse=True)

        # for rec input_pos=prop_pos
        x_rec = self.model_state.forward(x_obs_inp, node_type_inp, input_pos, input_pos, forward_steps)
        x_rec_unnorm = self.normalizer_state2(x_rec, inverse=True)
        mae_un_loss_reconstr = self.mae_criterion(x_rec_unnorm, x_unobs_unnorm, down_factor)

        if self.flip_xy:
            x_obs_rec = torch.cat([x_rec, x_obs], dim=-1)
            x_obs_rec_unnorm = torch.cat([x_rec_unnorm, x_obs_unnorm], dim=-1)
            x_gt_unnorm = torch.cat([x_unobs_unnorm, x_obs_unnorm], dim=-1)
        else:
            x_obs_rec = torch.cat([x_obs, x_rec], dim=-1)
            x_obs_rec_unnorm = torch.cat([x_obs_unnorm, x_rec_unnorm], dim=-1)
            x_gt_unnorm = torch.cat([x_obs_unnorm, x_unobs_unnorm], dim=-1)

        x_pred_inp = torch.cat([x_obs_rec, x[..., h_ch+u_ch:]], dim=-1)
        y_pred = self.model_time.forward(x_pred_inp, node_type_inp, node_type_prop, input_pos, prop_pos, forward_steps)  # [b t n c]

        if self.flip_xy:
            h_unnorm = self.normalizer_state2(y_pred[..., :u_ch], inverse=True)
            u_unnorm = self.normalizer_state1(y_pred[..., u_ch:u_ch+h_ch], inverse=True)
            y_pred_unnorm = torch.cat([h_unnorm, u_unnorm], dim=-1)

            y_unnorm = torch.cat([y_unnorm[..., u_ch:u_ch+h_ch], y_unnorm[..., :u_ch]], dim=-1)
        else:
            y_pred_unnorm = self.normalizer_target(y_pred, inverse=True)

        mae_un_loss_time_pred = self.mae_criterion(y_pred_unnorm, y_unnorm, down_factor)

        self.log('test_mae_un_rec', mae_un_loss_reconstr, prog_bar=True, on_epoch=True, on_step=False)
        self.log('test_mae_un_pred', mae_un_loss_time_pred, prog_bar=True, on_epoch=True, on_step=False)

        if forward_steps == 1:
            # [b, 1, n_time x n_x, c] -> [b, n_time, n_x, c]
            n_time = n_time[0].item()
            y_pred_unnorm = y_pred_unnorm.reshape(y_pred_unnorm.shape[0], n_time, -1, y_pred_unnorm.shape[-1])
            y_unnorm = y_unnorm.reshape(y_unnorm.shape[0], n_time, -1, y_unnorm.shape[-1])

            # combination of x obs and x rec
            x_in_unnorm = x_obs_rec_unnorm.reshape(x_obs_rec_unnorm.shape[0], n_time, -1, x_obs_rec_unnorm.shape[-1])
            x_in_gt_unnorm = x_gt_unnorm.reshape(x_gt_unnorm.shape[0], n_time, -1, x_gt_unnorm.shape[-1])

            # concatenate input and target via time dimension to calculate the pde loss
            state_full_pred_unnorm = torch.cat([x_in_unnorm, y_pred_unnorm], dim=1)
            state_full_gt_unnorm = torch.cat([x_in_gt_unnorm, y_unnorm], dim=1)

            # state_full_pred_unnorm = self.normalizer_target(state_full_pred, inverse=True)
            # state_full_gt_unnorm = self.normalizer_target(state_full_gt, inverse=True)

            pde_loss = self.get_pde_loss(state_full_pred_unnorm, clamp_loss=False, reduce=True)
            pde_loss_gt = self.get_pde_loss(state_full_gt_unnorm, clamp_loss=False, reduce=True)

            self.log('test_pde_loss', pde_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            self.log('test_pde_loss_gt', pde_loss_gt, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

            mask = torch.ones_like(state_full_gt_unnorm)
            if self.flip_xy:
                mask[:, :n_time, :, h_ch:h_ch+u_ch] = 0
            else:
                mask[:, :n_time, :, 0:h_ch] = 0

            mae_un_loss = self.mae_full_criterion(state_full_pred_unnorm, state_full_gt_unnorm, mask)
            self.log('test_mae_un', mae_un_loss, prog_bar=True, on_epoch=True, on_step=False)

            y_pred = state_full_pred_unnorm
            y = state_full_gt_unnorm

        return {'pred': y_pred, 'target': y}

    def get_pde_loss(self, x_unnorm, x_gt_unnorm=None, clamp_loss=False, reduce=True):
        if x_gt_unnorm is None:
            x_gt_unnorm = x_unnorm

        if self.flip_xy:
            # swap places for normalizers
            pde_error_dx_matrix = self.pde_loss(x_unnorm, x_gt_unnorm, self.normalizer_state2, self.normalizer_state1,
                                                return_d=False, calc_prob=False, clamp_loss=clamp_loss)
        else:
            pde_error_dx_matrix = self.pde_loss(x_unnorm, x_gt_unnorm, self.normalizer_state1, self.normalizer_state2,
                                                return_d=False, calc_prob=False, clamp_loss=clamp_loss)

        if reduce:
            n_batch = x_unnorm.shape[0]
            pde_loss = torch.sum(pde_error_dx_matrix) / n_batch
        else:
            pde_loss = pde_error_dx_matrix

        return pde_loss
