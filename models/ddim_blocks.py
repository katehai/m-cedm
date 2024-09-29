'''
The file is adopted from https://github.com/ermongroup/ddim/blob/main/models/diffusion.py
'''
import copy
import math

import numpy as np
import torch
from torch import nn as nn


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.type_as(timesteps)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # silu
    return x * torch.sigmoid(x)


class EmaModel(nn.Module):
    def __init__(self, model, beta):
        super().__init__()
        self.beta = beta
        self.ma_model = copy.deepcopy(model)

    def update(self, current_model):
        if isinstance(current_model, nn.parallel.DistributedDataParallel):
            current_model = current_model.module

        for current_params, ma_params in zip(current_model.parameters(), self.ma_model.parameters()):
            if current_params.requires_grad:
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def forward(self, *args, **kwargs):
        return self.ma_model(*args, **kwargs)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Model(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        ch, out_channels = hparams.model.ch, hparams.model.out_ch
        channel_mult = tuple(hparams.model.ch_mult)
        attn_resolutions = hparams.model.attn_resolutions
        dropout = hparams.model.dropout
        cond_channels = hparams.model.cond_channels if hasattr(hparams.model, 'cond_channels') else 0
        resolution = hparams.model.resolution
        resamp_with_conv = hparams.model.resamp_with_conv
        num_timesteps = hparams.diffusion.num_diffusion_timesteps

        # self-conditioning implementation similar to this repo
        # https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/guided_diffusion.py#L803
        # self-conditioning increases the training time but improves the quality of generated images
        self_condition = hparams.model.self_cond if hasattr(hparams.model, 'self_cond') else False
        self.self_condition = self_condition

        ## conditional info is concatenated with the input along channel dimension
        cat_cond = hparams.model.cat_cond if hasattr(hparams.model, 'cat_cond') else False
        self.cat_condition = cat_cond

        ## dx conditioning concatenated to the input or separately encoded
        self.dx_cond = hparams.model.dx_cond if hasattr(hparams.model, 'dx_cond') else False
        self.cat_dx = hparams.model.cat_dx if hasattr(hparams.model, 'cat_dx') else False

        if hparams.model.type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))

        self.ch = ch
        self.channel_mult_emb = 4  # hparams.model.ch_mult_emb
        self.temb_ch = self.ch * self.channel_mult_emb
        self.num_resolutions = len(channel_mult)
        self.num_res_blocks = hparams.model.num_res_blocks
        self.resolution = resolution

        in_channels = hparams.model.in_channels * (2 if self_condition else 1)  # self cond is stacked to the input
        in_channels1 = in_channels + cond_channels if cat_cond else in_channels  # cond is stacked to the input
        self.in_channels = in_channels1 + hparams.model.in_channels if self.dx_cond and self.cat_dx else in_channels1
        self.cond_channels = cond_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(self.in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        combine_ch = 0
        # a separate encoder for conditioning
        if cond_channels > 0 and not cat_cond:
            # encode conditional information with a separate encoder before stacking it to the main input
            self.cond_enc = nn.Sequential(
                torch.nn.Conv2d(cond_channels, self.ch, kernel_size=1, stride=1, padding=0),
                nn.GELU(),
                torch.nn.Conv2d(self.ch, self.ch, kernel_size=3, stride=1, padding=1, padding_mode='circular'))
            combine_ch += self.ch
        else:
            self.cond_enc = None

        # a separate encoder for dx conditioning
        if self.dx_cond and not self.cat_dx:
            dx_ch = hparams.model.in_channels
            self.dx_enc = nn.Sequential(
                torch.nn.Conv2d(dx_ch, self.ch, kernel_size=1, stride=1, padding=0),
                nn.GELU(),
                torch.nn.Conv2d(self.ch, self.ch, kernel_size=3, stride=1, padding=1, padding_mode='circular'))
            combine_ch += self.ch
        else:
            self.dx_enc = None

        # combine features of the input and the conditional and dx information
        if combine_ch > 0:
            self.combine_enc = torch.nn.Conv2d(self.ch + combine_ch, self.ch, kernel_size=1, stride=1, padding=0)
        else:
            self.combine_enc = None

        curr_res = resolution
        in_ch_mult = (1,) + channel_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * channel_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * channel_mult[i_level]
            skip_in = ch * channel_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def cat_conditioning(self, x, cond, x_self_cond, dx):
        bx, cx, hx, wx = x.shape
        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            x = torch.cat((x_self_cond, x), dim=1)

        # concatenate conditioning to the input
        if self.cat_condition and self.cond_channels > 0:
            if cond is None:
                b, _, h, w = x.shape
                cond = torch.zeros((b, self.cond_channels, h, w))
                cond = cond.type_as(x)
            x = torch.cat((cond, x), dim=1)

        # concatenate dx to the input
        if self.dx_cond and self.cat_dx:
            if dx is None:
                dx = torch.zeros((bx, cx, hx, wx))
                dx = dx.type_as(x)
            x = torch.cat((x, dx), dim=1)
        return x

    def combine_cond_feat(self, x_feat, cond, dx):
        bx, cx, hx, wx = x_feat.shape
        if self.cond_enc is not None:
            if cond is not None:
                cond_feat = self.cond_enc(cond)
            else:
                cond_feat = torch.zeros((bx, cx, hx, wx))
                cond_feat = cond_feat.type_as(x_feat)
            x_feat = torch.cat([x_feat, cond_feat], dim=1)

        if self.dx_enc is not None:
            if dx is not None:
                dx_feat = self.dx_enc(dx)
            else:
                dx_feat = torch.zeros((bx, cx, hx, wx))
                dx_feat = dx_feat.type_as(x_feat)
            x_feat = torch.cat([x_feat, dx_feat], dim=1)

        if self.combine_enc is not None:
            x_feat = self.combine_enc(x_feat)
        return x_feat

    def forward(self, x, t, cond=None, x_self_cond=None, dx=None):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # concatenate conditioning information to the input
        x = self.cat_conditioning(x, cond, x_self_cond, dx)
        x_feat = self.conv_in(x)

        # concatenate conditional and dx information to encoded features alternatively
        x_feat = self.combine_cond_feat(x_feat, cond, dx)

        # downsampling
        hs = [x_feat]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)

    betas = torch.from_numpy(betas).float()
    return betas
