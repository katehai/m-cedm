'''
The file is adopted from https://github.com/NVlabs/edm/blob/main/training/networks.py
'''
import numpy as np
import torch
from torch.nn.functional import silu


# Unified routine for initializing weights and biases.
def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


# Fully-connected layer.
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


# Convolutional layer with optional up/downsampling.
class Conv2d(torch.nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel, bias=True, up=False, down=False,
                 resample_filter=[1, 1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
                 ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels * kernel * kernel, fan_out=out_channels * kernel * kernel)
        self.weight = torch.nn.Parameter(
            weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(
            weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]),
                                                     groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad + f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]),
                                                         groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels,
                                               stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


# Group normalization.
class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype),
                                           bias=self.bias.to(x.dtype), eps=self.eps)
        return x


# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.
class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(
            dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2,
                                          input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk


# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.
class UNetBlock(torch.nn.Module):
    def __init__(self,
                 in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
                 num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
                 resample_filter=[1, 1], resample_proj=False, adaptive_scale=True,
                 init=dict(), init_zero=dict(init_weight=0), init_attn=None,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down,
                            resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels * (2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down,
                               resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels * 3, kernel=1,
                              **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3,
                                                      -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x


# Timestep embedding used in the DDPM++ and ADM architectures.
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2)
        freqs = freqs.type_as(x)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs)
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class DhariwalUNet(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        ch, out_channels = hparams.model.ch, hparams.model.out_ch
        channel_mult = tuple(hparams.model.ch_mult)
        cond_channels = hparams.model.cond_channels if hasattr(hparams.model, 'cond_channels') else 0
        attn_resolutions = hparams.model.attn_resolutions
        resolution = hparams.model.resolution
        num_res_blocks = hparams.model.num_res_blocks

        self.resolution = resolution

        # conditional info: input augmentation parameters and class labels
        augment_dim, label_dim = hparams.model.augment_dim, hparams.model.label_dim

        dropout, label_dropout = hparams.model.dropout, hparams.model.label_dropout
        self.label_dropout = label_dropout
        emb_channels = ch  # * hparams.model.ch_mult_emb

        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1 / 3), init_bias=np.sqrt(1 / 3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout, init=init,
                            init_zero=init_zero)

        # Self-conditioning
        self.self_condition = hparams.model.self_cond if hasattr(hparams.model, 'self_cond') else False

        ## conditional info is concatenated with the input along channel dimension
        cat_cond = hparams.model.cat_cond if hasattr(hparams.model, 'cat_cond') else False
        self.cat_condition = cat_cond

        self.dx_cond = hparams.model.dx_cond if hasattr(hparams.model, 'dx_cond') else False
        self.cat_dx = hparams.model.cat_dx if hasattr(hparams.model, 'cat_dx') else False

        in_channels = hparams.model.in_channels * (2 if self.self_condition else 1)  # self cond is stacked to the input
        in_channels1 = in_channels + cond_channels if cat_cond else in_channels  # cond is stacked to the input
        self.in_channels = in_channels1 + hparams.model.in_channels if self.dx_cond and self.cat_dx else in_channels1
        self.cond_channels = cond_channels

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=ch)
        self.map_augment = Linear(in_features=augment_dim, out_features=ch, bias=False,
                                  **init_zero) if augment_dim else None
        self.map_layer0 = Linear(in_features=ch, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.map_label = Linear(in_features=label_dim, out_features=emb_channels, bias=False,
                                init_mode='kaiming_normal',
                                init_weight=np.sqrt(label_dim)) if label_dim else None

        # Conditioning encoders, if not concatenated to the input
        combine_ch = 0
        self.ch = ch * channel_mult[0]  # features produces by  conv_in
        # a separate encoder for conditioning
        if cond_channels > 0 and not cat_cond:
            # encode conditional information with a separate encoder before stacking it to the main input
            self.cond_enc = torch.nn.Sequential(
                Conv2d(cond_channels, self.ch, kernel=3, **init),
                torch.nn.GELU(),
                Conv2d(self.ch, self.ch, kernel=3, **init))
            combine_ch += self.ch
        else:
            self.cond_enc = None

        # a separate encoder for dx conditioning
        if self.dx_cond and not self.cat_dx:
            dx_ch = hparams.model.in_channels
            self.dx_enc = torch.nn.Sequential(
                Conv2d(dx_ch, self.ch, kernel=3, **init),
                torch.nn.GELU(),
                Conv2d(self.ch, self.ch, kernel=3, **init))
            combine_ch += self.ch
        else:
            self.dx_enc = None

        # combine features of the input and the conditional and dx information
        if combine_ch > 0:
            self.combine_enc = Conv2d(self.ch + combine_ch, self.ch, kernel=3, **init)
        else:
            self.combine_enc = None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = self.in_channels
        for level, mult in enumerate(channel_mult):
            res = resolution >> level
            if level == 0:
                cin = cout
                cout = ch * mult
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True,
                                                          **block_kwargs)
            for idx in range(num_res_blocks):
                cin = cout
                cout = ch * mult
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout,
                                                                attention=(res in attn_resolutions), **block_kwargs)
        skips = [block.out_channels for block in self.enc.values()]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True,
                                                         **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_res_blocks + 1):
                cin = cout + skips.pop()
                cout = ch * mult
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout,
                                                                attention=(res in attn_resolutions), **block_kwargs)
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

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

    def forward(self, x, noise_labels, cond=None, x_self_cond=None, dx=None, class_labels=None, augment_labels=None):
        # The model doesn't support conditioning and self conditioning for now
        # Mapping.
        emb = self.map_noise(noise_labels)
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                rand_cond_var = (torch.rand([x.shape[0], 1]) >= self.label_dropout)
                rand_cond_var = rand_cond_var.type_as(tmp)
                tmp = tmp * rand_cond_var
            emb = emb + self.map_label(tmp)
        emb = silu(emb)

        # concatenate conditioning information to the input
        x = self.cat_conditioning(x, cond, x_self_cond, dx)
        res = self.resolution # >> 0
        conv_in_layer = self.enc[f'{res}x{res}_conv']
        x_feat = conv_in_layer(x)

        # concatenate conditional and dx information to encoded features alternatively
        x_feat = self.combine_cond_feat(x_feat, cond, dx)

        # Encoder.
        x = x_feat
        skips = [x]
        for block in self.enc.values():
            if isinstance(block, UNetBlock):
                x = block(x, emb)
                skips.append(x)

        # Decoder.
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        x = self.out_conv(silu(self.out_norm(x)))
        return x
