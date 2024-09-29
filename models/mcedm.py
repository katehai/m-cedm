import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange

from models.adm_blocks import DhariwalUNet
from models.ddim_blocks import EmaModel, Model
from models.losses import NoiseEstimationLoss, CorrelationLoss, MaskedLoss
from models.normalizer import Normalizer
from models.loss_helper import get_pde_loss_function
from utils import DotDict


class PlMcedm(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.cond_p = 1.0

        self.dx_norm = hparams.model.dx_norm if hasattr(hparams.model, 'dx_norm') else 'l2'
        self.dx_detach = hparams.model.dx_detach if hasattr(hparams.model, 'dx_detach') else False
        self.dx_cond = hparams.model.dx_cond if hasattr(hparams.model, 'dx_cond') else False
        self.add_cond_mask = hparams.model.add_cond_mask if hasattr(hparams.model, 'add_cond_mask') else False
        self.add_xt = hparams.model.add_xt if hasattr(hparams.model, 'add_xt') else False

        if self.add_cond_mask:
            # the conditioning mask has the same dimensionality as the input
            hparams.model.cond_channels = hparams.model.cond_channels + hparams.model.in_channels

        if self.add_xt:
            # the conditioning mask has the same dimensionality as the input
            hparams.model.cond_channels = hparams.model.cond_channels + 2

        if hparams.name.startswith('adm'):
            model = DhariwalUNet(hparams)
        else:
            model = Model(hparams)

        self.model = model
        self.ema_model = EmaModel(self.model, beta=hparams.model.ema_rate) if hparams.model.ema else None

        # preconditioning parameters
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data = 1.  # 0.5 in the original paper

        self.sigma_min = 0.002
        self.sigma_max = 80

        # data transformations
        self.normalization = hparams.data.normalization
        self.uniform_dequantization = hparams.data.uniform_dequantization
        self.gaussian_dequantization = hparams.data.gaussian_dequantization
        self.rescaled = hparams.data.rescaled

        self.normalizer_input = Normalizer(stats_shape=self.get_inp_stats_shape(hparams))
        self.normalizer_target = Normalizer(stats_shape=self.get_tar_stats_shape(hparams))

        # optimization parameters
        self.optimizer = hparams.optimization.optimizer
        self.lr = hparams.optimization.lr
        self.weight_decay = hparams.optimization.weight_decay
        self.beta1 = hparams.optimization.beta1
        self.amsgrad = hparams.optimization.amsgrad
        self.eps = hparams.optimization.eps

        self.factor = hparams.optimization.factor
        self.step_size = hparams.optimization.step_size
        self.loss = hparams.optimization.loss
        self.pde_loss_lambda = hparams.optimization.pde_loss_lambda if hasattr(hparams.optimization,
                                                                               'pde_loss_lambda') else 0.
        self.pde_loss_prop_t = hparams.optimization.pde_loss_prop_t if hasattr(hparams.optimization,
                                                                               'pde_loss_prop_t') else False
        self.use_gt_pde = hparams.optimization.use_gt_pde if hasattr(hparams.optimization, 'use_gt_pde') else False

        self.criteria = NoiseEstimationLoss()
        self.mae_criterion = MaskedLoss()
        self.correlation = CorrelationLoss()

        pde_loss_f, pde_loss_sim_f = get_pde_loss_function(system='swe', flip_xy=False)  # the loss is overriden by set_pde_loss
        self.pde_loss = pde_loss_f
        self.pde_loss_simulator = pde_loss_sim_f

        # set sampler params
        self.sparams = self.get_sampler_params(hparams)
        self.test_sparams = self.sparams  # can be overriden whe running test

    def get_inp_stats_shape(self, hparams):
        ch = hparams.model.out_ch // 2  # the model has both inputs and targets
        size = (ch,) if ch > 1 else ()
        return size

    def get_tar_stats_shape(self, hparams):
        ch = hparams.model.out_ch // 2  # the model has both inputs and targets
        size = (ch,) if ch > 1 else ()
        return size

    def set_pde_loss_function(self, system, flip_xy):
        pde_loss_f, pde_loss_sim_f = get_pde_loss_function(system, flip_xy)

        self.pde_loss = pde_loss_f
        self.pde_loss_simulator = pde_loss_sim_f

    @staticmethod
    def get_sampler_params(params):
        if params.get('sampler', None) is None:
            sparams = {}
            sparams['type'] = 'ddim'
            sparams['timesteps'] = 50  # 100  # 125  # 250  # self.num_timesteps (1000)
            sparams['skip_type'] = 'uniform'
            sparams['eta'] = 0.0
            sparams['n_samples'] = 1
            sparams['n_repeat'] = 5
            sparams['n_time_h'] = 128
            sparams['n_time_u'] = 0

            sparams = DotDict(sparams)
        else:
            sparams = params.sampler

        return sparams

    def set_test_sampler_params(self, params):
        self.test_sparams = params

    def setup(self, stage: str = None) -> None:
        if stage == "fit":
            stats = self.trainer.datamodule.get_norm_stats()
            if self.normalization == "min_max":
                self.normalizer_input.set_stats(stats["input_min"], stats["input_min_max"])
                self.normalizer_target.set_stats(stats["target_min"], stats["target_min_max"])
            else:
                self.normalizer_input.set_stats(stats["input_mean"], stats["input_std"])
                self.normalizer_target.set_stats(stats["target_mean"], stats["target_std"])
        return

    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                         betas=(self.beta1, 0.999), amsgrad=self.amsgrad, eps=self.eps)
        elif self.optimizer == 'RMSProp':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        else:
            raise NotImplementedError(
                'Optimizer {} not understood.'.format(self.optimizer))

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.factor)
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler
        #     },
        # }

        return {
            "optimizer": optimizer
        }

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        # update ema weights after optimization step
        if self.ema_model is not None:
            self.ema_model.update(self.model)

    def data_transform(self, h, u):
        # normalize data
        h = self.normalizer_input(h)
        u = self.normalizer_target(u)

        x = torch.cat([h, u], dim=-1)

        if self.uniform_dequantization:
            x = x / 256.0 * 255.0 + torch.rand_like(x) / 256.0
        if self.gaussian_dequantization:
            x = x + torch.randn_like(x) * 0.01
        if self.rescaled:
            x = 2 * x - 1.0  # make it from -1 to 1

        return x

    def inverse_data_transform(self, h, u):
        if self.rescaled:
            h = (h + 1.0) / 2.0
            u = (u + 1.0) / 2.0

        if self.normalization == "min_max":
            h = torch.clamp(h, 0.0, 1.0)
            u = torch.clamp(u, 0.0, 1.0)

        h = self.normalizer_input(h, inverse=True)
        u = self.normalizer_target(u, inverse=True)
        return h, u

    def model_precond(self, x_noise: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor = None,
                      x_self_cond: torch.Tensor = None, dx: torch.Tensor = None) -> torch.Tensor:
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x_noise), c_noise.flatten(), cond, x_self_cond=x_self_cond, dx=dx)

        D_x = c_skip * x_noise + c_out * F_x
        return D_x

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, noise: torch.Tensor, cond: torch.Tensor = None,
                mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            x_noise = x + mask * noise * sigma
        else:
            x_noise = x + noise * sigma

        # switch off dx conditioning with a small probability
        if self.dx_cond and torch.rand(1) > 0.1:
            h_in = cond[..., 0:self.h_ch, :, :]
            dx = self.get_dx_input(h_in, x_noise)
        else:
            dx = None

        # detach the dx conditioning so the gradients do not flow through its calculation
        if dx is not None and self.dx_detach:
            dx = dx.detach()

        if torch.rand(1) >= self.cond_p:
            cond = None  # switch off the conditioning on the current batch

        output = self.model_precond(x_noise, sigma.float(), cond, x_self_cond=None, dx=dx)
        return output

    def get_loss_weight(self, sigma):
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        return weight

    def get_cond_in(self, x, mask, dx, dt):
        if self.add_cond_mask:
            # similar to SSSD_S4
            x_cond = x * (1 - mask)
            cond_in = torch.cat([x_cond, (1. - mask)], dim=-1)
        else:
            cond_in = x * (1 - mask) + torch.randn_like(x) * mask

        if self.add_xt:
            cond_in = torch.cat([cond_in, dx, dt], dim=-1)

        return cond_in

    def training_step(self, train_batch, batch_idx):
        h_unnorm, dx, dt, u_unnorm, mask = train_batch
        self.h_ch = h_unnorm.shape[-1]
        self.u_ch = u_unnorm.shape[-1]

        x = self.data_transform(h_unnorm, u_unnorm)  # b h w c

        # both h and u are used as conditioning in points where measurements are available
        cond_in = self.get_cond_in(x, mask, dx, dt)
        cond_in = rearrange(cond_in, 'b h w c -> b c h w')

        x = rearrange(x, 'b h w c -> b c h w')
        noise = torch.randn_like(x)

        # calculate sigma and loss weight
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1])
        rnd_normal = rnd_normal.type_as(x)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = self.get_loss_weight(sigma)

        mask_c = rearrange(mask, 'b h w c -> b c h w')
        D_x = self.forward(x, sigma, noise, cond=cond_in, mask=mask_c)

        # calculate loss only for missing values
        loss = self.criteria(D_x * mask_c, x * mask_c, weight)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        if (self.current_epoch + 1) % 100 != 0 and self.current_epoch != 0:  # plot validation images every 100 epochs
            return {"epoch": self.current_epoch}

        h_unnorm, dx, dt, u_unnorm, masks = val_batch

        self.h_ch = h_ch = h_unnorm.shape[-1]
        self.u_ch = u_ch = u_unnorm.shape[-1]

        state_gt = self.data_transform(h_unnorm, u_unnorm)
        state_gt_c = rearrange(state_gt, 'b h w c -> b c h w')
        noise = torch.randn_like(state_gt_c)

        result_dict = {"epoch": self.current_epoch}
        for name, mask in masks.items():
            # both h and u are used as conditioning in points where measurements are available
            cond_in = self.get_cond_in(state_gt, mask, dx, dt)
            cond_in = rearrange(cond_in, 'b h w c -> b c h w')

            mask_c = rearrange(mask, 'b h w c -> b c h w')

            guide_dx = self.sparams.guide_dx
            if self.sparams.type == 'edm':
                xs = self.sample_edm(noise, cond_in, mask_c, self.sparams, return_last=True, guide_dx=guide_dx)
            else:
                raise "Non EDM sampler is not supported for the model"

            hu_last = xs[:, -1]  # b t h w c
            loss_hu = self.mae_criterion(hu_last, state_gt, mask)

            h_last = xs[:, -1, :, :, 0:h_ch]  # b t h w c
            u_last = xs[:, -1, :, :, h_ch:u_ch + h_ch]  # b t h w c

            # unnormalized error is calculated
            h_last_unnorm, u_last_unnorm = self.inverse_data_transform(h_last, u_last)
            hu_last_unnorm = torch.cat([h_last_unnorm, u_last_unnorm], dim=-1)
            state_gt_unnorm = torch.cat([h_unnorm, u_unnorm], dim=-1)
            loss_hu_un = self.mae_criterion(hu_last_unnorm, state_gt_unnorm, mask)

            self.log(f'val_mae_{name}', loss_hu, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'val_mae_{name}_un', loss_hu_un, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

            n_batch = len(h_unnorm)
            x0_t = xs[:, -1]
            pde_loss_sum = self.get_pde_loss(x0_t, clamp_loss=False, do_rearrange=False)
            pde_loss = pde_loss_sum / n_batch

            self.log(f'val_pde_loss_{name}', pde_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

            xs_traj = xs[:, -1].unsqueeze(dim=1)
            gt_plot = state_gt

            result_dict[f'loss_{name}'] = loss_hu
            result_dict[f'loss_{name}_un'] = loss_hu_un

            result_dict[f'traj_{name}'] = xs_traj
            result_dict[f'gt_{name}'] = gt_plot

        return result_dict

    def test_step(self, test_batch, test_idx):
        h_unnorm, dx, dt, u_unnorm, masks = test_batch
        self.h_ch = h_ch = h_unnorm.shape[-1]
        self.u_ch = u_ch = u_unnorm.shape[-1]
        down_factor = self.trainer.datamodule.down_factor if self.trainer.datamodule.down_interp else 1

        state_gt = self.data_transform(h_unnorm, u_unnorm)
        state_gt_c = rearrange(state_gt, 'b h w c -> b c h w')

        n_samples = self.test_sparams.n_samples  # for each input condition draw 5 samples
        return_last = self.test_sparams.return_last
        guide_dx = self.test_sparams.guide_dx

        state_gt_rep = state_gt_c.repeat(n_samples, 1, 1, 1)  # n*b h w c

        result_dict = {}
        use_loss_dim = True
        for name, mask in masks.items():
            if use_loss_dim:
                start = 0 if name.startswith('h') else h_ch
                end = h_ch if name.startswith('h') else h_ch + u_ch
                loss_dim = torch.arange(start, end, 1).long()
            else:
                loss_dim = None

            # both h and u are used as conditioning in points where measurements are available
            cond_in = self.get_cond_in(state_gt, mask, dx, dt)
            cond_in = rearrange(cond_in, 'b h w c -> b c h w')
            cond_in_rep = cond_in.repeat(n_samples, 1, 1, 1)

            noise = torch.randn_like(state_gt_rep)

            mask_c = rearrange(mask, 'b h w c -> b c h w')
            mask_c_rep = mask_c.repeat(n_samples, 1, 1, 1)

            if self.test_sparams.type == 'edm':
                xs = self.sample_edm(noise, cond_in_rep, mask_c_rep, self.test_sparams, return_last=return_last,
                                     guide_dx=guide_dx)
            else:
                raise "Non EDM sampler is not supported for the model"

            # average across predictions for each input condition
            xs_mean = rearrange(xs, '(n b) t h w c -> n b t h w c', n=n_samples)
            xs_mean = torch.mean(xs_mean, dim=0)

            hu_last = xs_mean[:, -1]  # b t h w c
            if down_factor > 1:
                each_x = 2 ** (down_factor - 1)

                # the loss is calculated in locations where the mask = 1
                mask_down = torch.zeros_like(mask)
                mask_down[:, ::each_x, ::each_x] = 1.
                mask_loss = mask * mask_down
            else:
                mask_loss = mask
            loss_hu = self.mae_criterion(hu_last, state_gt, mask_loss, loss_dim)

            h_last = xs_mean[:, -1, :, :, 0:h_ch]  # b t h w c
            u_last = xs_mean[:, -1, :, :, h_ch:u_ch + h_ch]  # b t h w c

            # unnormalized error is calculated
            h_last_unnorm, u_last_unnorm = self.inverse_data_transform(h_last, u_last)
            hu_last_unnorm = torch.cat([h_last_unnorm, u_last_unnorm], dim=-1)
            state_gt_unnorm = torch.cat([h_unnorm, u_unnorm], dim=-1)

            loss_hu_un = self.mae_criterion(hu_last_unnorm, state_gt_unnorm, mask_loss, loss_dim)

            print()
            print(f"Loss {name} {loss_hu}, loss {name} un {loss_hu_un}")

            self.log(f'test_mae_{name}', loss_hu, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'test_mae_{name}_un', loss_hu_un, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

            # calculate PDE loss for each prediction and then average
            n_batch = len(h_unnorm)
            pred = xs[:, -1]
            pde_loss_sum = self.get_pde_loss(pred, clamp_loss=False, do_rearrange=False)
            pde_loss = pde_loss_sum / n_samples / n_batch

            self.log(f'test_pde_loss_{name}', pde_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            print(f"Pde loss {name} is {pde_loss}")
            pde_loss_gt_sum = self.get_pde_loss(state_gt, clamp_loss=False, do_rearrange=False)
            pde_loss_gt = pde_loss_gt_sum / n_batch

            self.log('test_pde_loss_gt', pde_loss_gt, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            print(f"Pde loss gt is {pde_loss_gt}")

            xs = rearrange(xs[:, -1], '(n b) h w c -> b h w n c', n=n_samples).unsqueeze(dim=1)
            xs_traj = xs
            gt_plot = state_gt

            result_dict[f'loss_{name}'] = loss_hu
            result_dict[f'loss_{name}_un'] = loss_hu_un

            if n_samples < 15:
                result_dict[f'traj_{name}'] = xs_traj
                result_dict[f'gt_{name}'] = gt_plot

        return result_dict

    def get_denoised(self, model, xt, t, cond=None, x_self_cond=None, dx=None, w=None):
        xt = xt.to(torch.float32)
        t = t.to(torch.float32).reshape(-1, 1, 1, 1)
        sigma = t

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        if w is None or np.abs(w) < 0.001 or (cond is None and dx is None):
            F_x = model(c_in * xt, c_noise.flatten(), cond=cond, x_self_cond=x_self_cond, dx=dx)
        else:
            # use blending with coefficient w (classifier-free guidance)
            F_x = (w + 1) * model((c_in * xt), c_noise.flatten(), cond, x_self_cond=x_self_cond, dx=dx) \
                  - w * model(c_in * xt, c_noise.flatten(), x_self_cond=x_self_cond)

        D_x = c_skip * xt + c_out * F_x
        return D_x, F_x

    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        result = 0 if return_index else sigma
        return result

    def get_pde_loss(self, x_denoised, x_gt_unnorm=None, noise_level=None, clamp_loss=True, do_rearrange=True,
                     reduce=True):
        h_ch = self.h_ch
        u_ch = self.u_ch

        if do_rearrange:
            x_denoised = rearrange(x_denoised, 'b c h w -> b h w c')

        h_denoised, u_denoised = x_denoised[..., 0:h_ch], x_denoised[..., h_ch:u_ch+h_ch]
        h_denoised = h_denoised.to(torch.float32)
        u_denoised = u_denoised.to(torch.float32)

        h_unnorm, u_unnorm = self.inverse_data_transform(h_denoised, u_denoised)
        x_unnorm = torch.cat([h_unnorm, u_unnorm], dim=-1)

        if x_gt_unnorm is None:
            x_gt_unnorm = x_unnorm

        pde_error_dx_matrix = self.pde_loss(x_unnorm, x_gt_unnorm, self.normalizer_input, self.normalizer_target,
                                            return_d=False, calc_prob=False, clamp_loss=clamp_loss)

        if noise_level is not None:
            # pde error should be proportional to the current noise level during training
            noise_level = noise_level.reshape(-1, 1, 1, 1)
            pde_error_dx_matrix = pde_error_dx_matrix / (noise_level + 1.)

        if reduce:
            pde_loss = torch.sum(pde_error_dx_matrix)
        else:
            pde_loss = pde_error_dx_matrix
        return pde_loss

    def get_dx_pde(self, cond, x_denoised, calc_prob=False):
        h_ch = self.h_ch
        u_ch = self.u_ch
        h_denoised, u_denoised = x_denoised[..., 0:h_ch], x_denoised[..., h_ch:u_ch+h_ch]
        h_denoised = h_denoised.to(torch.float32)
        u_denoised = u_denoised.to(torch.float32)
        h_denoised = rearrange(h_denoised, 'b c h w -> b h w c')
        u_denoised = rearrange(u_denoised, 'b c h w -> b h w c')

        h_unnorm, u_unnorm = self.inverse_data_transform(h_denoised, u_denoised)
        x_unnorm = torch.cat([h_unnorm, u_unnorm], dim=-1)

        return_d = True
        pde_error_dx_matrix = self.pde_loss(x_unnorm, x_unnorm, self.normalizer_input, self.normalizer_target,
                                            return_d, calc_prob)

        pde_error_dx_matrix = rearrange(pde_error_dx_matrix, 'b h w c -> b c h w')
        return pde_error_dx_matrix

    def get_dx_input(self, cond, x_denoised):
        dx = None
        if self.dx_cond:
            # calculate the gradient of the PDE loss
            calc_prob = self.dx_norm == 'prob'
            dx1 = self.get_dx_pde(cond, x_denoised, calc_prob=calc_prob)
            has_nan = torch.any(torch.isnan(dx1)).item()
            if not has_nan:
                # Normalizing to unit norm -> the same result seems to be the same to clamp
                b, c, dim1, dim2 = dx1.shape

                if self.dx_norm == 'prob':
                    dx = dx1
                elif self.dx_norm == 'gauss':
                    dx1 = torch.clamp(dx1, -0.01, 0.01)  # the error is very small if the fir is good

                    # has quite big values, especially in the beginning of the training
                    dx = rearrange(dx1, 'b c h w -> b c (h w)')
                    dx_mean = torch.mean(dx, dim=2, keepdim=True)
                    dx_std = torch.std(dx, dim=2, keepdim=True)
                    dx = (dx - dx_mean) / (dx_std + 1e-6)
                    dx = rearrange(dx, 'b c (h w) -> b c h w', h=dim1, w=dim2)
                elif self.dx_norm == 'min_max':
                    dx1 = torch.clamp(dx1, -0.01, 0.01)

                    dx = rearrange(dx1, 'b c h w -> b c (h w)')
                    dx_min = torch.min(dx, dim=2, keepdim=True)[0]
                    dx_max = torch.max(dx, dim=2, keepdim=True)[0]
                    dx = 2 * (dx - dx_min) / (dx_max - dx_min + 1e-6) - 1.
                    dx = rearrange(dx, 'b c (h w) -> b c h w', h=dim1, w=dim2)
                elif self.dx_norm == 'clamp':
                    dx = torch.clamp(dx1, -5, 5)  # to have about the same scale as the input
                else:
                    dx = rearrange(dx1, 'b c h w -> b c (h w)')
                    dx = F.normalize(dx, p=2, dim=2)
                    dx = rearrange(dx, 'b c (h w) -> b c h w', h=dim1, w=dim2)
            else:
                print("Nan in dx")
        return dx

    def get_dx_log_prob(self, cond, x_denoised, guide_dx):
        dx = torch.zeros_like(x_denoised)
        if guide_dx:
            dx1 = self.get_dx_pde(cond, x_denoised, calc_prob=True)
            has_nan = torch.any(torch.isnan(dx1)).item()
            if not has_nan:
                dx = dx1
            else:
                print("Nan in dx")
        return dx

    def sample_edm(self, hu, cond, hu_mask, sparams, return_last=True, guide_dx=False):
        # the method is adopted from original repo:
        # https://github.com/NVlabs/edm/blob/main/generate.py
        w = sparams.w

        model = self.ema_model if self.ema_model is not None else self.model
        hu_noise = torch.randn_like(hu)

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sparams.sigma_min, self.sigma_min)
        sigma_max = min(sparams.sigma_max, self.sigma_max)
        num_steps = sparams.timesteps

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float64)
        step_indices = step_indices.type_as(hu.to(torch.float64))
        t_steps = (sigma_max ** (1 / sparams.rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / sparams.rho) - sigma_max ** (1 / sparams.rho))) ** sparams.rho
        t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

        hu_known = cond[:, 0:self.h_ch + self.u_ch]
        x = hu_noise

        # Main sampling loop.
        x_next = x.to(torch.float64) * t_steps[0]

        ## add known part
        x_next = hu_known * (1 - hu_mask) + x_next * hu_mask

        xs = [x_next]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            S_min = sparams.S_min
            S_max = float(sparams.S_max)
            gamma = min(sparams.S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = self.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * sparams.S_noise * torch.randn_like(x_cur) * hu_mask

            # Euler step.
            dx_in = self.get_dx_input(None, x_hat)
            denoised, et = self.get_denoised(model, x_hat, t_hat, cond=cond, dx=dx_in, w=w)
            denoised = denoised.to(torch.float64)

            dx = self.get_dx_log_prob(None, denoised, guide_dx)
            weight = 5.
            d_cur = (x_hat - denoised) / t_hat - weight * dx
            x_next = x_hat + (t_next - t_hat) * d_cur * hu_mask

            # Apply 2nd order correction.
            if i < num_steps - 1:
                dx_in = self.get_dx_input(None, x_next)
                denoised, et = self.get_denoised(model, x_next, t_next, cond=cond, dx=dx_in, w=w)
                denoised = denoised.to(torch.float64)

                dx = self.get_dx_log_prob(None, denoised, guide_dx)
                d_prime = (x_next - denoised) / t_next - weight * dx
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime) * hu_mask

            if return_last:
                # to save GPU memory
                xs = [x_next]
            else:
                xs.append(x_next)

        xs = torch.stack(xs, dim=0)
        xs = rearrange(xs, 't b c h w -> b t h w c')
        return xs

