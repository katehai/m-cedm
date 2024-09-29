import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange

from models.adm_blocks import DhariwalUNet
from models.ddim_blocks import EmaModel, Model, get_beta_schedule
from models.losses import NoiseEstimationLoss, CorrelationLoss, MaskedLoss
from models.normalizer import Normalizer
from models.loss_helper import get_pde_loss_function
from utils import DotDict


class PlDdim(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.model_var_type = hparams.model.var_type

        betas, posterior_variance = self.get_diffusion_schedule(hparams)
        self.register_buffer("betas", betas)
        self.num_timesteps = self.betas.shape[0]

        if self.model_var_type == "fixedlarge":
            self.register_buffer("logvar", betas.log())
        elif self.model_var_type == "fixedsmall":
            self.register_buffer("logvar", posterior_variance.clamp(min=1e-20).log())

        self.cond_p = 0.0  # in case some conditioning is passed to the model

        self.dx_norm = hparams.model.dx_norm if hasattr(hparams.model, 'dx_norm') else 'l2'
        self.dx_detach = hparams.model.dx_detach if hasattr(hparams.model, 'dx_detach') else False
        self.dx_cond = hparams.model.dx_cond if hasattr(hparams.model, 'dx_cond') else False
        self.node_type = hparams.model.node_type if hasattr(hparams.model, 'node_type') else False
        if self.node_type:
            hparams.model.cond_channels = hparams.model.cond_channels + 1

        if hparams.name.startswith('adm'):
            model = DhariwalUNet(hparams)
        else:
            model = Model(hparams)

        self.model = model
        self.ema_model = EmaModel(self.model, beta=hparams.model.ema_rate) if hparams.model.ema else None

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
        self.mae_criterion = nn.L1Loss()
        self.mae_criterion_mask = MaskedLoss()
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

        if params.type == 'edm':
            # save sampling steps and sigmas for EDM sampler
            self.edm_steps = self.get_edm_steps()
            self.sigma_min = float(self.edm_steps[self.num_timesteps - 1])
            self.sigma_max = float(self.edm_steps[0])

    def get_edm_steps(self):
        alphas_bar = (1.0 - self.betas).cumprod(dim=0)
        edm_steps = ((1 - alphas_bar) / alphas_bar).sqrt()

        # the convention in edm sampler is different, need the reserve the sequence
        edm_steps = edm_steps.flip(dims=(0,))
        return edm_steps

    def setup(self, stage: str = None) -> None:
        def remove_dim(t):
            if len(t.shape) == 1 and t.shape[0] == 1:
                return t.squeeze(0)
            else:
                return t

        if stage == "fit":
            stats = self.trainer.datamodule.get_norm_stats()
            if self.normalization == "min_max":
                self.normalizer_input.set_stats(remove_dim(stats["input_min"]), remove_dim(stats["input_min_max"]))
                self.normalizer_target.set_stats(remove_dim(stats["target_min"]), remove_dim(stats["target_min_max"]))
            else:
                self.normalizer_input.set_stats(remove_dim(stats["input_mean"]), remove_dim(stats["input_std"]))
                self.normalizer_target.set_stats(remove_dim(stats["target_mean"]), remove_dim(stats["target_std"]))

        return

    @staticmethod
    def get_diffusion_schedule(hparams):
        betas = get_beta_schedule(
            beta_schedule=hparams.diffusion.beta_schedule,
            beta_start=hparams.diffusion.beta_start,
            beta_end=hparams.diffusion.beta_end,
            num_diffusion_timesteps=hparams.diffusion.num_diffusion_timesteps,
        )
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        return betas, posterior_variance

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

    def forward(self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        a = (1 - self.betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x_noise = x * a.sqrt() + noise * (1.0 - a).sqrt()

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

        x_self_cond = None
        if self.model.self_condition and torch.rand(1) < 0.5:
            with torch.no_grad():
                self_cond_noise = self.model(x_noise, t.float(), cond, dx=dx)  # probably no cond info here ?
                x_self_cond = (x_noise - self_cond_noise * (1 - a).sqrt()) / a.sqrt()
                x_self_cond = x_self_cond.detach()

        output = self.model(x_noise, t.float(), cond, x_self_cond=x_self_cond, dx=dx)

        et = output
        xt = x_noise
        x0_t = (xt - et * (1 - a).sqrt()) / a.sqrt()

        return output, x0_t

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

    def training_step(self, train_batch, batch_idx):
        h_unnorm, dx, dt, u_unnorm = train_batch
        self.h_ch = h_ch = h_unnorm.shape[-1]
        self.u_ch = u_ch = u_unnorm.shape[-1]

        x = self.data_transform(h_unnorm, u_unnorm)
        x = rearrange(x, 'b h w c -> b c h w')
        n = x.size(0)

        noise = torch.randn_like(x)

        # antithetic sampling
        t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,))
        t = t.type_as(x).long()
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
        output, x0_t = self.forward(x, t, noise)

        loss = self.criteria(output, noise)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        if self.pde_loss_lambda > 0.:
            cond = None
            noise_level = t if self.pde_loss_prop_t else None
            x_gt_unnorm = torch.cat([h_unnorm, u_unnorm], dim=-1) if self.use_gt_pde else None
            pde_loss = self.get_pde_loss(cond, x0_t, x_gt_unnorm=x_gt_unnorm, noise_level=noise_level, clamp_loss=True)
            self.log('train_pde_loss', pde_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            loss = loss + self.pde_loss_lambda * pde_loss

        return loss

    def validation_step(self, val_batch, batch_idx):
        if (self.current_epoch + 1) % 100 != 0 and self.current_epoch != 0:  # plot validation images every 100 epochs
            return {"epoch": self.current_epoch}

        h_unnorm, dx, dt, u_unnorm = val_batch
        self.h_ch = h_ch = h_unnorm.shape[-1]
        self.u_ch = u_ch = u_unnorm.shape[-1]

        state_gt = self.data_transform(h_unnorm, u_unnorm)
        h = state_gt[..., 0:h_ch]  # b h w c
        u = state_gt[..., h_ch:u_ch+h_ch]  # b h w c

        u_noise = torch.randn_like(u)
        guide_dx = self.sparams.guide_dx
        if self.sparams.type == 'edm':
            xs = self.sample_edm(h, u_noise, self.sparams, return_last=True, guide_dx=guide_dx)
        else:
            xs, _ = self.sample_with_repeat(h, u_noise, self.sparams, return_last=True, guide_dx=guide_dx)

        h_last = xs[:, -1, :, :, 0:h_ch]  # b t h w c
        u_last = xs[:, -1, :, :, h_ch:u_ch+h_ch]  # b t h w c

        loss_h = self.mae_criterion(h_last, h)
        loss_u = self.mae_criterion(u_last, u)

        # unnormalized error is calculated
        h_last_unnorm, u_last_unnorm = self.inverse_data_transform(h_last, u_last)

        loss_h_un = self.mae_criterion(h_last_unnorm, h_unnorm)
        loss_u_un = self.mae_criterion(u_last_unnorm, u_unnorm)

        # normalize the predicted values and gt to be between zero and 1
        gt_scaled = self.scale_each_min_max(state_gt)
        xs_scaled = self.scale_each_min_max(xs[:, -1])

        # error between scaled values
        loss_h_scaled = self.mae_criterion(xs_scaled[:, :, :, 0:h_ch], gt_scaled[:, :, :, 0:h_ch])
        loss_u_scaled = self.mae_criterion(xs_scaled[:, :, :, h_ch:u_ch+h_ch], gt_scaled[:, :, :, h_ch:u_ch+h_ch])

        self.log('val_mae_h', loss_h, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log('val_mae_u', loss_u, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log('val_mae_h_un', loss_h_un, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log('val_mae_u_un', loss_u_un, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log('val_mae_h_scaled', loss_h_scaled, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log('val_mae_u_scaled', loss_u_scaled, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        # correlation between prediction and gt
        corr_hu = self.correlation(xs[:, -1], state_gt)
        corr_h = torch.mean(corr_hu[0:h_ch])
        corr_u = torch.mean(corr_hu[h_ch:u_ch+h_ch])
        self.log('val_corr_h', corr_h, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log('val_corr_u', corr_u, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        n_batch = len(h_unnorm)
        cond = None
        x0_t = xs[:, -1]
        pde_loss_sum = self.get_pde_loss(cond, x0_t, clamp_loss=False, do_rearrange=False)
        pde_loss = pde_loss_sum / n_batch

        ## calculate mean PDE loss for each pixel
        # n_t, n_x = x0_t.shape[1], x0_t.shape[2]
        # pde_loss = pde_loss / (n_t * n_x)

        self.log('val_pde_loss', pde_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        if self.sparams.plot_scaled:
            xs_traj = xs_scaled.unsqueeze(dim=1)
            gt_plot = gt_scaled
        else:
            xs_traj = xs[:, -1].unsqueeze(dim=1)
            gt_plot = state_gt

        return {"epoch": self.current_epoch,  # 'val_mae_un_loss': val_mae_un_loss,
                'loss_h': loss_h, 'loss': loss_u,
                'loss_h_un': loss_h_un, 'loss_u_un': loss_u_un,
                'val_loss_h_scaled': loss_h_scaled, 'val_loss_u_scaled': loss_u_scaled,
                'traj': xs_traj, 'gt': gt_plot}

    def test_step(self, test_batch, test_idx):
        # if test_idx > 0:
        #     return
        ## use simulator pde loss
        # self.pde_loss = self.pde_loss_simulator

        h_unnorm, dx, dt, u_unnorm = test_batch
        self.h_ch = h_ch = h_unnorm.shape[-1]
        self.u_ch = u_ch = u_unnorm.shape[-1]

        gt = torch.cat([h_unnorm, u_unnorm], dim=-1)
        state_gt = self.data_transform(h_unnorm, u_unnorm)
        h = state_gt[..., 0:h_ch]  # b h w c
        u = state_gt[..., h_ch:u_ch+h_ch]  # b h w c

        n_samples = self.test_sparams.n_samples  # for each input condition draw 5 samples
        # print(f"Draw {n_samples} samples for each input condition")
        state_gt_rep = state_gt.repeat(n_samples, 1, 1, 1)
        h_rep = state_gt_rep[..., 0:h_ch]  # n*b h w c
        u_rep = state_gt_rep[..., h_ch:u_ch+h_ch]  # n*b h w c

        n_time_all = h.shape[1]
        n_time_h = self.test_sparams.n_time_h
        n_time_u = self.test_sparams.n_time_u
        return_last = self.test_sparams.return_last
        guide_dx = self.test_sparams.guide_dx

        if self.test_sparams.type == 'edm':
            xs = self.sample_edm(h_rep, u_rep, self.test_sparams, return_last=return_last, guide_dx=guide_dx)
        else:
            xs, _ = self.sample_with_repeat(h_rep, u_rep, self.test_sparams, return_last=return_last, guide_dx=guide_dx)

        # average across predictios for each input condition
        xs_mean = rearrange(xs, '(n b) t h w c -> n b t h w c', n=n_samples)
        xs_mean = torch.mean(xs_mean, dim=0)

        h_last = xs_mean[:, -1, :, :, 0:h_ch]  # b t h w c
        u_last = xs_mean[:, -1, :, :, h_ch:u_ch+h_ch]  # b t h w c

        loss_h = self.mae_criterion(h_last, h)
        loss_u = self.mae_criterion(u_last, u)

        # unnormalized error is calculated
        h_last_unnorm, u_last_unnorm = self.inverse_data_transform(h_last, u_last)

        loss_h_un = self.mae_criterion(h_last_unnorm, h_unnorm)
        loss_u_un = self.mae_criterion(u_last_unnorm, u_unnorm)

        # calculate the loss by all unnormalized values
        hu_last_unnorm = torch.cat([h_last_unnorm, u_last_unnorm], dim=-1)
        hu_gt_unnorm = torch.cat([h_unnorm, u_unnorm], dim=-1)
        mask = torch.ones_like(hu_last_unnorm)
        if n_time_h > 0:
            mask[:, :n_time_h, :, :h_ch] = 0.0  # mask the h values
        if n_time_u > 0:
            mask[:, :n_time_u, :, h_ch:u_ch+h_ch] = 0.0  # mask the u values
        loss_hu_un = self.mae_criterion_mask(hu_last_unnorm, hu_gt_unnorm, mask)

        # normalize the predicted values and gt to be between zero and 1
        gt_scaled = self.scale_each_min_max(state_gt)
        xs_scaled = self.scale_each_min_max(xs[:, -1])

        # error between scaled values
        if self.test_sparams.select_by_pde:
            # select the best sample judging by PDE error
            print("Use the best sample determined by PDE error")
            use_gt = self.test_sparams.use_gt_pde_select
            indices, xs_scaled_mean = self.get_best_by_pde_error(gt, xs_scaled, n_samples, use_gt)

            xs1 = rearrange(xs, '(n b) t h w c -> b n t h w c', n=n_samples)
            xs_mean = xs1[[torch.arange(len(xs1))[:, None], indices]]
            xs_mean = xs_mean.squeeze(dim=1)  # b n t h w c -> b h w c  (n=1, because we took the best)
        else:
            xs_scaled_mean = rearrange(xs_scaled, '(n b) h w c -> n b h w c', n=n_samples)
            xs_scaled_mean = torch.mean(xs_scaled_mean, dim=0)  # b h w c

        loss_h_scaled = self.mae_criterion(xs_scaled_mean[:, :, :, 0:h_ch], gt_scaled[:, :, :, 0:h_ch])
        loss_u_scaled = self.mae_criterion(xs_scaled_mean[:, :, :, h_ch:u_ch+h_ch], gt_scaled[:, :, :, h_ch:u_ch+h_ch])

        # correlation between prediction and gt
        corr_hu = self.correlation(xs_mean[:, -1], state_gt)
        corr_h = torch.mean(corr_hu[0:h_ch])
        corr_u = torch.mean(corr_hu[h_ch:u_ch+h_ch])
        self.log('test_corr_h', corr_h, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_corr_u', corr_u, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        # losses with masking
        if n_time_h < n_time_all:
            loss_h_known = self.mae_criterion(h_last[:, :n_time_h], h[:, :n_time_h])  # should be 0
            loss_h_known_sc = self.mae_criterion(xs_scaled_mean[:, :n_time_h, :, 0:h_ch],
                                                 gt_scaled[:, :n_time_h, :, 0:h_ch])
            loss_h_unkn_sc = self.mae_criterion(xs_scaled_mean[:, n_time_h:, :, 0:h_ch],
                                                gt_scaled[:, n_time_h:, :, 0:h_ch])

            self.log('test_h_known', loss_h_known, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            self.log('test_h_kn_scaled', loss_h_known_sc, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            self.log('test_h_unkn_scaled', loss_h_unkn_sc, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        if n_time_all > n_time_u > 0:
            loss_u_known = self.mae_criterion(u_last[:, :n_time_u], u[:, :n_time_u])
            loss_u_known_sc = self.mae_criterion(xs_scaled_mean[:, :n_time_u, :, h_ch:u_ch+h_ch],
                                                 gt_scaled[:, :n_time_u, :, h_ch:u_ch+h_ch])
            loss_u_unkn_sc = self.mae_criterion(xs_scaled_mean[:, n_time_u:, :, h_ch:u_ch+h_ch],
                                                gt_scaled[:, n_time_u:, :, h_ch:u_ch+h_ch])

            self.log('test_u_known', loss_u_known, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            self.log('test_u_kn_scaled', loss_u_known_sc, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            self.log('test_u_unkn_scaled', loss_u_unkn_sc, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        print()
        print(f"Loss h {loss_h}, loss h un {loss_h_un}")
        print(f"Loss u {loss_u}, loss u un {loss_u_un}")
        print(f"Loss hu un {loss_hu_un}")
        print(f"Loss h scaled {loss_h_scaled}, loss u scaled {loss_u_scaled}")

        self.log('test_mae_h', loss_h, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_mae_u', loss_u, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_mae_h_un', loss_h_un, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_mae_u_un', loss_u_un, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_mae_hu_un', loss_hu_un, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        self.log('test_mae_h_scaled', loss_h_scaled, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_mae_u_scaled', loss_u_scaled, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        # calculate PDE loss for each prediction and then average
        n_batch = len(h_unnorm)
        pred = xs[:, -1]
        cond = None
        pde_loss_sum = self.get_pde_loss(cond, pred, clamp_loss=False, do_rearrange=False)
        pde_loss = pde_loss_sum / n_samples / n_batch

        self.log('test_pde_loss', pde_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        print(f"Pde loss is {pde_loss}")
        pde_loss_gt_sum = self.get_pde_loss(cond, state_gt, clamp_loss=False, do_rearrange=False)
        pde_loss_gt = pde_loss_gt_sum / n_batch

        self.log('test_pde_loss_gt', pde_loss_gt, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        print(f"Pde loss gt is {pde_loss_gt}")

        if return_last:
            # to plot the last step for each sample
            xs_scaled = rearrange(xs_scaled, '(n b) h w c -> b h w n c', n=n_samples).unsqueeze(dim=1)  # t = 1
            xs = rearrange(xs[:, -1], '(n b) h w c -> b h w n c', n=n_samples).unsqueeze(dim=1)  # t = 1
        else:
            # we plot time trajectory for the first sample otherwise
            xs = rearrange(xs[:, ::2], '(n b) t h w c -> b t h w n c', n=n_samples)
            xs = xs[:, :, :, :, 0, :]  # take the first sample and treat timesteps as samples
            t_steps = xs.size(1)
            xs = rearrange(xs, 'b t h w c -> (b t) h w c')
            xs_scaled = self.scale_each_min_max(xs)
            xs_scaled = rearrange(xs_scaled, '(b t) h w c -> b h w t c', t=t_steps).unsqueeze(dim=1)  # n = 1

        if self.test_sparams.plot_scaled:
            xs_traj = xs_scaled
            gt_plot = gt_scaled
        else:
            xs_traj = xs
            gt_plot = state_gt

        return {'loss_h': loss_h, 'loss': loss_u, 'loss_h_un': loss_h_un, 'loss_u_un': loss_u_un,
                'test_mae_u_scaled': loss_u_scaled,
                'traj': xs_traj, 'gt': gt_plot}

    def get_pde_loss(self, cond, x_denoised, x_gt_unnorm=None, noise_level=None, clamp_loss=True, do_rearrange=True,
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

    def get_dx(self, cond, x_denoised, guide_dx=True):
        dx = torch.zeros_like(x_denoised)
        if guide_dx:
            dx1 = self.get_dx_pde(cond, x_denoised)
            has_nan = torch.any(torch.isnan(dx1)).item()
            if not has_nan:
                # Normalizing to unit norm -> the same result seems to be the same to clamp
                b, c, dim1, dim2 = dx1.shape
                dx = rearrange(dx1, 'b c h w -> b c (h w)')
                dx = F.normalize(dx, p=2, dim=2)
                dx = rearrange(dx, 'b c (h w) -> b c h w', h=dim1, w=dim2)
            else:
                print("Nan in dx")
        return dx

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

    def get_best_by_pde_error(self, gt, xs_scaled, n_samples, use_gt=True):
        """
        Select the best sample judging by PDE error
        """
        gt_rep = gt.repeat(n_samples, 1, 1, 1)
        gt_scaled, gt_min, gt_max = self.scale_each_min_max(gt_rep, return_min_max=True)

        xs_gt = self.scale_back_min_max(xs_scaled, gt_min, gt_max)

        target = gt_rep if use_gt else xs_gt
        pde_error_matrix = self.pde_loss(xs_gt, target, self.normalizer_input, self.normalizer_target)
        pde_error_matrix = rearrange(pde_error_matrix, '(n b) h w c -> b n (h w c)', n=n_samples)
        # xs_gt = rearrange(xs_gt, '(n b) h w c -> n b h w c', n=n_samples)

        pde_error = torch.mean(pde_error_matrix, dim=2)
        # return indices of the sample with smallest PDE error for each batch input
        _, indices = torch.min(pde_error, dim=1, keepdim=True)

        xs_scaled = rearrange(xs_scaled, '(n b) h w c -> b n h w c', n=n_samples)
        xs_best = xs_scaled[[torch.arange(len(xs_scaled))[:, None], indices]]
        xs_best = xs_best.squeeze(dim=1)  # b n h w c -> b h w c  (n=1, because we took the best)

        return indices, xs_best

    def recover_correct_scale(self, gt, xs_scaled_mean):
        gt_scaled, gt_min, gt_max = self.scale_each_min_max(gt, return_min_max=True)
        xs_mean = self.scale_back_min_max(xs_scaled_mean, gt_min, gt_max)
        return xs_mean

    @staticmethod
    def scale_back_min_max(state_scaled, state_min, state_max):
        state = rearrange(state_scaled, 'b h w c -> b c (h w)')
        state = state * (state_max - state_min) + state_min
        state = rearrange(state, 'b c (h w) -> b h w c', h=state_scaled.size(1), w=state_scaled.size(2))
        return state

    @staticmethod
    def scale_each_min_max(state, return_min_max=False):
        state_scaled = rearrange(state, 'b h w c -> b c (h w)')
        state_scaled_min = torch.min(state_scaled, dim=2, keepdim=True)[0]
        state_scaled_max = torch.max(state_scaled, dim=2, keepdim=True)[0]
        state_scaled = (state_scaled - state_scaled_min) / (state_scaled_max - state_scaled_min)
        state_scaled = rearrange(state_scaled, 'b c (h w) -> b h w c', h=state.size(1), w=state.size(2))

        if return_min_max:
            return state_scaled, state_scaled_min, state_scaled_max
        return state_scaled

    def compute_alpha(self, t):
        zero_point = torch.zeros(1).type_as(self.betas)
        betas = torch.cat([zero_point, self.betas], dim=0)
        a = (1 - betas).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample(self, h, u_noise, sparams, return_last=True, guide_dx=False):
        # adapted from
        # https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution/blob/main_v1/functions/denoising_step.py
        model = self.ema_model if self.ema_model is not None else self.model
        w = sparams.w

        # change shape
        h = rearrange(h.unsqueeze(dim=-1), 'b h w c -> b c h w')
        u_noise = rearrange(u_noise.unsqueeze(dim=-1), 'b h w c -> b c h w')
        cond = None

        # generate sequence
        if sparams.skip_type == "uniform":
            skip = self.num_timesteps // sparams.timesteps
            seq = range(0, self.num_timesteps, skip)
        elif sparams.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(self.num_timesteps * 0.8), sparams.timesteps) ** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError

        h_noise = torch.randn_like(h)  # add noise to the input

        a = (1 - self.betas).cumprod(dim=0)
        total_noise_levels = a.size(0)  # or self.num_timesteps

        # in the other paper they use nearest neighbor interpolation to fill in the gaps and
        # then they denoise it with less amount of steps for the final prediction
        h_t = h * a[total_noise_levels - 1].sqrt() + h_noise * (1.0 - a[total_noise_levels - 1]).sqrt()
        x = torch.cat([h_t, u_noise], dim=1)

        n = h.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        x0_t = None

        add_h_x0 = False  # False
        add_h_x_next = True  # True
        for i, j in zip(reversed(seq), reversed(seq_next)):
            with torch.no_grad():
                t = (torch.ones(n) * i)
                t = t.type_as(h)
                next_t = (torch.ones(n) * j)
                next_t = next_t.type_as(h)
                at = self.compute_alpha(t.long())
                at_next = self.compute_alpha(next_t.long())
                xt = xs[-1]

                x_self_cond = x0_t if self.model.self_condition else None

                dx_in = self.get_dx_input(cond, xt)
                if w is None or np.abs(w) < 0.001 or dx_in is None:
                    et = model(xt, t, x_self_cond=x_self_cond)
                else:
                    # use blending with coefficient w
                    et = (w + 1) * model(xt, t, x_self_cond=x_self_cond, dx=dx_in) \
                         - w * model(xt, t, x_self_cond=x_self_cond)

                # the same algorithm as classifier guidance in 'Guided diffusion'
                # https://arxiv.org/abs/2105.05233
                dx = self.get_dx_log_prob(h, xt, guide_dx)
                weight = 5.
                et = et - weight * (1 - at).sqrt() * dx

                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                # if clamp_func is not None:  # clamp the result for x0 to be in the specified region
                #     x0_t = clamp_func(x0_t)

                if add_h_x0:
                    x0_t[..., 0, :, :] = h[..., 0, :, :]  # add know part

                if abs(sparams.eta) > 1e-10:
                    c1 = sparams.eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                    c2 = ((1 - at_next) - c1 ** 2).sqrt()
                    xt_next = at_next.sqrt() * x0_t + c1 * torch.rand_like(x) + c2 * et
                else:
                    c2 = (1 - at_next).sqrt()
                    xt_next = at_next.sqrt() * x0_t + c2 * et

                if add_h_x_next:
                    # add know noisy part
                    # h_t = at_next.sqrt() * h + c2 * et  # this option works significantly worse
                    h_t = at_next.sqrt() * h + c2 * h_noise
                    xt_next[..., 0, :, :] = h_t[..., 0, :, :]

                if return_last:
                    # to save GRU memory
                    x0_preds = [x0_t]
                    xs = [xt_next]
                else:
                    x0_preds.append(x0_t)
                    xs.append(xt_next)

        xs = torch.stack(xs, dim=0)
        x0_preds = torch.stack(x0_preds, dim=0)

        xs = rearrange(xs, 't b c h w -> b t h w c')
        x0_preds = rearrange(x0_preds, 't b c h w -> b t h w c')

        return xs, x0_preds

    def sample_with_repeat(self, h, u, sparams, return_last=True, guide_dx=False):
        # adapted from
        # https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution/blob/main_v1/functions/denoising_step.py
        model = self.ema_model if self.ema_model is not None else self.model

        n_repeat = sparams.n_repeat
        n_time_h = sparams.n_time_h
        n_time_u = sparams.n_time_u
        w = sparams.w

        # change shape
        hu = torch.cat([h, u], dim=-1)
        hu = rearrange(hu, 'b h w c -> b c h w')
        hu_mask = torch.ones_like(hu)
        hu_mask[:, 0:self.h_ch, n_time_h:, :] = 0.0  # use only the first n_time_h for conditioning
        hu_mask[:, self.h_ch:self.h_ch + self.u_ch, n_time_u:, :] = 0.0  # use only the first n_time_u for conditioning
        cond = None

        # generate sequence
        if sparams.skip_type == "uniform":
            skip = self.num_timesteps // sparams.timesteps
            seq = range(0, self.num_timesteps, skip)
        elif sparams.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(self.num_timesteps * 0.8), sparams.timesteps) ** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError

        hu_noise = torch.randn_like(hu)  # add noise to the input u

        a = (1 - self.betas).cumprod(dim=0)
        total_noise_levels = a.size(0)  # or self.num_timesteps

        # prepare input
        hu_t_known = hu * a[total_noise_levels - 1].sqrt() + hu_noise * (1.0 - a[total_noise_levels - 1]).sqrt()
        x = hu_t_known * hu_mask + hu_noise * (1.0 - hu_mask)

        n = hu.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        x0_t = None

        for i, j in zip(reversed(seq), reversed(seq_next)):
            with torch.no_grad():
                t = (torch.ones(n) * i)
                t = t.type_as(h)
                next_t = (torch.ones(n) * j)
                next_t = next_t.type_as(h)
                at = self.compute_alpha(t.long())
                at_next = self.compute_alpha(next_t.long())
                xt = xs[-1]

                for k in range(n_repeat):
                    # et = (w + 1) * model(xt, t, dx) - w * model(xt, t)  ## mix inpainted results with not inpainted
                    x_self_cond = x0_t if self.model.self_condition else None

                    dx_in = self.get_dx_input(cond, xt)
                    if w is None or np.abs(w) < 0.001 or dx_in is None:
                        et = model(xt, t, x_self_cond=x_self_cond)
                    else:
                        # use blending with coefficient w
                        et = (w + 1) * model(xt, t, x_self_cond=x_self_cond, dx=dx_in) \
                             - w * model(xt, t, x_self_cond=x_self_cond)

                    # the same algorithm as classifier guidance in 'Guided diffusion'
                    # https://arxiv.org/abs/2105.05233
                    dx = self.get_dx_log_prob(h, xt, guide_dx)
                    weight = 5.
                    et = et - weight * (1 - at).sqrt() * dx

                    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                    # add know part
                    x0_t = hu * hu_mask + x0_t * (1.0 - hu_mask)

                    if k < n_repeat - 1:
                        xt = at.sqrt() * x0_t + (1 - at).sqrt() * et

                if abs(sparams.eta) > 1e-10:
                    c1 = sparams.eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                    c2 = ((1 - at_next) - c1 ** 2).sqrt()
                    xt_next = at_next.sqrt() * x0_t + c1 * torch.rand_like(x) + c2 * et
                else:
                    c2 = (1 - at_next).sqrt()
                    xt_next = at_next.sqrt() * x0_t + c2 * et

                # add know noisy part
                hu_t_known = at_next.sqrt() * hu + c2 * hu_noise
                xt_next = hu_t_known * hu_mask + xt_next * (1.0 - hu_mask)

                if return_last:
                    # to save GRU memory
                    x0_preds = [x0_t]
                    xs = [xt_next]
                else:
                    x0_preds.append(x0_t)
                    xs.append(xt_next)

        xs = torch.stack(xs, dim=0)
        x0_preds = torch.stack(x0_preds, dim=0)

        xs = rearrange(xs, 't b c h w -> b t h w c')
        x0_preds = rearrange(x0_preds, 't b c h w -> b t h w c')

        return xs, x0_preds

    def get_denoised(self, model, xt, t, cond=None, x_self_cond=None, dx=None, w=None):
        # scale the output such that it corresponds to EDM sampling mechanism
        xt = xt.to(torch.float32)
        t = t.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = torch.float32

        sigma = t
        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = self.num_timesteps - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)

        x_self_cond = (c_in * x_self_cond).to(torch.float32) if x_self_cond is not None else None

        # if conditioning information is concatenated to the input it should be scaled as well
        if cond is not None:
            # cond = (c_in * cond).to(torch.float32)  # works a bit worse
            cond = (c_in * cond).to(torch.float32) if self.model.cat_condition else cond

        if dx is not None:
            dx = (c_in * dx).to(torch.float32)

        if w is None or np.abs(w) < 0.001 or (cond is None and dx is None):
            F_x = model(c_in * xt, c_noise.flatten(), cond=cond, x_self_cond=x_self_cond, dx=dx)
        else:
            # use blending with coefficient w (classifier-free guidance)
            F_x = (w + 1) * model(c_in * xt, c_noise.flatten(), cond=cond, x_self_cond=x_self_cond, dx=dx) \
                  - w * model(c_in * xt, c_noise.flatten(), x_self_cond=x_self_cond)

        assert F_x.dtype == dtype

        D_x = c_skip * xt + c_out * F_x
        return D_x, F_x

    def round_sigma(self, sigma, return_index=False):
        sigma32 = sigma.to(torch.float32)
        if self.edm_steps.device != sigma.device:
            self.edm_steps = self.edm_steps.type_as(sigma32)
        steps = self.edm_steps
        index = torch.cdist(sigma32.reshape(1, -1, 1), steps.reshape(1, -1, 1)).argmin(2)
        result = index if return_index else steps[index.flatten()]
        result = result.type_as(sigma)
        return result.reshape(sigma.shape)

    def sample_edm(self, h, u, sparams, return_last=True, guide_dx=False):
        # the method is adopted from original repo:
        # https://github.com/NVlabs/edm/blob/main/generate.py
        n_repeat = sparams.n_repeat
        n_time_h = sparams.n_time_h
        n_time_u = sparams.n_time_u
        w = sparams.w

        model = self.ema_model if self.ema_model is not None else self.model
        hu = torch.cat([h, u], dim=-1)
        hu = rearrange(hu, 'b h w c -> b c h w')
        hu_noise = torch.randn_like(hu)
        hu_mask = torch.ones_like(hu)
        hu_mask[:, 0:self.h_ch, n_time_h:, :] = 0.0  # use only the first n_time_h for conditioning
        hu_mask[:, self.h_ch:self.h_ch+self.u_ch, n_time_u:, :] = 0.0  # use only the first n_time_u for conditioning
        cond = None

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

        ## add known part
        aT = self.compute_alpha(t_steps[0].long())
        hu_t_known = hu * aT.sqrt() + hu_noise * (1.0 - aT).sqrt()
        x = hu_t_known * hu_mask + hu_noise * (1.0 - hu_mask)

        # Main sampling loop.
        x_next = x.to(torch.float64) * t_steps[0]
        xs = [x_next]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            S_min = sparams.S_min
            S_max = float(sparams.S_max)
            gamma = min(sparams.S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = self.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * sparams.S_noise * torch.randn_like(x_cur)

            for k in range(n_repeat):
                # Euler step.
                dx_in = self.get_dx_input(cond, x_hat)
                denoised, et = self.get_denoised(model, x_hat, t_hat, dx=dx_in, w=w)
                denoised = denoised.to(torch.float64)

                dx = self.get_dx_log_prob(h, denoised, guide_dx)
                weight = 5.
                d_cur = (x_hat - denoised) / t_hat - weight * dx
                x_next = x_hat + (t_next - t_hat) * d_cur

                # Apply 2nd order correction.
                if i < num_steps - 1:
                    dx_in = self.get_dx_input(cond, x_next)
                    denoised, et = self.get_denoised(model, x_next, t_next, dx=dx_in, w=w)
                    denoised = denoised.to(torch.float64)

                    dx = self.get_dx_log_prob(h, denoised, guide_dx)
                    d_prime = (x_next - denoised) / t_next - weight * dx
                    # d_prime = (x_next - denoised) / t_next
                    x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

                # replace the known part
                at_next = self.compute_alpha(t_next.long())
                hu_t_known = at_next.sqrt() * hu + (1 - at_next).sqrt() * hu_noise
                x_next = hu_t_known * hu_mask + x_next * (1.0 - hu_mask)

                if k < n_repeat - 1:
                    # add noise to go back from t_next to a new t_hat
                    gamma1 = np.sqrt(2) - 1
                    t_hat = self.round_sigma(t_next + gamma1 * t_next)
                    x_hat = x_next + (t_hat ** 2 - t_next ** 2).sqrt() * sparams.S_noise * torch.randn_like(x_next)

            if i == num_steps - 1:
                # replace the known part
                x_next = hu * hu_mask + x_next * (1.0 - hu_mask)

            if return_last:
                # to save GPU memory
                xs = [x_next]
            else:
                xs.append(x_next)

        xs = torch.stack(xs, dim=0)
        xs = rearrange(xs, 't b c h w -> b t h w c')
        return xs


class PlCondDdim(PlDdim):
    def __init__(self, hparams):
        super().__init__(hparams)

        # probability of using conditioning on the known region during training
        self.cond_p = hparams.model.cond_p if hasattr(hparams.model, 'cond_p') else 0.8

    def get_inp_stats_shape(self, hparams):
        ch = hparams.model.in_channels
        size = (ch,) if ch > 1 else ()
        return size

    def get_tar_stats_shape(self, hparams):
        ch = hparams.model.out_ch
        size = (ch,) if ch > 1 else ()
        return size

    def inverse_data_transform_u(self, u):
        if self.rescaled:
            u = (u + 1.0) / 2.0

        if self.normalization == "min_max":
            u = torch.clamp(u, 0.0, 1.0)

        u = self.normalizer_target(u, inverse=True)
        return u

    def get_cond_in(self, h, u, dx, dt):
        ## node_type is not calculated in the if statements below
        cond_ch = self.model.cond_channels - 1 if self.node_type else self.model.cond_channels
        if cond_ch == self.h_ch:
            cond_in = h
        elif cond_ch == self.h_ch + self.u_ch:
            # duplicate IC for the other state as an extra input
            # if the information is added as constant values, return them by a dataloader instead
            n_times = u.shape[1]  # u.shape = b, t, x, c
            u_ic = u[:, 0:1].repeat(1, n_times, 1, 1)
            cond_in = torch.cat([h, u_ic], dim=-1)
        elif cond_ch == self.h_ch + 2:
            # h.shape = b h w c
            t_grid, x_grid = dt, dx
            cond_in = torch.cat([h, t_grid, x_grid], dim=-1)
        elif cond_ch == self.h_ch + self.u_ch + 2:
            n_times = u.shape[1]  # u.shape = b, t, x, c
            u_ic = u[:, 0:1].repeat(1, n_times, 1, 1)
            t_grid, x_grid = dt, dx
            cond_in = torch.cat([h, u_ic, t_grid, x_grid], dim=-1)
        else:
            raise(f"Number of conditional channels {cond_ch} should be changed to match "
                  f"the known states channels {self.h_ch}")

        if self.node_type:
            b, hc, wc, c = h.shape
            # 0: domain point, 1 - a boundary point
            node_type = torch.zeros((b, hc, wc, 1))
            node_type = node_type.type_as(h).long()
            node_type[:, 0] = 1
            node_type[:, -1] = 1
            node_type[:, :, 0] = 1
            node_type[:, :, -1] = 1
            cond_in = torch.cat([cond_in, node_type], dim=-1)

        return cond_in

    def training_step(self, train_batch, batch_idx):
        h_unnorm, dx, dt, u_unnorm = train_batch
        self.h_ch = h_ch = h_unnorm.shape[-1]
        self.u_ch = u_ch = u_unnorm.shape[-1]

        x = self.data_transform(h_unnorm, u_unnorm)
        n = x.size(0)

        # use h as conditioning input and u as the target to be denoised
        h = x[..., 0:h_ch]
        u = x[..., h_ch:h_ch + u_ch]
        cond_in = self.get_cond_in(h, u, dx, dt)
        cond_in = rearrange(cond_in, 'b h w c -> b c h w')

        u = rearrange(u, 'b h w c -> b c h w')

        noise = torch.randn_like(u)

        # antithetic sampling
        t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,))
        t = t.type_as(x).long()
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
        output, x0_t = self.forward(u, t, noise, cond=cond_in)

        loss = self.criteria(output, noise)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        if self.pde_loss_lambda > 0.:
            noise_level = t if self.pde_loss_prop_t else None
            x_gt_unnorm = torch.cat([h_unnorm, u_unnorm], dim=-1) if self.use_gt_pde else None
            pde_loss = self.get_pde_loss(h, x0_t, x_gt_unnorm=x_gt_unnorm, noise_level=noise_level, clamp_loss=True)
            self.log('train_pde_loss', pde_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            loss = loss + self.pde_loss_lambda * pde_loss

        return loss

    def validation_step(self, val_batch, batch_idx):
        if (self.current_epoch + 1) % 100 != 0 and self.current_epoch != 0:  # plot validation images every 100 epochs
            return {"epoch": self.current_epoch}

        h_unnorm, dx, dt, u_unnorm = val_batch
        self.h_ch = h_ch = h_unnorm.shape[-1]
        self.u_ch = u_ch = u_unnorm.shape[-1]

        state_gt = self.data_transform(h_unnorm, u_unnorm)
        h = state_gt[..., :h_ch]  # b h w c
        u = state_gt[..., h_ch:u_ch+h_ch]  # b h w c

        u_noise = torch.randn_like(u)
        guide_dx = self.sparams.guide_dx
        cond_in = self.get_cond_in(h, u, dx, dt)
        if self.sparams.type == 'edm':
            xs = self.sample_edm(cond_in, u_noise, self.sparams, return_last=True, guide_dx=guide_dx)
        else:
            xs, _ = self.sample(cond_in, u_noise, self.sparams, return_last=True, guide_dx=guide_dx)

        u_last = xs[:, -1, :, :, :u_ch]  # b t h w c

        loss_u = self.mae_criterion(u_last, u)

        # unnormalized error is calculated
        u_last_unnorm = self.inverse_data_transform_u(u_last)

        loss_u_un = self.mae_criterion(u_last_unnorm, u_unnorm)

        # normalize the predicted values and gt to be between zero and 1
        gt_scaled = self.scale_each_min_max(state_gt)
        xs_scaled = self.scale_each_min_max(xs[:, -1])  # xs contains just u part of the state

        # error between scaled values
        loss_u_scaled = self.mae_criterion(xs_scaled, gt_scaled[:, :, :, h_ch:u_ch+h_ch])

        self.log('val_mae_u', loss_u, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log('val_mae_u_un', loss_u_un, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        self.log('val_mae_u_scaled', loss_u_scaled, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        # correlation between prediction and gt
        corr_u = self.correlation(xs[:, -1], state_gt[..., h_ch:u_ch+h_ch])
        corr_u = torch.mean(corr_u)
        self.log('val_corr_u', corr_u, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        n_batch = len(h_unnorm)
        h_in = state_gt[..., 0:h_ch]  # b h w c
        x0_t = xs[:, -1]
        pde_loss_sum = self.get_pde_loss(h_in, x0_t, clamp_loss=False, do_rearrange=False)
        pde_loss = pde_loss_sum / n_batch

        self.log('val_pde_loss', pde_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        if self.sparams.plot_scaled:
            xs_traj = xs_scaled.unsqueeze(dim=1)
            gt_plot = gt_scaled[..., h_ch:u_ch+h_ch]
        else:
            xs_traj = xs[:, -1].unsqueeze(dim=1)
            gt_plot = state_gt[..., h_ch:u_ch+h_ch]

        return {"epoch": self.current_epoch, # 'val_mae_un_loss': val_mae_un_loss,
                'loss': loss_u, 'loss_u_un': loss_u_un, 'val_loss_u_scaled': loss_u_scaled,
                'traj': xs_traj, 'gt': gt_plot}

    def test_step(self, test_batch, test_idx):
        h_unnorm, dx, dt, u_unnorm = test_batch
        self.h_ch = h_ch = h_unnorm.shape[-1]
        self.u_ch = u_ch = u_unnorm.shape[-1]

        state_gt = self.data_transform(h_unnorm, u_unnorm)
        h = state_gt[..., :h_ch]
        u = state_gt[..., h_ch:u_ch+h_ch]  # b h w c

        n_samples = self.test_sparams.n_samples  # for each input condition draw 5 samples
        return_last = self.test_sparams.return_last
        guide_dx = self.test_sparams.guide_dx
        state_gt_rep = state_gt.repeat(n_samples, 1, 1, 1)
        cond_in = self.get_cond_in(h, u, dx, dt)  # n*b h w c
        cond_in_rep = cond_in.repeat(n_samples, 1, 1, 1)

        u_rep = u.repeat(n_samples, 1, 1, 1)  # n*b h w c

        u_noise = torch.randn_like(u_rep)

        if self.test_sparams.type == 'edm':
            xs = self.sample_edm(cond_in_rep, u_noise, self.test_sparams, return_last=return_last, guide_dx=guide_dx)
        else:
            xs, _ = self.sample(cond_in_rep, u_noise, self.test_sparams, return_last=return_last, guide_dx=guide_dx)

        # average across predictions for each input condition
        xs_mean = rearrange(xs, '(n b) t h w c -> n b t h w c', n=n_samples)
        xs_mean = torch.mean(xs_mean, dim=0)
        u_last = xs_mean[:, -1, :, :, :u_ch]  # b t h w c
        loss_u = self.mae_criterion(u_last, u)

        # unnormalized error is calculated
        u_last_unnorm = self.inverse_data_transform_u(u_last)
        loss_u_un = self.mae_criterion(u_last_unnorm, u_unnorm)

        # normalize the predicted values and gt to be between zero and 1
        gt_scaled = self.scale_each_min_max(state_gt)
        xs_scaled = self.scale_each_min_max(xs[:, -1])

        # error between scaled values for the averaged prediction or the best prediction selected by PDE error
        if self.test_sparams.select_by_pde:
            # select the best sample judging by PDE error
            print("Use the best sample determined by PDE error")
            # concatenate predicted u and conditioning h for PDE loss
            gt = torch.cat([h_unnorm, u_unnorm], dim=-1)
            h_rep = h.repeat(n_samples, 1, 1, 1)  # n*b h w c
            h_rep_scaled = self.scale_each_min_max(h_rep.unsqueeze(dim=-1))
            xs_h_scaled = torch.cat([h_rep_scaled, xs_scaled], dim=-1)
            use_gt = self.test_sparams.use_gt_pde_select
            indices, xs_h_scaled_mean = self.get_best_by_pde_error(gt, xs_h_scaled, n_samples, use_gt)
            xs_scaled_mean = xs_h_scaled_mean[..., -1:]

            xs1 = rearrange(xs, '(n b) t h w c -> b n t h w c', n=n_samples)
            xs_mean = xs1[[torch.arange(len(xs1))[:, None], indices]]
            xs_mean = xs_mean.squeeze(dim=1)  # b n t h w c -> b h w c  (n=1, because we took the best)
        else:
            xs_scaled_mean = rearrange(xs_scaled, '(n b) h w c -> n b h w c', n=n_samples)
            xs_scaled_mean = torch.mean(xs_scaled_mean, dim=0)

        loss_u_scaled = self.mae_criterion(xs_scaled_mean, gt_scaled[:, :, :, h_ch:u_ch+h_ch])

        # correlation between prediction and gt
        corr_u = self.correlation(xs_mean[:, -1], state_gt[..., h_ch:u_ch+h_ch])
        corr_u = torch.mean(corr_u)
        self.log('test_corr_u', corr_u, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        print()
        print(f"Loss u {loss_u}, loss u un {loss_u_un}")
        print(f"Loss u scaled {loss_u_scaled}")

        self.log('test_mae_u', loss_u, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_mae_u_un', loss_u_un, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        self.log('test_mae_u_scaled', loss_u_scaled, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        # calculate PDE loss for each prediction and then average
        n_batch = len(h_unnorm)
        pred = xs[:, -1]
        pde_loss_sum = self.get_pde_loss(state_gt_rep[..., 0:h_ch], pred, clamp_loss=False, do_rearrange=False)
        pde_loss = pde_loss_sum / n_samples / n_batch

        self.log('test_pde_loss', pde_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        print(f"Pde loss is {pde_loss}")
        pde_loss_gt_sum = self.get_pde_loss(state_gt[..., 0:h_ch], state_gt[..., h_ch:u_ch+h_ch],
                                            clamp_loss=False, do_rearrange=False)
        pde_loss_gt = pde_loss_gt_sum / n_batch

        self.log('test_pde_loss_gt', pde_loss_gt, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        print(f"Pde loss gt is {pde_loss_gt}")

        if self.test_sparams.plot_scaled:
            xs_scaled = rearrange(xs_scaled, '(n b) h w c -> b h w n c', n=n_samples).unsqueeze(dim=1)  # t = 1
            xs_traj = xs_scaled
            gt_plot = gt_scaled[..., h_ch:u_ch+h_ch]
        else:
            xs = rearrange(xs[:, -1], '(n b) h w c -> b h w n c', n=n_samples).unsqueeze(dim=1)
            xs_traj = xs
            gt_plot = state_gt[..., h_ch:u_ch+h_ch]

        return {'loss': loss_u, 'loss_u_un': loss_u_un, 'test_mae_u_scaled': loss_u_scaled,
                'traj': xs_traj, 'gt': gt_plot}

    def print_unroll_metrics(self, xs, h_unnorm, u_unnorm, state_gt, n_samples):
        ## use simulator pde loss
        self.pde_loss = self.pde_loss_simulator

        h_ch = h_unnorm.shape[-1]
        u_ch = u_unnorm.shape[-1]

        ## compare pde loss for ground truth vs predicted values
        xs = rearrange(xs, '(n b) t h w c -> n b t h w c', n=n_samples)
        cond = state_gt[..., 0:h_ch]  # b h w c
        pde_unroll_error_h, pde_unroll_error_u = 0.0, 0.0
        n_unroll_res = []
        for i in range(len(xs)):
            x0_t = xs[i][:, -1]  # take t=-1

            # unroll system dynamics from the initial conditions predicted by diffusion model and gt conditioning
            pred_unnorm = self.get_x_unnorm(cond, x0_t, do_rearrange=False)
            pde_unroll_error, unroll_res = self.pde_loss.unroll_loss(pred_unnorm, pred_unnorm,
                                                                     self.normalizer_input, self.normalizer_target,
                                                                     return_unroll=True)
            n_unroll_res.append(unroll_res)
            pde_unroll_error_h += torch.sum(pde_unroll_error[..., 0:h_ch])
            pde_unroll_error_u += torch.sum(pde_unroll_error[..., h_ch:u_ch+h_ch])
        pde_unroll_error_h /= len(xs)
        pde_unroll_error_u /= len(xs)

        n_unroll_res = torch.stack(n_unroll_res, dim=0)  # n b h w c
        unroll_res = rearrange(n_unroll_res, 'n b h w c -> (n b) h w c')

        x_gt1 = torch.cat([h_unnorm, u_unnorm], dim=-1)
        pde_unroll_error_gt, unroll_res_gt = self.pde_loss.unroll_loss(x_gt1, x_gt1,
                                                                       self.normalizer_input, self.normalizer_target,
                                                                       return_unroll=True)
        pde_unroll_error_gt_h = torch.sum(pde_unroll_error_gt[..., 0:h_ch])
        pde_unroll_error_gt_u = torch.sum(pde_unroll_error_gt[..., h_ch:u_ch+h_ch])
        self.log('test_pde_unroll_error', pde_unroll_error_u, prog_bar=True, on_epoch=True,
                 on_step=False, sync_dist=True)
        print(f"Tests Unroll error h is {pde_unroll_error_h}")
        print(f"Tests Unroll error u is {pde_unroll_error_u}")

        self.log('test_pde_unroll_error_gt', pde_unroll_error_gt_u, prog_bar=True, on_epoch=True,
                 on_step=False, sync_dist=True)
        print(f"Tests Unroll error gt h is {pde_unroll_error_gt_h}")
        print(f"Tests Unroll error gt u is {pde_unroll_error_gt_u}")

        unroll_res_gt_rep = unroll_res_gt.repeat(n_samples, 1, 1, 1)
        pde_unrolled_mae_h = self.mae_criterion(unroll_res[..., 0:h_ch], unroll_res_gt_rep[..., 0:h_ch])
        pde_unrolled_mae_u = self.mae_criterion(unroll_res[..., h_ch:u_ch+h_ch], unroll_res_gt_rep[..., h_ch:u_ch+h_ch])

        self.log("test_pde_unrolled_mae_h", pde_unrolled_mae_h, prog_bar=True, on_epoch=True, on_step=False,
                 sync_dist=True)
        self.log("test_pde_unrolled_mae_u", pde_unrolled_mae_u, prog_bar=True, on_epoch=True, on_step=False,
                 sync_dist=True)

        unroll_res = rearrange(unroll_res, '(n b) h w c -> b h w n c', n=n_samples).unsqueeze(dim=1)  # add dim t = 1
        return unroll_res, unroll_res_gt

    def get_x_unnorm(self, cond, x_denoised, do_rearrange=True):
        h, u_denoised = cond, x_denoised
        if do_rearrange:
            h = rearrange(h, 'b c h w -> b h w c')
            u_denoised = rearrange(u_denoised, 'b c h w -> b h w c')

        h_unnorm, u_unnorm = self.inverse_data_transform(h, u_denoised)
        x_unnorm = torch.cat([h_unnorm, u_unnorm], dim=-1)
        return x_unnorm

    def get_pde_loss(self, cond, x_denoised, x_gt_unnorm=None, noise_level=None, clamp_loss=True, do_rearrange=True,
                     reduce=True):
        h = cond[..., :self.h_ch]  # conditioning may have extra info on top of one of the states
        u_denoised = x_denoised
        h = h.to(torch.float32)
        u_denoised = u_denoised.to(torch.float32)

        if do_rearrange:
            h = rearrange(h, 'b c h w -> b h w c')
            u_denoised = rearrange(u_denoised, 'b c h w -> b h w c')

        h_unnorm, u_unnorm = self.inverse_data_transform(h, u_denoised)
        x_unnorm = torch.cat([h_unnorm, u_unnorm], dim=-1)

        if x_gt_unnorm is None:
            x_gt_unnorm = x_unnorm

        pde_error_dx_matrix = self.pde_loss(x_unnorm, x_gt_unnorm, self.normalizer_input, self.normalizer_target,
                                            return_d=False, calc_prob=False, clamp_loss=clamp_loss)

        if len(pde_error_dx_matrix.shape) > 3:
            ## select just u -> bad for darcy system!
            ## sum over the channel location
            pde_error_dx_matrix = torch.sum(pde_error_dx_matrix, dim=-1)

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
        h = cond[:, :self.h_ch]  # conditioning may have extra info on top of one of the states
        u_denoised = x_denoised
        h = h.to(torch.float32)
        u_denoised = u_denoised.to(torch.float32)
        h = rearrange(h, 'b c h w -> b h w c')
        u_denoised = rearrange(u_denoised, 'b c h w -> b h w c')

        h_unnorm, u_unnorm = self.inverse_data_transform(h, u_denoised)
        x_unnorm = torch.cat([h_unnorm, u_unnorm], dim=-1)

        return_d = True
        pde_error_dx_matrix = self.pde_loss(x_unnorm, x_unnorm, self.normalizer_input, self.normalizer_target,
                                            return_d, calc_prob)

        # rearrange the matrix back
        pde_error_dx_matrix = rearrange(pde_error_dx_matrix, 'b h w c -> b c h w')

        if len(pde_error_dx_matrix.shape) > 3:
            ## select just u -> bad for darcy system!
            ## sum over the channel location
            if calc_prob:
                pde_error_dx_matrix = torch.mean(pde_error_dx_matrix, dim=1, keepdim=True)
            else:
                pde_error_dx_matrix = torch.sum(pde_error_dx_matrix, dim=1)

        return pde_error_dx_matrix

    def sample(self, h, u_noise, sparams, return_last=True, guide_dx=False):
        # adapted from
        # https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution/blob/main_v1/functions/denoising_step.py
        model = self.ema_model if self.ema_model is not None else self.model
        w = sparams.w

        # change shape
        h = rearrange(h, 'b h w c -> b c h w')
        u_noise = rearrange(u_noise, 'b h w c -> b c h w')

        # generate sequence
        if sparams.skip_type == "uniform":
            skip = self.num_timesteps // sparams.timesteps
            seq = range(0, self.num_timesteps, skip)
        elif sparams.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(self.num_timesteps * 0.8), sparams.timesteps) ** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError

        x = u_noise

        n = h.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        x0_t = None

        for i, j in zip(reversed(seq), reversed(seq_next)):
            with torch.no_grad():
                t = (torch.ones(n) * i)
                t = t.type_as(h)
                next_t = (torch.ones(n) * j)
                next_t = next_t.type_as(h)
                at = self.compute_alpha(t.long())
                at_next = self.compute_alpha(next_t.long())
                xt = xs[-1]

                x_sc = x0_t if self.model.self_condition else None

                dx_in = self.get_dx_input(h, xt)  # if there is no dx conditioning
                if w is None or np.abs(w) < 0.001:
                    et = model(xt, t, cond=h, x_self_cond=x_sc, dx=dx_in)
                else:
                    # use blending with coefficient w
                    et = (w + 1) * model(xt, t, cond=h, x_self_cond=x_sc, dx=dx_in) - w * model(xt, t, x_self_cond=x_sc)

                # the same algorithm as classifier guidance in 'Guided diffusion'
                # https://arxiv.org/abs/2105.05233
                dx = self.get_dx_log_prob(h, xt, guide_dx)
                weight = 5.
                et = et - weight * (1 - at).sqrt() * dx

                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                ## PI-DDIM subtracts dx directly from x_next
                ## https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution/blob/main_v1/functions/denoising_step.py
                if abs(sparams.eta) > 1e-10:
                    c1 = sparams.eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                    c2 = ((1 - at_next) - c1 ** 2).sqrt()
                    xt_next = at_next.sqrt() * x0_t + c1 * torch.rand_like(x) + c2 * et  # - dx
                else:
                    c2 = (1 - at_next).sqrt()
                    xt_next = at_next.sqrt() * x0_t + c2 * et  # - dx

                if return_last:
                    x0_preds = [x0_t]
                    xs = [xt_next]
                else:
                    x0_preds.append(x0_t)
                    xs.append(xt_next)

        xs = torch.stack(xs, dim=0)
        x0_preds = torch.stack(x0_preds, dim=0)

        xs = rearrange(xs, 't b c h w -> b t h w c')
        x0_preds = rearrange(x0_preds, 't b c h w -> b t h w c')

        return xs, x0_preds

    def sample_edm(self, h, u_noise, sparams, return_last=True, guide_dx=False):
        # the method is adopted from original repo:
        # https://github.com/NVlabs/edm/blob/main/generate.py

        model = self.ema_model if self.ema_model is not None else self.model
        w = sparams.w

        # change shape
        h = rearrange(h, 'b h w c -> b c h w')
        u_noise = rearrange(u_noise, 'b h w c -> b c h w')

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sparams.sigma_min, self.sigma_min)
        sigma_max = min(sparams.sigma_max, self.sigma_max)
        num_steps = sparams.timesteps

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float64)
        step_indices = step_indices.type_as(h.to(torch.float64))
        t_steps = (sigma_max ** (1 / sparams.rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / sparams.rho) - sigma_max ** (1 / sparams.rho))) ** sparams.rho
        t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0
        x = u_noise

        # Main sampling loop.
        x_next = x.to(torch.float64) * t_steps[0]
        xs = [x_next]
        x_sc = None
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            S_min = sparams.S_min
            S_max = float(sparams.S_max)
            gamma = min(sparams.S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = self.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * sparams.S_noise * torch.randn_like(x_cur)

            # Euler step.
            dx_in = self.get_dx_input(h, x_hat)
            denoised, et1 = self.get_denoised(model, x_hat, t_hat, cond=h, x_self_cond=x_sc, dx=dx_in, w=w)
            x_sc = self.get_self_cond_edm(denoised)
            denoised = denoised.to(torch.float64)

            dx = self.get_dx_log_prob(h, denoised, guide_dx)
            weight = 5.
            d_cur = (x_hat - denoised) / t_hat - weight * dx / t_hat

            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                dx_in = self.get_dx_input(h, x_next)
                denoised, et2 = self.get_denoised(model, x_next, t_next, cond=h, x_self_cond=x_sc, dx=dx_in, w=w)
                x_sc = self.get_self_cond_edm(denoised)
                denoised = denoised.to(torch.float64)

                dx = self.get_dx_log_prob(h, denoised, guide_dx)
                d_prime = (x_next - denoised) / t_next - weight * dx / t_hat

                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

            if return_last:
                xs = [x_next]
            else:
                xs.append(x_next)

        xs = torch.stack(xs, dim=0)
        xs = rearrange(xs, 't b c h w -> b t h w c')
        return xs

    def get_self_cond_edm(self, denoised):
        # self conditioning doesn't help during edm sampling for this model
        return None


class PlCondEdm(PlCondDdim):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data = 1.  # 0.5 in the original paper

        self.sigma_min = 0.002
        self.sigma_max = 80

        self.log_lr = False

    @staticmethod
    def get_edm_sampler_params():
        sparams = {}
        sparams['name'] = 'edm'
        sparams['type'] = 'edm'
        sparams['timesteps'] = 50
        sparams['sigma_min'] = 0.002
        sparams['sigma_max'] = 80
        sparams['rho'] = 7
        sparams['S_churn'] = 15.0
        sparams['S_min'] = 0
        sparams['S_max'] = 'inf'
        sparams['S_noise'] = 1
        sparams['n_samples'] = 5
        sparams['n_repeat'] = 2
        sparams['n_time_h'] = 128
        sparams['n_time_u'] = 0
        sparams['return_last'] = True
        sparams['select_by_pde'] = False
        sparams['use_gt_pde_select'] = True
        sparams['guide_dx'] = False
        sparams['w'] = 0.0
        sparams['plot_scaled'] = False
        sparams = DotDict(sparams)

        return sparams

    def set_test_sampler_params(self, params):
        if params.type != 'edm':
            print("Model with EDM preconditioning supports only EDM sampler ")
            params = self.get_edm_sampler_params()

        self.test_sparams = params

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

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, noise: torch.Tensor, cond: torch.Tensor = None) \
            -> torch.Tensor:
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

        x_self_cond = None
        if self.model.self_condition and torch.rand(1) < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_precond(x_noise, sigma.float(), cond, dx=dx)
                x_self_cond = x_self_cond.detach()

        output = self.model_precond(x_noise, sigma.float(), cond, x_self_cond=x_self_cond, dx=dx)
        x0_t = output
        return output, x0_t

    def get_loss_weight(self, sigma):
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        return weight

    def training_step(self, train_batch, batch_idx):
        h_unnorm, dx, dt, u_unnorm = train_batch
        self.h_ch = h_ch = h_unnorm.shape[-1]
        self.u_ch = u_ch = u_unnorm.shape[-1]

        x = self.data_transform(h_unnorm, u_unnorm)

        # use h as conditioning input and u as the target to be denoised
        h = x[..., 0:h_ch]
        u = x[..., h_ch:h_ch + u_ch]
        cond_in = self.get_cond_in(h, u, dx, dt)
        cond_in = rearrange(cond_in, 'b h w c -> b c h w')

        u = rearrange(u, 'b h w c -> b c h w')
        noise = torch.randn_like(u)

        # calculate sigma and loss weight
        rnd_normal = torch.randn([u.shape[0], 1, 1, 1])
        rnd_normal = rnd_normal.type_as(u)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = self.get_loss_weight(sigma)

        D_x, x0_t = self.forward(u, sigma, noise, cond=cond_in)
        loss = self.criteria(D_x, u, weight)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        if self.pde_loss_lambda > 0.:
            noise_level = sigma if self.pde_loss_prop_t else None
            x_gt_unnorm = torch.cat([h_unnorm, u_unnorm], dim=-1) if self.use_gt_pde else None
            pde_loss = self.get_pde_loss(h, x0_t, x_gt_unnorm=x_gt_unnorm, noise_level=noise_level, clamp_loss=True)
            self.log('train_pde_loss', pde_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            loss = loss + self.pde_loss_lambda * pde_loss

        if self.log_lr:
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('lr', lr, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        return loss

    def sample(self, h, u_noise, sparams, return_last=True, guide_dx=False):
        raise NotImplementedError("Only EDM sampler is supported for the model with EDM pre-conditioning")

    def sample_with_repeat(self, h, u, sparams, return_last=True, guide_dx=False):
        raise NotImplementedError("Only EDM sampler is supported for the model with EDM pre-conditioning")

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

    def get_self_cond_edm(self, denoised):
        self_cond = self.ema_model.ma_model.self_condition if self.ema_model is not None else self.model.self_condition
        x_sc = denoised if self_cond else None
        return x_sc
