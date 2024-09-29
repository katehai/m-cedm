'''
Adapted from: https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d_time.py
and https://github.com/jaggbow/magnet/blob/main/models/fno_2d.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pytorch_lightning as pl
from models.fno_2d import SpectralConv2d
from models.losses import LpLoss, CorrelationLoss, ScaledMaeLoss, DownsampledLoss, MaskedLoss
from models.normalizer import set_target_unnorm, Normalizer
from models.loss_helper import get_pde_loss_function


class FnoState2d(nn.Module):
    def __init__(self, hparams):
        super(FnoState2d, self).__init__()

        self.modes_1 = hparams.modes_1
        self.modes_2 = hparams.modes_2
        self.width = hparams.width
        self.num_layers = hparams.num_layers
        self.padding_t = hparams.padding_t
        self.padding_x = hparams.padding_x
        self.input_size = hparams.input_size
        self.state_size = hparams.state_size
        self.inst_norm = hparams.inst_norm
        self.decoder_norm = None

        if self.inst_norm:
            self.norm = nn.InstanceNorm2d(self.width)

        self.fc0 = nn.Linear(self.input_size + 2, self.width)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.state_size)

        fourier_layers = []
        conv_layers = []
        for i in range(self.num_layers):
            fourier_layers.append(SpectralConv2d(self.width, self.width, self.modes_1, self.modes_2))
            conv_layers.append(nn.Conv2d(self.width, self.width, 1))
        self.fourier_layers = nn.ModuleList(fourier_layers)
        self.conv_layers = nn.ModuleList(conv_layers)

    def set_norm_decoder(self, decoder_norm):
        self.decoder_norm = decoder_norm  # unnormalize the output of the model if needed

    def forward(self, u: torch.Tensor, dx: torch.Tensor = None, dt: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of FNO network.
        The input to the forward pass has the shape [batch, H, T, C].
        1. Add grid x and grid t as channel dimension to the input channels
        2. Lift the input to the desired channel dimension by self.fc0
        3. 5 (default) FNO layers
        4. Project from the channel space to the output space by self.fc1 and self.fc2.
        The output has the shape [batch, time_future, H, W].
        Args:
            u (torch.Tensor): input tensor of shape[batch, H, T, C_in]
            dx (torch.Tensor): optional spatial distances, otherwise absolute spatial coordinates are used
            dt (torch.Tensor): optional temporal distances, otherwise absolute time coordinates are used
        Returns: torch.Tensor: output has the shape [batch, H, T, C_out]
        """
        B, H, T, C = u.shape  # B, X, T, C

        if dx is not None and dt is not None:
            gridx, gridt = dx, dt
            if len(dx.shape) == 1:
                gridx = dx[:, None, None, None].to(u.device).repeat(1, H, T, 1)
            if len(dt.shape) == 1:
                gridt = dt[:, None, None, None].to(u.device).repeat(1, H, T, 1)
        else:
            gridx, gridt = self.get_grid(u.shape)

        grid = torch.cat((gridx, gridt), dim=-1).to(u.device)

        x = torch.cat((u, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # B, X, T, C -> B, C, X, T
        x = F.pad(x, (0, self.padding_t, 0, self.padding_x))  # time dimension is not periodic, so we need to add padding

        for fourier, conv in zip(self.fourier_layers, self.conv_layers):
            if self.inst_norm:
                x1 = self.norm(fourier(self.norm(x)))
            else:
                x1 = fourier(x)
            x2 = conv(x)
            x = x1 + x2
            x = F.gelu(x)

        # add padding if the domain is not periodic
        if self.padding_t > 0:
            x = x[..., :-self.padding_t]  # B, C, X, T

        if self.padding_x > 0:
            x = x[:, :, :-self.padding_x]  # B, C, X, T

        x = x.permute(0, 2, 3, 1)  # B, X, T, C
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        # B, X, T, C -> B, T, X, C
        x = x.permute(0, 2, 1, 3)

        if self.decoder_norm is not None:
            x = self.decoder_norm(x)

        return x

    def get_grid(self, shape):
        bs, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([bs, 1, size_y, 1])
        gridt = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridt = gridt.reshape(1, 1, size_y, 1).repeat([bs, size_x, 1, 1])
        return gridx, gridt


class PlFnoStateReconstr2d(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters()
        self.model = FnoState2d(hparams)
        self.time_history = hparams.time_history
        self.lr = hparams.lr
        self.weight_decay = hparams.weight_decay
        self.factor = hparams.factor
        self.step_size = hparams.step_size
        self.loss = hparams.loss

        # normalization parameters to be loaded with model weights
        self.normalization = 'gauss'
        self.norm_input = True  # the flags correspond to the normalization of inputs and targets in dataloader
        self.norm_target = True
        self.normalizer_input = Normalizer(stats_shape=tuple(hparams.norm_shape))
        self.normalizer_target = Normalizer(stats_shape=tuple(hparams.norm_shape))

        if self.loss == 'l1':
            self.criterion = nn.L1Loss()
        elif self.loss == 'l2':
            self.criterion = nn.MSELoss()
        elif self.loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
        elif self.loss == 'lp':
            self.criterion = LpLoss(p=2, reduction='sum')

        self.mse_criterion = DownsampledLoss(type='l2')  # nn.MSELoss()
        self.mae_criterion = DownsampledLoss(type='l1')  # nn.L1Loss()

        self.correlation = CorrelationLoss()
        self.scaled_mae = ScaledMaeLoss()

        pde_loss_f, pde_loss_sim_f = get_pde_loss_function(system='swe', flip_xy=False)  # the loss is overriden by set_pde_loss
        self.pde_loss = pde_loss_f
        self.pde_loss_simulator = pde_loss_sim_f

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
        # The stats are not saved correctly
        # set the data normalization statistics
        if self.normalization == "min_max":
            self.normalizer_input.set_stats(remove_dim(stats["input_min"]), remove_dim(stats["input_min_max"]))
            self.normalizer_target.set_stats(remove_dim(stats["target_min"]), remove_dim(stats["target_min_max"]))
        else:
            self.normalizer_input.set_stats(remove_dim(stats["input_mean"]), remove_dim(stats["input_std"]))
            self.normalizer_target.set_stats(remove_dim(stats["target_mean"]), remove_dim(stats["target_std"]))

        return

    def forward(self, u: torch.Tensor, dx: torch.Tensor = None, dt: torch.Tensor = None) -> torch.Tensor:
        return self.model(u, dx, dt)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.factor)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
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
        u, x, t, s = train_batch
        s, s_unnorm = self.get_unnorm_target(s)
        if len(x.shape) == 1 and len(t.shape) == 1:
            dx, dt = x, t  # (B, )
        else:
            dx = dt = None  # the grid is generated in the model

        t_history = self.time_history  # 100
        s_hist_gt = s[:, :t_history]  # B, T_history, N=128, C
        u_history = u[:, :t_history]  # B, T_history, N, C
        u_hist_inp = u_history.permute(0, 2, 1, 3)  # B, N, T_history, C

        s_hist_hat = self.model(u_hist_inp, dx, dt)   # B, T_history, N, C

        loss = self.criterion(s_hist_hat, s_hist_gt)
        mae_loss = self.mae_criterion(s_hist_hat, s_hist_gt)

        # calculate loss on unnormalized data
        s_hist_gt_unnorm = s_unnorm[:, :t_history]
        s_hist_hat_unnorm = self.normalizer_target(s_hist_hat, inverse=True)
        mae_un_loss = self.mae_criterion(s_hist_hat_unnorm, s_hist_gt_unnorm)

        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=u.shape[0])
        self.log('train_mae_u', mae_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=u.shape[0])
        self.log('train_mae_u_un', mae_un_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=u.shape[0])

        return loss

    def validation_step(self, val_batch, batch_idx):
        u, x, t, s = val_batch
        s, s_unnorm = self.get_unnorm_target(s)
        if len(x.shape) == 1 and len(t.shape) == 1:
            dx, dt = x, t  # (B, )
        else:
            dx = dt = None  # the grid is generated in the model

        t_history = self.time_history  # 128
        s_hist_gt = s[:, :t_history]  # B, T_history, N=128, C
        u_history = u[:, :t_history]  # B, T_history, N, C
        u_hist_inp = u_history.permute(0, 2, 1, 3)  # B, N, T_history, C

        s_hist_hat = self.model(u_hist_inp, dx, dt)  # B, T_history, N

        loss = self.criterion(s_hist_hat, s_hist_gt)
        mae_loss = self.mae_criterion(s_hist_hat, s_hist_gt)
        corr = self.correlation(s_hist_hat, s_hist_gt)
        corr = torch.mean(corr)

        # calculate loss on unnormalized data
        s_hist_gt_unnorm = s_unnorm[:, :t_history]
        s_hist_hat_unnorm = self.normalizer_target(s_hist_hat, inverse=True)
        mae_un_loss = self.mae_criterion(s_hist_hat_unnorm, s_hist_gt_unnorm)

        # naming is the same as for other models
        loss_u_scaled = self.scaled_mae(s_hist_hat, s_hist_gt)

        pde_loss = self.get_pde_loss(u_history, s_hist_hat, clamp_loss=False, reduce=True)
        pde_loss_gt = self.get_pde_loss(u[:, :t_history], s_hist_gt[:, :t_history], clamp_loss=False, reduce=True)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=u.shape[0])
        self.log('val_mae_u', mae_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=u.shape[0])
        self.log('val_mae_u_un', mae_un_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=u.shape[0])

        self.log('val_corr', corr, prog_bar=True, on_epoch=True, on_step=False, batch_size=u.shape[0])
        self.log('val_mae_u_scaled', loss_u_scaled, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        self.log('val_pde_loss', pde_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log('val_pde_loss_gt', pde_loss_gt, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        pred = s_hist_hat
        target = s_hist_gt

        return {'pred': pred, 'target': target, 'loss': loss}

    def test_step(self, test_batch, batch_idx):
        u, x, t, s = test_batch
        down_factor = self.trainer.datamodule.down_factor if self.trainer.datamodule.down_interp else 1

        s, s_unnorm = self.get_unnorm_target(s)
        if len(x.shape) == 1 and len(t.shape) == 1:
            dx, dt = x, t  # (B, )
        else:
            dx = dt = None  # the grid is generated in the model

        t_history = self.time_history  # 100
        s_hist_gt = s[:, :t_history]  # B, T_history, N=128, C
        u_history = u[:, :t_history]  # B, T_history, N, C
        u_hist_inp = u_history.permute(0, 2, 1, 3)  # B, N, T_history, C

        s_hist_hat = self.model(u_hist_inp, dx, dt)  # B, T_history, N, C

        loss = self.criterion(s_hist_hat, s_hist_gt)
        mae_loss = self.mae_criterion(s_hist_hat, s_hist_gt, down_factor)
        corr = self.correlation(s_hist_hat, s_hist_gt)
        corr = torch.mean(corr)

        # calculate loss on unnormalized data
        s_hist_gt_unnorm = s_unnorm[:, :t_history]
        s_hist_hat_unnorm = self.normalizer_target(s_hist_hat, inverse=True)
        mae_un_loss = self.mae_criterion(s_hist_hat_unnorm, s_hist_gt_unnorm, down_factor)

        # naming is the same as for other models
        loss_u_scaled = self.scaled_mae(s_hist_hat, s_hist_gt)

        pde_loss = self.get_pde_loss(u_history, s_hist_hat, clamp_loss=False, reduce=True)
        pde_loss_gt = self.get_pde_loss(u_history, s_hist_gt, clamp_loss=False, reduce=True)

        self.log('test_loss', loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=u.shape[0])
        self.log('test_mae_u', mae_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=u.shape[0])
        self.log('test_mae_u_un', mae_un_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=u.shape[0])

        self.log('test_corr', corr, prog_bar=True, on_epoch=True, on_step=False, batch_size=u.shape[0])
        self.log('test_mae_u_scaled', loss_u_scaled, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        self.log('test_pde_loss', pde_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_pde_loss_gt', pde_loss_gt, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        pred = s_hist_hat
        target = s_hist_gt

        return {'pred': pred, 'target': target, 'loss': loss}

    def get_pde_loss(self, cond, pred, x_gt_unnorm=None, clamp_loss=False, reduce=True):
        cond_unnorm = self.normalizer_input(cond, inverse=True)
        pred_unnorm = self.normalizer_target(pred, inverse=True)

        # print("Cond shape", cond_unnorm.shape)
        # print("Pred unnorm shape", pred_unnorm.shape)

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


class PlFnoTimePred2d(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters()
        self.model = FnoState2d(hparams)
        self.time_history = hparams.time_history
        self.lr = hparams.lr
        self.weight_decay = hparams.weight_decay
        self.factor = hparams.factor
        self.step_size = hparams.step_size
        self.loss = hparams.loss

        # normalization parameters to be loaded with model weights
        self.normalization = 'gauss'
        self.norm_input = True  # the flags correspond to the normalization of inputs and targets in dataloader
        self.norm_target = True
        self.normalizer_input = Normalizer(stats_shape=tuple(hparams.norm_shape))
        self.normalizer_target = Normalizer(stats_shape=tuple(hparams.norm_shape))

        if self.loss == 'l1':
            self.criterion = nn.L1Loss()
        elif self.loss == 'l2':
            self.criterion = nn.MSELoss()
        elif self.loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
        elif self.loss == 'lp':
            self.criterion = LpLoss(p=2, reduction='sum')

        self.mse_criterion = DownsampledLoss(type='l2')  # nn.MSELoss()
        self.mae_criterion = DownsampledLoss(type='l1')  # nn.L1Loss()

        self.correlation = CorrelationLoss()
        self.scaled_mae = ScaledMaeLoss()

        pde_loss_f, pde_loss_sim_f = get_pde_loss_function(system='swe', flip_xy=False)  # the loss is overriden by set_pde_loss
        self.pde_loss = pde_loss_f
        self.pde_loss_simulator = pde_loss_sim_f

    def set_pde_loss_function(self, system, flip_xy):
        pde_loss_f, pde_loss_sim_f = get_pde_loss_function(system, flip_xy)

        self.pde_loss = pde_loss_f
        self.pde_loss_simulator = pde_loss_sim_f

    def setup(self, stage: str = None) -> None:
        stats = self.trainer.datamodule.get_norm_stats()
        self.norm_input = stats.norm_input
        self.norm_target = stats.norm_target
        # if stage == "fit":
        # The stats are not saved correctly
        # set the data normalization statistics
        if self.normalization == "min_max":
            self.normalizer_input.set_stats(stats["input_min"], stats["input_min_max"])
            self.normalizer_target.set_stats(stats["target_min"], stats["target_min_max"])
        else:
            self.normalizer_input.set_stats(stats["input_mean"], stats["input_std"])
            self.normalizer_target.set_stats(stats["target_mean"], stats["target_std"])

        return

    def forward(self, u: torch.Tensor, dx: torch.Tensor = None, dt: torch.Tensor = None) -> torch.Tensor:
        return self.model(u, dx, dt)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.factor)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }

    def get_unnorm_input(self, u):
        if self.norm_input:
            u_unnorm = self.normalizer_input(u, inverse=True)
        else:
            u_unnorm = u
            u = self.normalizer_input(u, inverse=False)
        return u, u_unnorm

    def get_unnorm_target(self, s):
        if self.norm_target:
            s_unnorm = self.normalizer_target(s, inverse=True)
        else:
            s_unnorm = s
            s = self.normalizer_target(s, inverse=False)
        return s, s_unnorm

    def training_step(self, train_batch, batch_idx):
        # combine u and s as inputs and predict future steps
        u, x, t, s = train_batch
        u, u_unnorm = self.get_unnorm_input(u)
        s, s_unnorm = self.get_unnorm_target(s)
        if len(x.shape) == 1 and len(t.shape) == 1:
            dx, dt = x, t  # (B, )
        else:
            dx = dt = None  # the grid is generated in the model

        state = torch.cat([u, s], dim=-1)

        t_history = self.time_history  # 64
        state_target = state[:, t_history:]  # B, T_future, N, C, T_future = 64
        state_inp = state[:, :t_history]  # B, T_history, N=128, C, T_history = 64
        state_inp = state_inp.permute(0, 2, 1, 3)  # B, N, T_history, C
        state_pred = self.model(state_inp, dx, dt)   # B, T_history, N, C

        loss = self.criterion(state_pred, state_target)
        mae_loss = self.mae_criterion(state_pred, state_target)

        # calculate loss on unnormalized data
        state_target_unnorm = torch.cat([u_unnorm, s_unnorm], dim=-1)[:, t_history:]
        _, u_pred_unnorm = self.get_unnorm_input(state_pred[:, :, :, :u.shape[-1]])
        _, s_pred_unnorm = self.get_unnorm_target(state_pred[:, :, :, u.shape[-1]:])
        state_pred_unnorm = torch.cat([u_pred_unnorm, s_pred_unnorm], dim=-1)

        mae_un_loss = self.mae_criterion(state_pred_unnorm, state_target_unnorm)

        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=u.shape[0])
        self.log('train_mae_u', mae_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=u.shape[0])
        self.log('train_mae_u_un', mae_un_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=u.shape[0])

        return loss

    def validation_step(self, val_batch, batch_idx):
        u, x, t, s = val_batch
        u, u_unnorm = self.get_unnorm_input(u)
        s, s_unnorm = self.get_unnorm_target(s)
        if len(x.shape) == 1 and len(t.shape) == 1:
            dx, dt = x, t  # (B, )
        else:
            dx = dt = None  # the grid is generated in the model

        state = torch.cat([u, s], dim=-1)

        t_history = self.time_history  # 64
        state_target = state[:, t_history:]  # B, T_future, N, C, T_future = 64
        state_inp = state[:, :t_history]  # B, T_history, N=128, C, T_history = 64
        state_inp = state_inp.permute(0, 2, 1, 3)  # B, N, T_history, C
        state_pred = self.model(state_inp, dx, dt)  # B, T_history, N, C

        loss = self.criterion(state_pred, state_target)
        mae_loss = self.mae_criterion(state_pred, state_target)
        corr = self.correlation(state_pred, state_target)
        corr = torch.mean(corr)

        # calculate loss on unnormalized data
        state_target_unnorm_full = torch.cat([u_unnorm, s_unnorm], dim=-1)
        state_target_unnorm = state_target_unnorm_full[:, t_history:]
        _, u_pred_unnorm = self.get_unnorm_input(state_pred[:, :, :, :u.shape[-1]])
        _, s_pred_unnorm = self.get_unnorm_target(state_pred[:, :, :, u.shape[-1]:])
        state_pred_unnorm = torch.cat([u_pred_unnorm, s_pred_unnorm], dim=-1)

        mae_un_loss = self.mae_criterion(state_pred_unnorm, state_target_unnorm)

        # naming is the same as for other models
        loss_u_scaled = self.scaled_mae(state_pred, state_target)

        # concatenate states by time
        state_pred_unnorm_full = torch.cat([state_target_unnorm_full[:, :t_history], state_pred_unnorm], dim=1)
        pde_loss = self.get_pde_loss(state_pred_unnorm_full, clamp_loss=False, reduce=True)
        pde_loss_gt = self.get_pde_loss(state_target_unnorm_full, clamp_loss=False, reduce=True)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=u.shape[0])
        self.log('val_mae_u', mae_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=u.shape[0])
        self.log('val_mae_u_un', mae_un_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=u.shape[0])

        self.log('val_corr', corr, prog_bar=True, on_epoch=True, on_step=False, batch_size=u.shape[0])
        self.log('val_mae_u_scaled', loss_u_scaled, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        self.log('val_pde_loss', pde_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log('val_pde_loss_gt', pde_loss_gt, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        return {'pred': state_pred_unnorm_full, 'target': state_target_unnorm_full, 'loss': loss}

    def test_step(self, test_batch, batch_idx):
        u, x, t, s = test_batch
        down_factor = self.trainer.datamodule.down_factor if self.trainer.datamodule.down_interp else 1

        u, u_unnorm = self.get_unnorm_input(u)
        s, s_unnorm = self.get_unnorm_target(s)
        if len(x.shape) == 1 and len(t.shape) == 1:
            dx, dt = x, t  # (B, )
        else:
            dx = dt = None  # the grid is generated in the model

        state = torch.cat([u, s], dim=-1)

        t_history = self.time_history  # 64
        state_target = state[:, t_history:]  # B, T_future, N, C, T_future = 64
        state_inp = state[:, :t_history]  # B, T_history, N=128, C, T_history = 64
        state_inp = state_inp.permute(0, 2, 1, 3)  # B, N, T_history, C
        state_pred = self.model(state_inp, dx, dt)  # B, T_history, N, C

        loss = self.criterion(state_pred, state_target)
        mae_loss = self.mae_criterion(state_pred, state_target, down_factor)
        corr = self.correlation(state_pred, state_target)
        corr = torch.mean(corr)

        # calculate loss on unnormalized data
        state_target_unnorm_full = torch.cat([u_unnorm, s_unnorm], dim=-1)
        state_target_unnorm = state_target_unnorm_full[:, t_history:]
        _, u_pred_unnorm = self.get_unnorm_input(state_pred[:, :, :, :u.shape[-1]])
        _, s_pred_unnorm = self.get_unnorm_target(state_pred[:, :, :, u.shape[-1]:])
        state_pred_unnorm = torch.cat([u_pred_unnorm, s_pred_unnorm], dim=-1)

        mae_un_loss = self.mae_criterion(state_pred_unnorm, state_target_unnorm, down_factor)

        # naming is the same as for other models
        loss_u_scaled = self.scaled_mae(state_pred, state_target)

        # concatenate states by time
        state_pred_unnorm_full = torch.cat([state_target_unnorm_full[:, :t_history], state_pred_unnorm], dim=1)
        pde_loss = self.get_pde_loss(state_pred_unnorm_full, clamp_loss=False, reduce=True)
        pde_loss_gt = self.get_pde_loss(state_target_unnorm_full, clamp_loss=False, reduce=True)

        self.log('test_loss', loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=u.shape[0])
        self.log('test_mae_u', mae_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=u.shape[0])
        self.log('test_mae_u_un', mae_un_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=u.shape[0])

        self.log('test_corr', corr, prog_bar=True, on_epoch=True, on_step=False, batch_size=u.shape[0])
        self.log('test_mae_u_scaled', loss_u_scaled, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        self.log('test_pde_loss', pde_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_pde_loss_gt', pde_loss_gt, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        return {'pred': state_pred_unnorm_full, 'target': state_target_unnorm_full, 'loss': loss}

    def get_pde_loss(self, x_unnorm, x_gt_unnorm=None, clamp_loss=False, reduce=True):
        if x_gt_unnorm is None:
            x_gt_unnorm = x_unnorm

        pde_error_dx_matrix = self.pde_loss(x_unnorm, x_gt_unnorm, self.normalizer_input, self.normalizer_target,
                                            return_d=False, calc_prob=False, clamp_loss=clamp_loss)

        if reduce:
            n_batch = x_unnorm.shape[0]
            pde_loss = torch.sum(pde_error_dx_matrix) / n_batch
        else:
            pde_loss = pde_error_dx_matrix

        return pde_loss


class PlFnoStateTimePred2d(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters()
        self.model_state = PlFnoStateReconstr2d(hparams.hparams_state)
        self.model_time = PlFnoTimePred2d(hparams.hparams_time)

        self.time_history = hparams.time_history
        self.flip_xy = False

        # normalization parameters to be loaded with model weights
        self.normalization = 'gauss'
        self.norm_input = True  # the flags correspond to the normalization of inputs and targets in dataloader
        self.norm_target = True
        self.normalizer_input = Normalizer(stats_shape=tuple(hparams.norm_shape))
        self.normalizer_target = Normalizer(stats_shape=tuple(hparams.norm_shape))

        self.mae_criterion = DownsampledLoss(type='l1')  # nn.L1Loss()
        self.mae_full_criterion = MaskedLoss()

        # the loss is overriden by set_pde_loss
        pde_loss_f, pde_loss_sim_f = get_pde_loss_function(system='swe', flip_xy=False)
        self.pde_loss = pde_loss_f
        self.pde_loss_simulator = pde_loss_sim_f

    def set_pde_loss_function(self, system, flip_xy):
        self.flip_xy = flip_xy
        flip_xy = False
        pde_loss_f, pde_loss_sim_f = get_pde_loss_function(system, flip_xy)

        self.pde_loss = pde_loss_f
        self.pde_loss_simulator = pde_loss_sim_f

    def setup(self, stage: str = None) -> None:
        stats = self.trainer.datamodule.get_norm_stats()
        self.norm_input = stats.norm_input
        self.norm_target = stats.norm_target
        # if stage == "fit":
        # The stats are not saved correctly
        # set the data normalization statistics
        if self.normalization == "min_max":
            self.normalizer_input.set_stats(stats["input_min"], stats["input_min_max"])
            self.normalizer_target.set_stats(stats["target_min"], stats["target_min_max"])
        else:
            self.normalizer_input.set_stats(stats["input_mean"], stats["input_std"])
            self.normalizer_target.set_stats(stats["target_mean"], stats["target_std"])

        return

    def forward(self, u: torch.Tensor, dx: torch.Tensor = None, dt: torch.Tensor = None) -> torch.Tensor:
        return self.model(u, dx, dt)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.factor)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }

    def get_unnorm_input(self, u):
        if self.norm_input:
            u_unnorm = self.normalizer_input(u, inverse=True)
        else:
            u_unnorm = u
            u = self.normalizer_input(u, inverse=False)
        return u, u_unnorm

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
        u, x, t, s = test_batch
        down_factor = self.trainer.datamodule.down_factor if self.trainer.datamodule.down_interp else 1

        u, u_unnorm = self.get_unnorm_input(u)
        s, s_unnorm = self.get_unnorm_target(s)
        if len(x.shape) == 1 and len(t.shape) == 1:
            dx, dt = x, t  # (B, )
        else:
            dx = dt = None  # the grid is generated in the model

        # run state reconstruction first
        t_history = self.time_history # 128  # self.time_history  # 64
        u_history = u[:, :t_history]  # B, T_history, N, C
        u_hist_inp = u_history.permute(0, 2, 1, 3)  # B, N, T_history, C

        s_hist_hat = self.model_state.model(u_hist_inp, dx, dt)  # B, T_history, N, C

        # calculate loss on unnormalized data
        s_hist_gt_unnorm = s_unnorm[:, :t_history]
        s_hist_hat_unnorm = self.normalizer_target(s_hist_hat, inverse=True)
        mae_un_loss_reconstr = self.mae_criterion(s_hist_hat_unnorm, s_hist_gt_unnorm, down_factor)

        if self.flip_xy:
            state_reconstr = torch.cat([s_hist_hat, u_history], dim=-1)  # B, T_history, N=128, C, T_history = 64
        else:
            state_reconstr = torch.cat([u_history, s_hist_hat], dim=-1)  # B, T_history, N=128, C, T_history = 64
        state_inp = state_reconstr.permute(0, 2, 1, 3)  # B, N, T_history, C
        state_pred = self.model_time.model(state_inp, dx, dt)  # B, T_history, N, C

        # calculate loss on unnormalized data
        if self.flip_xy:
            state_target_unnorm_full = torch.cat([s_unnorm, u_unnorm], dim=-1)
            state_target_unnorm = state_target_unnorm_full[:, t_history:]
            _, u_pred_unnorm = self.get_unnorm_input(state_pred[:, :, :, s.shape[-1]:])
            _, s_pred_unnorm = self.get_unnorm_target(state_pred[:, :, :, :s.shape[-1]])
            state_pred_unnorm = torch.cat([s_pred_unnorm, u_pred_unnorm], dim=-1)

            # concatenate states by time
            state_history_unnorm = torch.cat([s_hist_hat_unnorm, u_unnorm[:, :t_history]], dim=-1)
            state_pred_unnorm_full = torch.cat([state_history_unnorm, state_pred_unnorm], dim=1)

            mask = torch.ones_like(state_target_unnorm_full)
            mask[:, :t_history, :, s.shape[-1]:] = 0
        else:
            state_target_unnorm_full = torch.cat([u_unnorm, s_unnorm], dim=-1)
            state_target_unnorm = state_target_unnorm_full[:, t_history:]
            _, u_pred_unnorm = self.get_unnorm_input(state_pred[:, :, :, :u.shape[-1]])
            _, s_pred_unnorm = self.get_unnorm_target(state_pred[:, :, :, u.shape[-1]:])
            state_pred_unnorm = torch.cat([u_pred_unnorm, s_pred_unnorm], dim=-1)

            # concatenate states by time
            state_history_unnorm = torch.cat([u_unnorm[:, :t_history], s_hist_hat_unnorm], dim=-1)
            state_pred_unnorm_full = torch.cat([state_history_unnorm, state_pred_unnorm], dim=1)

            mask = torch.ones_like(state_target_unnorm_full)
            mask[:, :t_history, :, :u.shape[-1]] = 0

        mae_un_loss_time_pred = self.mae_criterion(state_pred_unnorm, state_target_unnorm, down_factor)

        pde_loss = self.get_pde_loss(state_pred_unnorm_full, clamp_loss=False, reduce=True)
        pde_loss_gt = self.get_pde_loss(state_target_unnorm_full, clamp_loss=False, reduce=True)

        mae_un_loss = self.mae_full_criterion(state_pred_unnorm_full, state_target_unnorm_full, mask)

        self.log('test_mae_un_rec', mae_un_loss_reconstr, prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=u.shape[0])
        self.log('test_mae_un_pred', mae_un_loss_time_pred, prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=u.shape[0])

        self.log('test_mae_un', mae_un_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=u.shape[0])

        self.log('test_pde_loss', pde_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_pde_loss_gt', pde_loss_gt, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        return {'pred': state_pred_unnorm_full, 'target': state_target_unnorm_full, 'loss': mae_un_loss}

    def get_pde_loss(self, x_unnorm, x_gt_unnorm=None, clamp_loss=False, reduce=True):
        if x_gt_unnorm is None:
            x_gt_unnorm = x_unnorm

        if self.flip_xy:
            # input and target normalizers are flipped
            pde_error_dx_matrix = self.pde_loss(x_unnorm, x_gt_unnorm, self.normalizer_target, self.normalizer_input,
                                                return_d=False, calc_prob=False, clamp_loss=clamp_loss)
        else:
            pde_error_dx_matrix = self.pde_loss(x_unnorm, x_gt_unnorm, self.normalizer_input, self.normalizer_target,
                                                return_d=False, calc_prob=False, clamp_loss=clamp_loss)

        if reduce:
            n_batch = x_unnorm.shape[0]
            pde_loss = torch.sum(pde_error_dx_matrix) / n_batch
        else:
            pde_loss = pde_error_dx_matrix

        return pde_loss
