import torch
import torch.nn as nn
from einops import rearrange
from models.normalizer import GaussianNormalizeDecoder


class MultiLoss(nn.Module):
    """
    Loss function is reduced by summation over channel dimension and by other reduction method over other dims
    """
    def __init__(self, loss="mse", reduction="mean"):
        super().__init__()
        self.loss = loss
        if self.loss == 'l1':
            self.criterion = nn.L1Loss(reduction='none')
        elif self.loss == 'l2' or self.loss == 'mse':
            self.criterion = nn.MSELoss(reduction='none')
        elif self.loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss(reduction='none')

        self.reduction = reduction

    def forward(self, pred, target):
        loss_matrix = self.criterion(pred, target)
        loss_matrix = torch.sum(loss_matrix, dim=-1)  # Sum over channel dimension

        if self.reduction == 'mean':
            loss = torch.mean(loss_matrix, dim=(1, 2))
            loss = torch.mean(loss)
            # loss = torch.mean(loss_matrix)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_matrix)
        else:
            loss = loss_matrix

        return loss


class NoiseEstimationLoss(nn.Module):
    """
    Loss function is reduced by summation over all dimensions except the batch dim. Other reduction is used instead
    """
    def __init__(self, reduction="mean"):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.reduction = reduction

    def forward(self, pred, target, weight=1.):
        loss_matrix = weight * self.criterion(pred, target)
        loss_matrix = torch.sum(loss_matrix, dim=(1, 2, 3))  # Sum over channel dimension

        if self.reduction == 'mean':
            loss = torch.mean(loss_matrix)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_matrix)
        else:
            loss = loss_matrix

        return loss


class MaskedLoss(nn.Module):
    def __init__(self, loss='l1'):
        super().__init__()
        self.criterion = nn.L1Loss(reduction='sum') if loss == 'l1' else nn.MSELoss(reduction='sum')

    def forward(self, pred, target, mask, loss_dim=None):
        pred = pred * mask
        target = target * mask

        if loss_dim is None:
            loss = self.criterion(pred, target)
            n_elements = torch.sum(mask)
        else:
            loss = self.criterion(pred[..., loss_dim], target[..., loss_dim])
            n_elements = torch.sum(mask[..., loss_dim])
        loss = loss / n_elements
        return loss


class DownsampledLoss(nn.Module):
    def __init__(self, type='l1'):
        super().__init__()
        self.criterion = nn.L1Loss() if type == 'l1' else nn.MSELoss()

    def forward(self, pred, target, down_factor=1):
        if down_factor > 1:
            each_x = 2 ** (down_factor - 1)
            pred = pred[:, ::each_x, ::each_x]
            target = target[:, ::each_x, ::each_x]

        loss = self.criterion(pred, target)
        return loss


class CorrelationLoss(nn.Module):
    def __init__(self, reduction="none"):
        super().__init__()
        self.reduction = reduction

    @staticmethod
    def calculate_correlation(x, y):
        x_bar = x - torch.mean(x, dim=1, keepdim=True)  # b (h w) c
        y_bar = y - torch.mean(y, dim=1, keepdim=True)  # b (h w) c

        cov_diag_unnorm = torch.sum((y_bar * x_bar), dim=1)  # / (N - 1), N = x_bar.shape[1]
        var_x_unnorm = torch.sum(x_bar * x_bar, dim=1)  # / (N - 1)
        var_y_unnorm = torch.sum(y_bar * y_bar, dim=1)  # / (N - 1)
        denominator = torch.sqrt(var_x_unnorm * var_y_unnorm)

        zero_mask = denominator == 0
        denominator[zero_mask] += 1e-7

        corr = cov_diag_unnorm / denominator
        corr_mean = torch.mean(corr, dim=0)  # average over the batch dimension, shape (c,)
        return corr_mean

    def forward(self, pred, target):
        pred = pred.reshape(pred.shape[0], -1, pred.shape[-1])  # b h w c -> b (h w) c
        target = target.reshape(target.shape[0], -1, target.shape[-1])
        corr = self.calculate_correlation(pred, target)
        if self.reduction == 'mean':
            loss = torch.mean(corr)
        elif self.reduction == 'sum':
            loss = torch.sum(corr)
        else:
            loss = corr
        return loss


class ScaledMaeLoss(nn.Module):
    def __init__(self, keep_channels=False):
        super().__init__()
        self.loss = nn.L1Loss(reduction='none')
        self.keep_channels = keep_channels

    @staticmethod
    def scale_each_min_max(state):
        state_scaled = rearrange(state, 'b h w c -> b c (h w)')
        state_scaled_min = torch.min(state_scaled, dim=2, keepdim=True)[0]
        state_scaled_max = torch.max(state_scaled, dim=2, keepdim=True)[0]
        state_scaled = (state_scaled - state_scaled_min) / (state_scaled_max - state_scaled_min)
        state_scaled = rearrange(state_scaled, 'b c (h w) -> b h w c', h=state.size(1), w=state.size(2))
        return state_scaled

    def forward(self, pred, target):
        # scale pred and target between 0 and 1
        pred = self.scale_each_min_max(pred)
        target = self.scale_each_min_max(target)
        loss = self.loss(pred, target)
        if self.keep_channels:
            loss = torch.mean(loss, dim=(0, 1, 2))
        else:
            loss = torch.mean(loss)

        return loss


# Adapted from: https://github.com/zongyi-li/fourier_neural_operator/
class LpLoss(nn.Module):
    def __init__(self, p=2, reduction="mean"):
        super().__init__()
        assert p > 0, f"Const p must be greater than 0, current value is {p}"

        self.p = p
        self.reduction = reduction

    def forward(self, pred, target):
        batch_size = pred.size()[0]

        diff_norms = torch.norm(pred.reshape(batch_size, -1) - target.reshape(batch_size, -1), self.p, 1)
        y_norms = torch.norm(target.reshape(batch_size, -1), self.p, 1)
        loss_norm = diff_norms / y_norms

        if self.reduction == 'mean':
            loss = torch.mean(loss_norm)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_norm)
        else:
            loss = loss_norm

        return loss


class UnnormalizedLoss(nn.Module):
    def __init__(self, loss, decoder=None, stats_shape=()):
        super().__init__()
        self.loss = loss
        zero_mean = torch.zeros(stats_shape)
        unit_std = torch.ones(stats_shape)
        self.norm_decoder = decoder if decoder is not None else GaussianNormalizeDecoder(zero_mean, unit_std)

    def set_norm_decoder(self, decoder):
        self.norm_decoder = decoder

    def forward(self, pred, target):
        if self.norm_decoder is not None:
            pred = self.norm_decoder(pred)
            target = self.norm_decoder(target)
        else:
            print("Decoder for normalization is not set!!!")

        loss = self.loss(pred, target)
        return loss
