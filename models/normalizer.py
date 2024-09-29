import torch
import torch.nn as nn


class Normalizer(nn.Module):
    def __init__(self, subtract=None, divide=None, stats_shape=()):
        super().__init__()
        if subtract is None:
            subtract = torch.zeros(stats_shape)
        if divide is None:
            divide = torch.ones(stats_shape)

        if not torch.is_tensor(subtract):
            subtract = torch.tensor(subtract)
        if not torch.is_tensor(divide):
            divide = torch.tensor(divide)

        self.register_buffer('subtract', subtract)
        self.register_buffer('divide', divide)

    def set_stats(self, subtract, divide):
        self.subtract = subtract
        self.divide = divide

    def forward(self, x, inverse=False):
        if inverse:
            return x * self.divide + self.subtract
        else:
            return (x - self.subtract) / self.divide


class GaussianNormalizeDecoder(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        if not torch.is_tensor(mean):
            mean = torch.tensor(mean)
        if not torch.is_tensor(std):
            std = torch.tensor(std)

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        # unnormalize data
        return x * self.std + self.mean


def set_target_unnorm(plmodel, norm_target, target_mean, target_std):
    decoder = GaussianNormalizeDecoder(target_mean, target_std)
    if norm_target:
        # if targets are normalized by data loader, set the decoder for the loss function
        plmodel.mae_un_criterion.set_norm_decoder(decoder)
    else:
        # if targets are unnormalized from dataloader, we unnormalize model outputs
        plmodel.set_norm_decoder(decoder)
        plmodel.mae_un_criterion = plmodel.mae_criterion


def set_input_unnorm(plmodel, norm_input, input_mean, input_std):
    decoder = GaussianNormalizeDecoder(input_mean, input_std)
    if norm_input:
        # if inputs are normalized by data loader, set the decoder for the loss function
        plmodel.mae_un_inp_criterion.set_norm_decoder(decoder)
    else:
        plmodel.mae_un_inp_criterion = plmodel.mae_criterion
