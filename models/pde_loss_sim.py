import numpy as np
import torch
import torch.nn as nn

from generate.src.sim_dam_break_1d import SwPerturbation1D
from models.pde_loss import SweFvGtLoss


class SweSimulatorLoss(SweFvGtLoss):
    """
    PDE loss function for shallow water equations using PyClaw simulator (used to generate the data)
    """
    def __init__(self, Tn=0.128, x_min=-2.5, x_max=2.5, reduction='none', flip_xy=False):
        super(SweSimulatorLoss, self).__init__(Tn=Tn, x_min=x_min, x_max=x_max, flip_xy=flip_xy)
        self.criterion = nn.MSELoss(reduction=reduction)
        self.g = 1.0
        self.Tn = Tn
        self.x_min = x_min
        self.x_max = x_max
        self.eps = 1e-8
        self.xdim = 128
        self.scenario = SwPerturbation1D(self.xdim)

    def unroll_from_init(self, ic, n_steps):
        """
        Unroll the model from initial condition for n_steps
        :param ic: initial condition, (b, 1, w, 2)
        :param n_steps: number of steps to unroll
        :return: unrolled sequence, (b, n_steps, w, 2)
        """
        states = []
        n_batch = ic.shape[0]
        dt = self.Tn / n_steps
        for b in range(n_batch):
            states_batch = [ic[b, 0]]
            h = ic[b, 0, :, 0]
            hu = ic[b, 0, :, 1] * h
            for i in range(n_steps):
                h_next, hu_next = self.scenario.simulate_step(h, hu, dt)
                pred_next = np.stack((h_next, hu_next / (h_next + self.eps)), axis=-1)
                states_batch.append(pred_next)
                h = h_next
                hu = hu_next

            states.append(np.stack(states_batch, axis=0))

        states = np.stack(states, axis=0)
        return states

    def unroll_loss(self, pred, gt, normalizer_h, normalizer_u, return_unroll=False):
        pred = pred.cpu().numpy()
        gt = gt.cpu().numpy()
        pred_ic = pred[:, 0:1]
        pred_unrolled = self.unroll_from_init(pred_ic, pred.shape[1] - 1)

        # division by scale is equivalent to the loss between normalized inputs
        scale = self.get_scaling(normalizer_h, normalizer_u).cpu().numpy()
        loss = (pred_unrolled - gt) ** 2 / scale
        loss = torch.tensor(loss).float()

        if return_unroll:
            pred_unrolled = torch.tensor(pred_unrolled).float()
            return loss, pred_unrolled
        return loss

    def calculate_loss(self, pred, gt, normalizer_h, normalizer_u):
        pred_next_with_ic = []
        n_batch = pred.shape[0]
        n_times = pred.shape[1]
        dt = self.Tn / n_times
        for b in range(n_batch):
            pred_ic_batch = [pred[b, 0]]
            for i in range(n_times - 1):
                h = pred[b, i, :, 0]
                hu = pred[b, i, :, 1] * h
                h_next, hu_next = self.scenario.simulate_step(h, hu, dt)
                pred_next = np.stack((h_next, hu_next / (h_next + self.eps)), axis=-1)
                pred_ic_batch.append(pred_next)
            pred_next_with_ic.append(np.stack(pred_ic_batch, axis=0))

        pred_next_with_ic = np.stack(pred_next_with_ic, axis=0)
        pred_next_with_ic[np.isnan(pred_next_with_ic)] = 0.0

        # division by scale is equivalent to the loss between normalized inputs
        scale = self.get_scaling(normalizer_h, normalizer_u).cpu().numpy()
        loss = (pred_next_with_ic - gt) ** 2 / scale
        return loss

    def forward(self, pred, gt, normalizer_h, normalizer_u, return_d=False, calc_prob=False, clamp_loss=False):
        if self.flip_xy:
            # the input and target were flipped in the model, so I have to flip them back so the pde loss is correct
            h_ch = len(normalizer_h.subtract) if len(normalizer_h.subtract.size()) > 0 else 1
            u_ch = len(normalizer_u.subtract) if len(normalizer_u.subtract.size()) > 0 else 1

            pred_h = pred[..., :h_ch]
            pred_u = pred[..., h_ch:u_ch+h_ch]
            pred = torch.cat([pred_u, pred_h], dim=-1)  # flip h and u back

            gt_h = gt[..., :h_ch]
            gt_u = gt[..., h_ch:u_ch+h_ch]
            gt = torch.cat([gt_u, gt_h], dim=-1)

        if return_d:
            print(f"The ground truth simulator is non-differentiable.")
            return 0.0
        else:
            pred = pred.to('cpu').numpy()
            gt = gt.to('cpu').numpy()
            loss = self.calculate_loss(pred, gt, normalizer_h, normalizer_u)
            loss = torch.tensor(loss)
            if clamp_loss:
                loss = torch.clamp(loss, max=1.)

        return loss
