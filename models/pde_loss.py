import torch
import torch.nn as nn
import torch.nn.functional as F


def flip_state(pred, gt, normalizer_h, normalizer_u):
    # the input and target were flipped in the model, so I have to flip them back so the pde loss is correct
    h_ch = len(normalizer_h.subtract) if len(normalizer_h.subtract.size()) > 0 else 1
    u_ch = len(normalizer_u.subtract) if len(normalizer_u.subtract.size()) > 0 else 1
    pred_h = pred[..., :h_ch]
    pred_u = pred[..., h_ch:u_ch + h_ch]
    pred = torch.cat([pred_u, pred_h], dim=-1)  # flip h and u back
    gt_h = gt[..., :h_ch]
    gt_u = gt[..., h_ch:u_ch + h_ch]
    gt = torch.cat([gt_u, gt_h], dim=-1)
    return pred, gt


class DarcyLoss(nn.Module):
    """
    PDE loss function for Darcy flow equation
    """
    def __init__(self, reduction='none', flip_xy=False):
        super(DarcyLoss, self).__init__()
        self.flip_xy = flip_xy
        self.criterion = nn.MSELoss(reduction=reduction)  # criterion is not used
        self.D = 1.0
        self.eps = 1e-8

    def calculate_loss(self, pred):
        batchsize = pred.shape[0]
        size = pred.shape[1]
        a = pred[..., 0]
        u = pred[..., 1]
        u = u.reshape(batchsize, size, size)
        a = a.reshape(batchsize, size, size)
        dx = self.D / size
        dy = dx

        # ux: (batch, size-2, size-2)
        ux = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
        uy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dy)

        a = a[:, 1:-1, 1:-1]
        aux = a * ux
        auy = a * uy
        auxx = (aux[:, 2:, 1:-1] - aux[:, :-2, 1:-1]) / (2 * dx)
        auyy = (auy[:, 1:-1, 2:] - auy[:, 1:-1, :-2]) / (2 * dy)
        Du = - (auxx + auyy)

        # the right hand side of the equation, beta = 1. in the train dataset
        pde_loss = (Du - 1.) ** 2

        return pde_loss

    def forward(self, pred, gt, normalizer_h, normalizer_u, return_d=False, calc_prob=False, clamp_loss=False):
        if self.flip_xy:
            pred, gt = flip_state(pred, gt, normalizer_h, normalizer_u)

        if return_d:
            with torch.inference_mode(False):
                pred = pred.clone()
                pred.requires_grad_(True)

                loss_matrix = self.calculate_loss(pred)

                if calc_prob:
                    # calculate log of probabilities out of the mse loss
                    loss_matrix = 2 * (1. - torch.sigmoid(1e5 * loss_matrix))
                    loss_matrix = torch.log(loss_matrix + 1e-12)

                loss = loss_matrix.mean()  # calculate the mean for backprop
                dloss = torch.autograd.grad(loss, pred)[0]

                # set nan values to zero
                dloss[torch.isnan(dloss)] = 0.0
                loss = dloss
        else:
            # divide loss by spacial locations (because the order of magnitude is quite big
            loss = self.calculate_loss(pred)
            b, t, n = loss.shape
            loss = loss / (t * n)
            if clamp_loss:
                loss = torch.clamp(loss, max=1.)

        return loss


class SweFvLoss(nn.Module):
    """
    PDE loss function for shallow water equations using finite volume method
    """
    def __init__(self, Tn=0.128, x_min=-2.5, x_max=2.5, n_ghosts=2, reduction='none', flip_xy=False):
        super(SweFvLoss, self).__init__()
        self.flip_xy = flip_xy
        self.criterion = nn.MSELoss(reduction=reduction)
        self.g = 1.0
        self.Tn = Tn
        self.x_min = x_min
        self.x_max = x_max
        self.n_ghosts = n_ghosts    # number of ghost cells for boundary conditions
        self.eps = 1e-8

    def gen_x(self, nx, s_t):
        step = (self.x_max - self.x_min) / nx

        # add ghost cells
        n_ghosts = self.n_ghosts
        nx += 2 * n_ghosts

        if nx % 2 == 0:
            # get the sequence mirrored around center
            # x = np.linspace(x_min + step / 2, x_max - step / 2, nx)
            x = torch.linspace(self.x_min + step / 2 - step * n_ghosts, self.x_max - step / 2 + step * n_ghosts, nx)
        else:
            # x = np.linspace(x_min, x_max, nx)
            x = torch.linspace(self.x_min - step * n_ghosts, self.x_max + step * n_ghosts, nx)
        x = x.type_as(s_t)
        return x

    def set_boundary(self, s_t):
        # boundary points stay the same as neighbors
        # set boundary conditions (ghost cells)
        n_ghosts = self.n_ghosts
        s_t_ext = F.pad(s_t, (0, 0, n_ghosts, n_ghosts), mode='replicate')

        h = s_t_ext[..., 0]
        hu = s_t_ext[..., 1] * s_t_ext[..., 0]
        return h, hu

    def f_t_swp1d(self, s_t, dt):
        """
        Transition function that predicts for all given timesteps simultaneously one step ahead.
        The prediction is done by finite volume method with is a bit different from the ground truth simulator

        :param s_t: state at time t, (b, t, s, 2)
        """
        n_ghosts = self.n_ghosts
        b, t, nx, c = s_t.shape

        x = self.gen_x(nx, s_t)
        dx = x[1] - x[0]

        # enforce boundary conditions
        h, hu = self.set_boundary(s_t)

        # use FORCE method
        # take a half time step estimating h and hu at the nx-1 spatial midpoints
        hm = 0.5 * (h[..., :-1] + h[..., 1:]) - 0.5 * dt * (hu[..., 1:] - hu[..., :-1]) / dx
        hum_upd = hu ** 2 / (h + self.eps) + 0.5 * self.g * h ** 2
        hum = 0.5 * (hu[..., :-1] + hu[..., 1:]) - 0.5 * dt * (hum_upd[..., 1:] - hum_upd[..., :-1]) / dx

        # take a full time step with the derivative at the half time step to estimate the solution at the nx-2 nodes
        # the sizes of h_next and hu_next are already reduced by 2 in the spatial dimension
        h_next = 0.5 * (hm[..., :-1] + hm[..., 1:]) - 0.5 * dt * (hum[..., 1:] - hum[..., :-1]) / dx

        hu_upd = hum ** 2 / (hm + self.eps) + 0.5 * self.g * hm ** 2
        hu_next = 0.5 * (hum[..., :-1] + hum[..., 1:]) - 0.5 * dt * (hu_upd[..., 1:] - hu_upd[..., :-1]) / dx

        # remove ghost cells
        h = h_next[..., n_ghosts-1:-n_ghosts+1]
        u = hu_next[..., n_ghosts-1:-n_ghosts+1] / (h + self.eps)

        s_next = torch.stack((h, u), dim=-1)
        return s_next

    def unroll_from_init(self, ic, n_steps):
        """
        Unroll the model from initial condition for n_steps
        :param ic: initial condition, (b, 1, w, 2)
        :param n_steps: number of steps to unroll
        :return: unrolled sequence, (b, n_steps, w, 2)
        """
        states = [ic]
        s_t = ic
        dt = self.Tn / n_steps
        for i in range(n_steps):
            s_t = self.f_t_swp1d(s_t, dt)
            states.append(s_t)

        states = torch.cat(states, dim=1)
        return states

    def unroll_loss(self, pred, gt, normalizer_h, normalizer_u, return_unroll=False):
        if self.flip_xy:
            pred, gt = flip_state(pred, gt, normalizer_h, normalizer_u)

        pred_ic = pred[:, 0:1]
        pred_unrolled = self.unroll_from_init(pred_ic, pred.shape[1] - 1)

        # division by scale is equivalent to the loss between normalized inputs
        scale = self.get_scaling(normalizer_h, normalizer_u)
        loss = (pred_unrolled - gt) ** 2 / scale

        if return_unroll:
            return loss, pred_unrolled
        return loss

    def get_scaling(self, normalizer_h, normalizer_u):
        # get the scaling of the variables
        scale_h = normalizer_h.divide
        scale_u = normalizer_u.divide

        if self.flip_xy:
            scale = torch.stack((scale_u, scale_h), dim=-1)
        else:
            scale = torch.stack((scale_h, scale_u), dim=-1)
        scale = scale ** 2
        return scale

    def calculate_loss(self, pred, gt, normalizer_h, normalizer_u):
        # pred_next = self.f_t_swp1d(pred, dt)
        # pde_loss_matrix = (pred_next[:, :-1] - pred[:, 1:]) ** 2
        # loss = pde_loss_matrix   # or calculate squared values

        n_times = pred.shape[1]
        dt = self.Tn / n_times
        pred_next = self.f_t_swp1d(pred, dt)
        pred_next_with_ic = torch.cat((pred[:, 0:1], pred_next[:, :-1]), dim=1)
        pred_next_with_ic[torch.isnan(pred_next_with_ic)] = 0.0
        # division by scale is equivalent to the loss between normalized inputs
        scale = self.get_scaling(normalizer_h, normalizer_u)

        loss = (pred_next_with_ic - gt) ** 2 / scale
        return loss

    def forward(self, pred, gt, normalizer_h, normalizer_u, return_d=False, calc_prob=False, clamp_loss=False):
        if self.flip_xy:
            pred, gt = flip_state(pred, gt, normalizer_h, normalizer_u)

        if return_d:
            with torch.inference_mode(False):
                pred = pred.clone()
                pred.requires_grad_(True)

                loss_matrix = self.calculate_loss(pred, gt, normalizer_h, normalizer_u)
                loss = loss_matrix.mean()  # calculate the mean for backprop
                dloss = torch.autograd.grad(loss, pred)[0]

                # set nan values to zero
                dloss[torch.isnan(dloss)] = 0.0
                loss = dloss
        else:
            loss = self.calculate_loss(pred, gt, normalizer_h, normalizer_u)
            if clamp_loss:
                loss = torch.clamp(loss, max=1.)

        return loss


class SweFvGtLoss(nn.Module):
    """
    PDE loss function for shallow water equations using finite volume method
    """
    def __init__(self, Tn=0.128, x_min=-2.5, x_max=2.5, n_ghosts=2, reduction='none', flip_xy=False):
        super(SweFvGtLoss, self).__init__()
        self.flip_xy = flip_xy
        self.criterion = nn.MSELoss(reduction=reduction)
        self.g = 1.0
        self.Tn = Tn
        self.x_min = x_min
        self.x_max = x_max
        self.n_ghosts = n_ghosts    # number of ghost cells for boundary conditions
        self.eps = 1e-8

    def gen_x(self, nx, s_t):
        step = (self.x_max - self.x_min) / nx

        # add ghost cells
        n_ghosts = self.n_ghosts
        nx += 2 * n_ghosts

        if nx % 2 == 0:
            x = torch.linspace(self.x_min + step / 2 - step * n_ghosts, self.x_max - step / 2 + step * n_ghosts, nx)
        else:
            x = torch.linspace(self.x_min - step * n_ghosts, self.x_max + step * n_ghosts, nx)
        x = x.type_as(s_t)
        return x

    def set_boundary(self, s_t):
        # boundary points stay the same as neighbors
        # set boundary conditions (ghost cells)
        n_ghosts = self.n_ghosts
        s_t_ext = F.pad(s_t, (0, 0, n_ghosts, n_ghosts), mode='replicate')

        h = s_t_ext[..., 0]
        hu = s_t_ext[..., 1] * s_t_ext[..., 0]
        return h, hu

    def f_t_swp1d(self, s_t, dt):
        """
        Transition function that predicts for all given timesteps simultaneously one step ahead.
        The prediction is done by finite volume method with is a bit different from the ground truth simulator

        :param s_t: state at time t, (b, t, s, 2)
        """
        n_ghosts = self.n_ghosts
        b, t, nx, c = s_t.shape

        x = self.gen_x(nx, s_t)
        dx = x[1] - x[0]

        # enforce boundary conditions
        h, hu = self.set_boundary(s_t)

        # use FORCE method
        # take a half time step estimating h and hu at the nx-1 spatial midpoints
        hm = 0.5 * (h[..., :-1] + h[..., 1:]) - 0.5 * dt * (hu[..., 1:] - hu[..., :-1]) / dx
        hum_upd = hu ** 2 / (h + self.eps) + 0.5 * self.g * h ** 2
        hum = 0.5 * (hu[..., :-1] + hu[..., 1:]) - 0.5 * dt * (hum_upd[..., 1:] - hum_upd[..., :-1]) / dx

        # take a full time step with the derivative at the half time step to estimate the solution at the nx-2 nodes
        # the sizes of h_next and hu_next are already reduced by 2 in the spatial dimension
        h_next = 0.5 * (hm[..., :-1] + hm[..., 1:]) - 0.5 * dt * (hum[..., 1:] - hum[..., :-1]) / dx

        hu_upd = hum ** 2 / (hm + self.eps) + 0.5 * self.g * hm ** 2
        hu_next = 0.5 * (hum[..., :-1] + hum[..., 1:]) - 0.5 * dt * (hu_upd[..., 1:] - hu_upd[..., :-1]) / dx

        # remove ghost cells
        h = h_next[..., n_ghosts-1:-n_ghosts+1]
        u = hu_next[..., n_ghosts-1:-n_ghosts+1] / (h + self.eps)

        s_next = torch.stack((h, u), dim=-1)
        return s_next

    def unroll_from_init(self, ic, n_steps):
        """
        Unroll the model from initial condition for n_steps
        :param ic: initial condition, (b, 1, w, 2)
        :param n_steps: number of steps to unroll
        :return: unrolled sequence, (b, n_steps, w, 2)
        """
        states = [ic]
        s_t = ic
        dt = self.Tn / n_steps
        for i in range(n_steps):
            s_t = self.f_t_swp1d(s_t, dt)
            states.append(s_t)

        states = torch.cat(states, dim=1)
        return states

    def unroll_loss(self, pred, gt, normalizer_h, normalizer_u, return_unroll=False):
        if self.flip_xy:
            pred, gt = flip_state(pred, gt, normalizer_h, normalizer_u)

        pred_ic = pred[:, 0:1]
        pred_unrolled = self.unroll_from_init(pred_ic, pred.shape[1] - 1)

        # division by scale is equivalent to the loss between normalized inputs
        scale = self.get_scaling(normalizer_h, normalizer_u)
        loss = (pred_unrolled - gt) ** 2 / scale

        if return_unroll:
            return loss, pred_unrolled
        return loss

    def get_scaling(self, normalizer_h, normalizer_u):
        # get the scaling of the variables
        scale_h = normalizer_h.divide
        scale_u = normalizer_u.divide

        if self.flip_xy:
            scale = torch.stack([scale_u, scale_h], dim=-1)
        else:
            scale = torch.stack([scale_h, scale_u], dim=-1)
        scale = scale ** 2
        return scale

    def calculate_loss(self, pred, gt, normalizer_h, normalizer_u):
        n_times = pred.shape[1]
        dt = self.Tn / n_times
        pred_next = self.f_t_swp1d(pred, dt)
        pred_next_with_ic = torch.cat((pred[:, 0:1], pred_next[:, :-1]), dim=1)
        pred_next_with_ic[torch.isnan(pred_next_with_ic)] = 0.0

        # division by scale is equivalent to the loss between normalized inputs
        scale = self.get_scaling(normalizer_h, normalizer_u)
        loss = (pred_next_with_ic - gt) ** 2 / scale
        return loss

    def forward(self, pred, gt, normalizer_h, normalizer_u, return_d=False, calc_prob=False, clamp_loss=False):
        if self.flip_xy:
            pred, gt = flip_state(pred, gt, normalizer_h, normalizer_u)

        if return_d:
            with torch.inference_mode(False):
                pred = pred.clone()
                pred.requires_grad_(True)

                loss_matrix = self.calculate_loss(pred, gt, normalizer_h, normalizer_u)

                if calc_prob:
                    # calculate log of probabilities out of the mse loss
                    loss_matrix = 2 * (1. - torch.sigmoid(1e5 * loss_matrix))
                    loss_matrix = torch.log(loss_matrix + 1e-12)

                loss = loss_matrix.mean()  # calculate the mean for backprop
                dloss = torch.autograd.grad(loss, pred)[0]

                # set nan values to zero
                dloss[torch.isnan(dloss)] = 0.0
                loss = dloss
        else:
            loss = self.calculate_loss(pred, gt, normalizer_h, normalizer_u)
            if clamp_loss:
                loss = torch.clamp(loss, max=1.)

        return loss
