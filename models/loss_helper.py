from models.pde_loss import SweFvLoss, DarcyLoss

try:
    from models.pde_loss_sim import SweSimulatorLoss
except ImportError:
    # use SweFvLoss instead of SweSimulatorLoss if it is not available
    class SweSimulatorLoss(SweFvLoss):
        def __init__(self, Tn=0.128, x_min=-2.5, x_max=2.5, n_ghosts=2, reduction='none', flip_xy=False):
            super(SweSimulatorLoss, self).__init__(Tn, x_min, x_max, n_ghosts, reduction, flip_xy)
            print("SWE FV loss is used instead of SweSimulatorLoss")


def get_pde_loss_function(system, flip_xy, Tn_mult=1.):
    print(f"PDE error: system = {system}")
    if system == "swe":
        Tn = 1.28 * Tn_mult
        pde_loss_f = SweFvLoss(Tn=Tn, flip_xy=flip_xy)
        pde_loss_sim_f = SweSimulatorLoss(Tn=Tn, flip_xy=flip_xy)
    elif system == "swe_per":
        Tn = 0.128 * Tn_mult
        x_min = -0.5
        x_max = 0.5
        pde_loss_f = SweFvLoss(Tn=Tn, x_min=x_min, x_max=x_max, flip_xy=flip_xy)
        pde_loss_sim_f = SweSimulatorLoss(Tn=Tn, x_min=x_min, x_max=x_max, flip_xy=flip_xy)
    elif system == "darcy":
        pde_loss_f = DarcyLoss(flip_xy=flip_xy)
        pde_loss_sim_f = DarcyLoss(flip_xy=flip_xy)
    elif system == "reactor":
        Tn = 1.28 * Tn_mult
        pde_loss_f = ReactorLoss(Tn=Tn, flip_xy=flip_xy)
        pde_loss_sim_f = ReactorLoss(Tn=Tn, flip_xy=flip_xy)
    else:
        # use swe system by default
        Tn = 1.28 * Tn_mult
        pde_loss_f = SweFvLoss(Tn=Tn, flip_xy=flip_xy)
        pde_loss_sim_f = SweSimulatorLoss(Tn=Tn, flip_xy=flip_xy)

    return pde_loss_f, pde_loss_sim_f
