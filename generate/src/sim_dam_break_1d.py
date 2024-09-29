"""
This file is adapted from the following file
https://github.com/pdebench/PDEBench/blob/main/pdebench/data_gen/src/sim_radial_dam_break.py
"""
import copy
from abc import abstractmethod
from abc import ABC

import os
import sys
import time

import h5py
import numpy as np
import torch
from clawpack import riemann
from clawpack import pyclaw

sys.path.append(os.path.join(sys.path[0], "..", "utils"))
sys.path.append(os.path.join(sys.path[0], "..", "riemann_solvers"))
# import src.riemann_solvers as riemann_solvers


class Basic1DScenario(ABC):
    name = ""

    def __init__(self):
        self.solver = None
        self.claw_state = None
        self.domain = None
        self.solution = None
        self.claw = None
        self.save_state = {}
        self.state_getters = {}

        self.setup_solver()
        self.create_domain()
        self.set_boundary_conditions()
        self.set_initial_conditions()
        self.register_state_getters()
        self.outdir = os.sep.join(["./", self.name.replace(" ", "") + "1D"])

    @abstractmethod
    def setup_solver(self):
        pass

    @abstractmethod
    def create_domain(self):
        pass

    @abstractmethod
    def set_initial_conditions(self):
        pass

    @abstractmethod
    def set_boundary_conditions(self):
        pass

    def __get_h(self):
        return self.claw_state.q[self.depthId, :].tolist()

    def __get_u(self):
        return (
                self.claw_state.q[self.momentumId_x, :] / self.claw_state.q[self.depthId, :]
        ).tolist()

    def __get_hu(self):
        return self.claw_state.q[self.momentumId_x, :].tolist()

    def register_state_getters(self):
        self.state_getters = {
            "h": self.__get_h,
            "u": self.__get_u,
            "hu": self.__get_hu,
        }

    def add_save_state(self):
        for key, getter in self.state_getters.items():
            self.save_state[key].append(getter())

    def init_save_state(self, T, tsteps):
        self.save_state = {}
        self.save_state["x"] = self.domain.grid.x.centers.tolist()
        self.save_state["t"] = np.linspace(0.0, T, tsteps + 1).tolist()
        for key, getter in self.state_getters.items():
            self.save_state[key] = [getter()]

    def save_state_to_disk(self, data_f, seed_str):
        T = np.asarray(self.save_state["t"])
        X = np.asarray(self.save_state["x"])
        H = np.expand_dims(np.asarray(self.save_state["h"]), -1)
        u = np.expand_dims(np.asarray(self.save_state["u"]), -1)

        data_f.create_dataset(f"{seed_str}/data/input", data=H, dtype="f")
        data_f.create_dataset(f"{seed_str}/data/target", data=u, dtype="f")

        data_f.create_dataset(f"{seed_str}/grid/x", data=X, dtype="f")
        data_f.create_dataset(f"{seed_str}/grid/t", data=T, dtype="f")

    def simulate(self, t):
        if all(v is not None for v in [self.domain, self.claw_state, self.solver]):
            self.solver.evolve_to_time(self.solution, t)
        else:
            print("Simulate failed: No scenario defined.")

    def run(self, T=1.0, tsteps=20):
        self.init_save_state(T, tsteps)
        self.solution = pyclaw.Solution(self.claw_state, self.domain)
        dt = T / tsteps
        for tstep in range(1, tsteps + 1):
            t = tstep * dt
            self.simulate(t)
            self.add_save_state()


class RadialDamBreak1D(Basic1DScenario):
    name = "RadialDamBreak1D"

    def __init__(self, xdim, grav=1.0, dam_radius=0.5, inner_height=3.0, x0=0.0, u_inner=0.0, u_outer=0.0):
        self.depthId = 0
        self.momentumId_x = 1
        self.grav = grav
        self.xdim = xdim
        self.dam_radius = dam_radius
        self.inner_height = inner_height
        self.x0 = x0
        self.u_inner = u_inner
        self.u_outer = u_outer
        super().__init__()
        # self.state_getters['bathymetry'] = self.__get_bathymetry

    def setup_solver(self):
        rs = riemann.shallow_roe_with_efix_1D
        self.solver = pyclaw.ClawSolver1D(rs)
        self.solver.limiters = pyclaw.limiters.tvd.vanleer
        # self.solver.fwave = True
        # the constants can be imported from Riemann:
        # from clawpack.riemann.shallow_roe_with_efix_1D_constants import depth, momentum, num_eqn
        self.solver.num_waves = 2
        self.solver.num_eqn = 2
        self.depthId = 0
        self.momentumId_x = 1

    def create_domain(self):
        xlower = -2.5
        xupper = 2.5
        mx = self.xdim
        x = pyclaw.Dimension(xlower, xupper, mx, name="x")
        self.domain = pyclaw.Domain(x)
        self.claw_state = pyclaw.State(self.domain, self.solver.num_eqn)

    def set_boundary_conditions(self):
        """
        Sets homogeneous Neumann boundary conditions at each end for q=(u, h*u)
        and for the bathymetry (auxiliary variable).
        """
        self.solver.bc_lower[0] = pyclaw.BC.extrap
        self.solver.bc_upper[0] = pyclaw.BC.extrap

    @staticmethod
    def initial_h(coords):
        x = coords[:, 0]
        x0 = self.x0
        r = np.abs(x - x0)

        h_in = self.inner_height
        h_out = 1.0

        # from this python example:
        # https://github.com/clawpack/pyclaw/blob/master/examples/shallow_1d/dam_break.py
        # state.q[depth, :] = h_in * (x <= x0) + h_out * (x > x0)
        # state.q[momentum, :] = h_in * ul * (x <= x0) + h_out * ur * (x > x0)
        return h_in * (r <= self.dam_radius) + h_out * (r > self.dam_radius)

    @staticmethod
    def initial_momentum_x(coords):
        x = coords[:, 0]
        x0 = self.x0
        r = np.abs(x - x0)

        h_in = self.inner_height
        h_out = 1.0
        u_in = self.u_inner
        u_out = self.u_outer
        hu_init = h_in * u_in * (r <= self.dam_radius) + h_out * u_out * (r > self.dam_radius)
        return hu_init

    def __get_bathymetry(self):
        return self.claw_state.aux[0, :].tolist()

    def set_initial_conditions(self):
        self.claw_state.problem_data["grav"] = self.grav

        x0 = self.x0
        X = self.claw_state.p_centers[0]
        r = np.abs(X - x0)
        h_in = self.inner_height
        h_out = 1.0
        u_in = self.u_inner
        u_out = self.u_outer

        self.claw_state.q[self.depthId, :] = h_in * (r <= self.dam_radius) + h_out * (r > self.dam_radius)
        self.claw_state.q[self.momentumId_x, :] = h_in * u_in * (r <= self.dam_radius) \
                                                  + h_out * u_out * (r > self.dam_radius)

    def save_state_to_disk(self, data_f, seed_str):
        super().save_state_to_disk(data_f, seed_str)

        # save constants
        dam_radius = np.expand_dims(np.asarray(self.dam_radius), -1)
        inner_h = np.expand_dims(np.asarray(self.inner_height), -1)
        u_inner = np.expand_dims(np.asarray(self.u_inner), -1)
        u_outer = np.expand_dims(np.asarray(self.u_outer), -1)
        x0 = np.expand_dims(np.asarray(self.x0), -1)

        data_f.create_dataset(f"{seed_str}/const/dam_radius", data=dam_radius, dtype="f")
        data_f.create_dataset(f"{seed_str}/const/inner_h", data=inner_h, dtype="f")
        data_f.create_dataset(f"{seed_str}/const/u_inner", data=u_inner, dtype="f")
        data_f.create_dataset(f"{seed_str}/const/u_outer", data=u_outer, dtype="f")
        data_f.create_dataset(f"{seed_str}/const/x0", data=x0, dtype="f")


class SwPerturbation1D(Basic1DScenario):
    name = "SwPerturbation1D"

    def __init__(self, xdim, grav=1.0, init_stimulus=0.1, sigma=0.5, inner_height=1.0, x0=0.0, init_u=0.0):
        self.depthId = 0
        self.momentumId_x = 1
        self.grav = grav
        self.xdim = xdim
        self.init_stimulus = init_stimulus
        self.sigma = sigma
        self.inner_height = inner_height
        self.x0 = x0
        self.init_u = init_u

        # I can set up bathemetry and Gaussian distribution for the initial condition of the momentum if needed
        # similar to this example: http://depts.washington.edu/clawpack/sampledocs/v570_gallery/pyclaw/gallery/sill.html
        super().__init__()
        # self.state_getters['bathymetry'] = self.__get_bathymetry

    def set_rk_params(self):
        # RK family, RK4 is default
        # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        s = 4
        a = np.zeros((s, s), dtype=np.float64)
        a[1, 0] = 0.5
        a[2, 1] = 0.5
        a[3, 2] = 1.0
        self.solver.a = a  # matrix of RK coefficients
        self.solver.b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6], dtype=np.float64)  # array of RK weights
        self.solver.c = np.array([0., 0.5, 0.5, 1.], dtype=np.float64)  # array of RK nodes

        self.solver.cfl_max = 0.25  ## ???
        self.solver.cfl_desired = 0.24

    def setup_solver(self):
        # solver repo: https://github.com/clawpack/pyclaw/blob/master/src/pyclaw/classic/solver.py
        kernel_language = "Fortran"  # "Python"  # "Fortran"
        print("Kernel language: ", kernel_language)
        if kernel_language == "Fortran":
            rs = riemann.shallow_roe_with_efix_1D
            # rs = riemann.shallow_1D_py.shallow_fwave_1d
            # rs = riemann.shallow_hlle_1D
        else:
            # rs = riemann.shallow_1D_py.shallow_fwave_1d  # need to set aux or fwave = True -?
            # rs = riemann.shallow_1D_py.shallow_hll_1D
            # rs = riemann_solvers.shallow_hll_1D  # local code for the solver
            # rs = riemann_solvers.shallow_roe_1D
            rs = riemann_solvers.shallow_exact_1D

        solver_type = 'claw'
        if solver_type == 'sharpclaw':
            print("Use Sharp Claw as a solver")
            # this solver is implemented only for 1D case
            self.solver = pyclaw.SharpClawSolver1D(rs)
            # self.solver.time_integrator = 'RK'
            # self.set_rk_params()
        else:
            self.solver = pyclaw.ClawSolver1D(rs)
            self.solver.limiters = pyclaw.limiters.tvd.vanleer

        self.solver.kernel_language = kernel_language

        # self.solver.fwave = True
        self.solver.num_waves = 2
        self.solver.num_eqn = 2
        self.depthId = 0
        self.momentumId_x = 1

    def create_domain(self):
        xlower = -2.5
        xupper = 2.5
        mx = self.xdim
        x = pyclaw.Dimension(xlower, xupper, mx, name="x")
        self.domain = pyclaw.Domain(x)
        self.claw_state = pyclaw.State(self.domain, self.solver.num_eqn)

    def set_boundary_conditions(self):
        """
        Sets homogeneous Neumann boundary conditions at each end for q=(u, h*u)
        and for the bathymetry (auxiliary variable).
        """
        self.solver.bc_lower[0] = pyclaw.BC.extrap
        self.solver.bc_upper[0] = pyclaw.BC.extrap

    @staticmethod
    def initial_h(coords):
        x = coords[:, 0]
        x0 = self.x0
        h_in = self.inner_height
        eps = self.init_stimulus
        sigma = self.sigma

        # from this python example:
        # https://github.com/clawpack/pyclaw/blob/master/examples/shallow_1d/dam_break.py
        return h_in + eps * np.exp(-0.5 * (x - x0) ** 2 / sigma ** 2)

    @staticmethod
    def initial_momentum_x(coords):
        return torch.tensor(self.init_u, dtype=torch.float32)

    def __get_bathymetry(self):
        return self.claw_state.aux[0, :].tolist()

    def set_initial_conditions(self):
        self.set_problem_data(self.claw_state)

        # print("Grav: ")
        # print(self.claw_state.problem_data["grav"])
        #
        # print("Ghosts: ")
        # print("self.solver.num_ghost: ", self.solver.num_ghost)

        X = self.claw_state.p_centers[0]

        x0 = self.x0
        h_in = self.inner_height
        eps = self.init_stimulus
        sigma = self.sigma

        self.claw_state.q[self.depthId, :] = h_in + eps * np.exp(-0.5 * (X - x0) ** 2 / sigma ** 2)
        self.claw_state.q[self.momentumId_x, :] = self.init_u

    def set_problem_data(self, state):
        state.problem_data["grav"] = self.grav
        state.problem_data["dry_tolerance"] = 1e-3
        state.problem_data["sea_level"] = 0.0

    def save_state_to_disk(self, data_f, seed_str):
        super().save_state_to_disk(data_f, seed_str)

        # save constants
        eps = np.expand_dims(np.asarray(self.init_stimulus), -1)
        inner_h = np.expand_dims(np.asarray(self.inner_height), -1)
        x0 = np.expand_dims(np.asarray(self.x0), -1)
        init_u = np.expand_dims(np.asarray(self.init_u), -1)
        sigma = np.expand_dims(np.asarray(self.sigma), -1)

        data_f.create_dataset(f"{seed_str}/const/init_stimulus", data=eps, dtype="f")
        data_f.create_dataset(f"{seed_str}/const/inner_h", data=inner_h, dtype="f")
        data_f.create_dataset(f"{seed_str}/const/x0", data=x0, dtype="f")
        data_f.create_dataset(f"{seed_str}/const/init_u", data=init_u, dtype="f")
        data_f.create_dataset(f"{seed_str}/const/sigma", data=sigma, dtype="f")

        # remove solver when it is not needed anymore, otherwise some of the Fortran objects are not deleted
        # needed in case SharpClaw solver is used
        del self.solver

    def simulate_step(self, h, hu, dt):
        state = pyclaw.State(self.domain, self.solver.num_eqn)
        self.set_problem_data(state)
        state.q[self.depthId, :] = h
        state.q[self.momentumId_x, :] = hu
        solution = pyclaw.Solution(state, self.domain)

        # st = copy.deepcopy(self.solution.state.q)
        self.solver.evolve_to_time(solution, dt)

        h_next = solution.state.q[self.depthId, :]
        hu_next = solution.state.q[self.momentumId_x, :]
        # 'State diff is ', np.sum(np.abs(self.solution.state.q - st)))
        return h_next, hu_next


class SwPeriodic1D(Basic1DScenario):
    name = "SwPeriodic1D"

    def __init__(self, xdim, lambdas, gammas, grav=1.0, init_u=0.0):
        self.depthId = 0
        self.momentumId_x = 1
        self.grav = grav
        self.xdim = xdim
        self.lambdas = lambdas
        self.gammas = gammas
        self.init_u = init_u

        # I can set up bathemetry and Gaussian distribution for the initial condition of the momentum if needed
        # similar to this example: http://depts.washington.edu/clawpack/sampledocs/v570_gallery/pyclaw/gallery/sill.html
        super().__init__()
        # self.state_getters['bathymetry'] = self.__get_bathymetry

    def set_rk_params(self):
        # RK family, RK4 is default
        # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        s = 4
        a = np.zeros((s, s), dtype=np.float64)
        a[1, 0] = 0.5
        a[2, 1] = 0.5
        a[3, 2] = 1.0
        self.solver.a = a  # matrix of RK coefficients
        self.solver.b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6], dtype=np.float64)  # array of RK weights
        self.solver.c = np.array([0., 0.5, 0.5, 1.], dtype=np.float64)  # array of RK nodes

        self.solver.cfl_max = 0.25  ## ???
        self.solver.cfl_desired = 0.24

    def setup_solver(self):
        # solver repo: https://github.com/clawpack/pyclaw/blob/master/src/pyclaw/classic/solver.py
        kernel_language = "Fortran"  # "Python"  # "Fortran"
        print("Kernel language: ", kernel_language)
        if kernel_language == "Fortran":
            rs = riemann.shallow_roe_with_efix_1D
            # rs = riemann.shallow_1D_py.shallow_fwave_1d
            # rs = riemann.shallow_hlle_1D
        else:
            # rs = riemann.shallow_1D_py.shallow_fwave_1d  # need to set aux or fwave = True -?
            # rs = riemann.shallow_1D_py.shallow_hll_1D
            # rs = riemann_solvers.shallow_hll_1D  # local code for the solver
            # rs = riemann_solvers.shallow_roe_1D
            rs = riemann_solvers.shallow_exact_1D

        solver_type = 'claw'
        if solver_type == 'sharpclaw':
            print("Use Sharp Claw as a solver")
            # this solver is implemented only for 1D case
            self.solver = pyclaw.SharpClawSolver1D(rs)
            # self.solver.time_integrator = 'RK'
            # self.set_rk_params()
        else:
            self.solver = pyclaw.ClawSolver1D(rs)
            self.solver.limiters = pyclaw.limiters.tvd.vanleer

        self.solver.kernel_language = kernel_language

        # self.solver.fwave = True
        self.solver.num_waves = 2
        self.solver.num_eqn = 2
        self.depthId = 0
        self.momentumId_x = 1

    def create_domain(self):
        xlower = -0.5
        xupper = 0.5
        mx = self.xdim
        x = pyclaw.Dimension(xlower, xupper, mx, name="x")
        self.domain = pyclaw.Domain(x)
        self.claw_state = pyclaw.State(self.domain, self.solver.num_eqn)

    def set_boundary_conditions(self):
        """
        Sets homogeneous Neumann boundary conditions at each end for q=(u, h*u)
        and for the bathymetry (auxiliary variable).
        """
        self.solver.bc_lower[0] = pyclaw.BC.extrap
        self.solver.bc_upper[0] = pyclaw.BC.extrap

    @staticmethod
    def initial_h(coords, lambdas, gammas):
        x = coords[:, 0]

        return SwPeriodic1D.calc_init_h(x, lambdas, gammas)

    @staticmethod
    def calc_init_h(x, lambdas, gammas):
        # set up the initial conditions as a superposition of sin and cos
        n = np.min([len(lambdas), len(gammas)])
        n2 = n // 2  # symmetric around zero + 1
        # n=7, n2 =3, k is from -3 to 3
        h_hat_comb = [lambdas[i] * np.cos(2 * np.pi * (i - n2) * x) + gammas[i] * np.sin(2 * np.pi * (i - n2) * x)
                      for i in range(n)]
        h_hat = np.sum(np.array(h_hat_comb), axis=0)
        h_init = 1. + (h_hat - np.min(h_hat)) / (np.max(h_hat) - np.min(h_hat))
        return h_init

    @staticmethod
    def initial_momentum_x(coords, init_u):
        return torch.tensor(init_u, dtype=torch.float32)

    def __get_bathymetry(self):
        return self.claw_state.aux[0, :].tolist()

    def set_initial_conditions(self):
        self.set_problem_data(self.claw_state)

        X = self.claw_state.p_centers[0]
        init_h = SwPeriodic1D.calc_init_h(X, self.lambdas, self.gammas)

        self.claw_state.q[self.depthId, :] = init_h
        self.claw_state.q[self.momentumId_x, :] = self.init_u

    def set_problem_data(self, state):
        state.problem_data["grav"] = self.grav
        state.problem_data["dry_tolerance"] = 1e-3
        state.problem_data["sea_level"] = 0.0

    def save_state_to_disk(self, data_f, seed_str):
        super().save_state_to_disk(data_f, seed_str)

        # save constants
        data_f.create_dataset(f"{seed_str}/const/lambdas", data=self.lambdas, dtype="f")
        data_f.create_dataset(f"{seed_str}/const/gammas", data=self.gammas, dtype="f")
        data_f.create_dataset(f"{seed_str}/const/init_u", data=self.init_u, dtype="f")

        # remove solver when it is not needed anymore, otherwise some of the Fortran objects are not deleted
        # needed in case SharpClaw solver is used
        del self.solver

    def simulate_step(self, h, hu, dt):
        state = pyclaw.State(self.domain, self.solver.num_eqn)
        self.set_problem_data(state)
        state.q[self.depthId, :] = h
        state.q[self.momentumId_x, :] = hu
        solution = pyclaw.Solution(state, self.domain)

        # st = copy.deepcopy(self.solution.state.q)
        self.solver.evolve_to_time(solution, dt)

        h_next = solution.state.q[self.depthId, :]
        hu_next = solution.state.q[self.momentumId_x, :]
        # 'State diff is ', np.sum(np.abs(self.solution.state.q - st)))
        return h_next, hu_next


if __name__ == "__main__":
    # scenario = RadialDamBreak1D(xdim=128)
    scenario = SwPerturbation1D(xdim=128)
    scenario.run(T=2, tsteps=100)
