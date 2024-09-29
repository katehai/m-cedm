"""
The file is adopted from
https://github.com/pdebench/PDEBench/blob/main/pdebench/data_gen/gen_radial_dam_break.py
"""

from copy import deepcopy
import os

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
# this allows us to keep defaults local to the machine
# e.g. HPC versus local laptop
import dotenv

dotenv.load_dotenv()

# or if the environment variables will be fixed for all executions, we can hard-code the environment variables like this:
num_threads = "4"

os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads
os.environ["NUMEXPR_MAX_THREADS"] = num_threads

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import h5py
import logging
import multiprocessing as mp
from itertools import repeat
from src import utils
import numpy as np
import time
from src.sim_dam_break_1d import RadialDamBreak1D, SwPerturbation1D

log = logging.getLogger(__name__)


def simulator(base_config, i):
    config = deepcopy(base_config)
    config.sim.seed = i
    log.info(f"Starting seed {i}")

    np.random.seed(config.sim.seed)

    ## Dam Break scenario in 1D
    # config.sim.inner_height = np.random.uniform(1.2, 2.2)
    # config.sim.dam_radius = np.random.uniform(0.25, 0.85)
    # # config.sim.u_inner = np.random.uniform(0.0, 1.5) # -> causes NaNs for some seeds
    # # config.sim.u_outer = -config.sim.u_inner
    #
    # scenario = RadialDamBreak1D(
    #     grav=config.sim.gravity,
    #     dam_radius=config.sim.dam_radius,
    #     inner_height=config.sim.inner_height,
    #     # u_inner=config.sim.u_inner,
    #     # u_outer=config.sim.u_outer,
    #     xdim=config.sim.xdim
    # )

    config.sim.inner_height = np.random.uniform(1.2, 5.2)  # h is strictly positive, default and u0 datasets
    # config.sim.inner_height = 3.  # ih dataset
    config.sim.init_stimulus = np.random.uniform(0.05, 1.)
    config.sim.x0 = np.random.uniform(-1, 1)
    config.sim.init_u = np.random.uniform(-2.2, 2.2)  # default dataset
    # config.sim.init_u = np.random.uniform(1., 5.4)  # u0_dataset
    # config.sim.init_u = np.random.uniform(1., 4.)
    config.sim.sigma = np.random.uniform(0.2, 2.)

    scenario = SwPerturbation1D(
        grav=config.sim.gravity,
        init_stimulus=config.sim.init_stimulus,
        inner_height=config.sim.inner_height,
        x0=config.sim.x0,
        init_u=config.sim.init_u,
        sigma=config.sim.sigma,
        xdim=config.sim.xdim
    )

    start_time = time.time()
    scenario.run(T=config.sim.T_end, tsteps=config.sim.n_time_steps)
    duration = time.time() - start_time
    seed_str = str(i).zfill(4)
    log.info(f"Seed {seed_str} took {duration} to finish")

    while True:
        try:
            print(f"Path is {utils.expand_path(config.output_path)}")
            with h5py.File(utils.expand_path(config.output_path), "a") as h5_file:
                scenario.save_state_to_disk(h5_file, seed_str)
                h5_file.attrs["config"] = OmegaConf.to_yaml(config)
        except IOError:
            time.sleep(0.1)
            continue
        else:
            break


@hydra.main(config_path="configs/", config_name="sw_perturb_1d", version_base="1.1")
def main(config: DictConfig):
    """
    use config specifications to generate dataset
    """

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934

    # Change to original working directory to import modules
    import os

    temp_path = os.getcwd()
    os.chdir(get_original_cwd())

    # Change back to the hydra working directory
    os.chdir(temp_path)

    work_path = os.path.dirname(config.work_dir)
    output_path = os.path.join(work_path, config.data_dir, config.output_path)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    mode = "_test" if config.test else "_train"

    if config.test:
        # test set
        num_samples_init = 1000
        num_samples_final = 1100

        seed = np.arange(num_samples_init, num_samples_final)
        seed = seed.tolist()
    else:
        # train set
        num_samples_init = 0
        num_samples_final = 1000

        # if one wants to generate more than 1000 training examples
        num_samples_init1 = 1100
        num_samples_final1 = 1100  # 2000

        seed = np.arange(num_samples_init, num_samples_final)
        if num_samples_final1 > num_samples_init1:
            seed1 = np.arange(num_samples_init1, num_samples_final1)
            seed = seed.tolist() + seed1.tolist()
        else:
            seed = seed.tolist()

    n_samples_str = f"_{len(seed)}" if len(seed) > 1000 else ""
    filename = config.output_path + mode + n_samples_str + '.h5'
    config.output_path = os.path.join(output_path, filename)

    pool = mp.Pool(mp.cpu_count())
    print(f"Number of generated sequences is {len(seed)}")
    pool.starmap(simulator, zip(repeat(config), seed))

    # # sharpclaw doesn't work with multiprocessing
    # for seed_i in seed:
    #     simulator(config, seed_i)

    return


if __name__ == "__main__":
    main()
