'''
Adapted from: https://github.com/jaggbow/magnet/blob/main/run.py
'''
import os.path
import warnings
from typing import List

warnings.filterwarnings("ignore")

import numpy as np
from pytorch_lightning.loggers import WandbLogger
# from lightning_lite.utilities.seed import seed_everything  # for older versions of pytorch
from pytorch_lightning import seed_everything
from pytorch_lightning import Callback, Trainer, LightningDataModule
from pytorch_lightning.utilities import rank_zero_only

import hydra
from omegaconf import DictConfig
import wandb
from utils import override_data_folders

import logger_utils

log = logger_utils.get_logger(__name__)


os.environ["WANDB_INIT_TIMEOUT"] = "300"


@hydra.main(config_path="configs/", config_name="config.yaml", version_base="1.1")
def main(cfg: DictConfig):
    dataset = cfg.datamodule.name
    model = cfg.model.hparams.name

    # override paths to datasets depending on the system that is selected
    res = cfg.get("res", 128)
    n_train = cfg.get("n_train", 1000)
    cfg.datamodule = override_data_folders(cfg.datamodule, cfg.dataroot, cfg.system, res, n_train=n_train)

    print("This run trains and tests the model", model, "on the", dataset, "dataset")
    seed_everything(cfg.seed, workers=True)
    sampler = cfg.diff_sampler.name if cfg.get("diff_sampler", None) is not None else ""
    subname = f"_{cfg.subname}" if len(cfg.get("subname", "")) > 0 else ""
    wandb_logger = WandbLogger(project='gen_no', name=f"{model}_{dataset}_{cfg.seed}{sampler}{subname}", offline=True)

    # save output_dir information
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if rank_zero_only.rank == 0 and wandb_logger.experiment.config.get("output_dir", None) is None:
        wandb_logger.experiment.config["output_dir"] = output_dir
    print(f"Output dir is {output_dir}")

    # if there is a diffusion sampler with 100 samples, set test_batch_size to 1 for testing
    if cfg.get("diff_sampler", None) is not None and cfg.diff_sampler.n_samples == 100:
        cfg.datamodule.test_batch_size = 1

    # Initialize the datamodule0  x
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    if cfg.ckpt_path is not None and os.path.isdir(cfg.ckpt_path):
        # add typical ckpt.ckpt file name if directory name is passed
        ckpt_path = os.path.join(cfg.ckpt_path, "checkpoints/last.ckpt")
    else:
        ckpt_path = cfg.ckpt_path

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=wandb_logger, _convert_="partial"
    )

    override_epochs = cfg.override_epochs if hasattr(cfg, "override_epochs") else False
    if override_epochs and ckpt_path is not None and trainer.max_epochs < cfg.trainer.max_epochs:
        trainer.max_epochs = cfg.trainer.max_epochs

    model = hydra.utils.instantiate(cfg.model)

    # set sampler params for testing
    if cfg.get("diff_sampler", None) is not None:
        print("Set sampler params")
        model.set_test_sampler_params(cfg.diff_sampler)
        if rank_zero_only.rank == 0 and wandb_logger.experiment.config.get("test_sampler", None) is None:
            wandb_logger.experiment.config["test_sampler"] = cfg.diff_sampler

    # set PDE loss function
    if cfg.get("system", None) is not None:
        print("Set pde loss for a concrete system")
        flip_xy = datamodule.flip_xy
        model.set_pde_loss_function(cfg.system, flip_xy)
        if rank_zero_only.rank == 0 and wandb_logger.experiment.config.get("system", None) is None:
            wandb_logger.experiment.config["system"] = cfg.system

    trainer.fit(model, datamodule, ckpt_path=ckpt_path)  # Train the model
    log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")  # print datapath to best checkpoint

    metric_key = "val_mae_u_scaled"
    metric = np.inf
    if rank_zero_only.rank == 0:
        if trainer.logger.experiment.summary.get('metric_key', None) is None:
            log.warning(f"Metric {metric_key} not found in summary")
            metric = np.inf
        else:
            metric = trainer.logger.experiment.summary[metric_key]

    ## run test
    trainer.test(model, datamodule, verbose=True)  # Test the model

    ## save the best model ckpt to wandb: this takes too much wandb storage
    # logger_utils.save_wandb_artifact(wandb_logger, trainer.checkpoint_callback.best_model_path, "best")
    # logger_utils.save_wandb_artifact(wandb_logger, trainer.checkpoint_callback.last_model_path, "latest")
    # wandb.finish()

    return metric


if __name__ == "__main__":
    main()
