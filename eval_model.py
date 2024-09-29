import os
import hydra
from omegaconf import DictConfig

from typing import List
# from lightning_lite.utilities.seed import seed_everything  # for older versions of pytorch
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Callback, Trainer, LightningDataModule
from utils import override_data_folders

import logger_utils

log = logger_utils.get_logger(__name__)


@hydra.main(config_path="configs/", config_name="config.yaml", version_base="1.1")
def main(cfg: DictConfig):
    dataset = cfg.datamodule.name
    model = cfg.model.hparams.name
    sampler = cfg.diff_sampler.name if cfg.get("diff_sampler", None) is not None else ""
    subname = f"_{cfg.subname}" if len(cfg.get("subname", "")) > 0 else ""
    wandb_logger = WandbLogger(project='gen_no', name=f"{model}_{dataset}_{cfg.seed}_test_{sampler}{subname}",
                               offline=True)

    # override paths to datasets depending on the system that is selected
    res = cfg.get("res", 128)
    cfg.datamodule = override_data_folders(cfg.datamodule, cfg.dataroot, cfg.system, res)

    # save output_dir information
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if wandb_logger.experiment.config.get("output_dir", None) is None:
        wandb_logger.experiment.config["output_dir"] = output_dir

    print(f"Evaluate the model {cfg.model.hparams.name} on the {cfg.datamodule.name} dataset")
    seed_everything(cfg.seed, workers=True)
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    model_path = os.path.join(cfg.ckpt_path, "checkpoints/last.ckpt")  # to set the checkpoint in the config

    # save the path to the used checkpoint to wandb
    if wandb_logger.experiment.config.get("checkpoint_dir", None) is None:
        wandb_logger.experiment.config["checkpoint_dir"] = model_path

    model = hydra.utils.instantiate(cfg.model)

    # set sampler params for testing
    if cfg.get("diff_sampler", None) is not None:
        print("Set sampler params")
        model.set_test_sampler_params(cfg.diff_sampler)

        ## allow overriding test_sampler for sweep runs
        wandb_logger.experiment.config.update({"test_sampler": cfg.diff_sampler}, allow_val_change=True)

    # set PDE loss function
    if cfg.get("system", None) is not None:
        print("Set pde loss for a concrete system")
        flip_xy = datamodule.flip_xy
        model.set_pde_loss_function(cfg.system, flip_xy)
        if wandb_logger.experiment.config.get("system", None) is None:
            wandb_logger.experiment.config["system"] = cfg.system

    model.eval()

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=wandb_logger, _convert_="partial"
    )

    trainer.test(model, datamodule, ckpt_path=model_path, verbose=True)  # Test the model

    metric_key = "test_mae_u_scaled"
    if trainer.callback_metrics.get(metric_key, None) is None:
        log.warning(f"Metric {metric_key} not found in summary")
        metric = 0.0
    else:
        metric = trainer.callback_metrics[metric_key]
    print(f'Metric {metric_key}: {metric}')

    return metric


if __name__ == "__main__":
    main()
