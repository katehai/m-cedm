'''
The file is adopted from https://github.com/jaggbow/magnet/blob/main/utils.py
'''
import wandb
import logging
from pytorch_lightning.utilities import rank_zero_only


def get_logger(name=__name__):
    """
    Initializes multi-GPU-friendly python command line logger.
    https://github.com/ashleve/lightning-hydra-template/blob/8b62eef9d0d9c863e88c0992595688d6289d954f/src/utils/utils.py#L12
    """

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
            "debug",
            "info",
            "warning",
            "error",
            "exception",
            "fatal",
            "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def save_wandb_artifact(logger, ckpt_path: str, alies: str):
    if logger is not None:
        artifact = wandb.Artifact(name=f"model-{logger.experiment.id}", type="model")
        artifact.add_file(local_path=ckpt_path)
        logger.experiment.log_artifact(artifact, aliases=[alies])
    return
