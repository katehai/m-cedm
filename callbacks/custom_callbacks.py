import os
from pytorch_lightning import Callback
import wandb
import numpy as np
from matplotlib import pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable


def set_colorbar(fig, ax1, im1, add_colorbar):
    if add_colorbar:
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='4%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')


class PlotModelPredictions(Callback):
    def __init__(self, num_samples=5, log_every=100):
        super().__init__()
        self.num_samples = num_samples
        self.log_every = log_every
        self.val_pred = None
        self.val_gt = None
        self.test_pred = None
        self.test_gt = None
        self.plot_error_val = False
        self.plot_error_test = True
        self.add_colorbar = True

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0 and trainer.logger is not None and 'pred' in outputs.keys() and 'target' in outputs.keys():
            pred = outputs['pred'].detach().cpu().numpy()
            gt = outputs['target'].detach().cpu().numpy()
            num_samples = self.num_samples if len(pred) > self.num_samples else len(pred)

            self.val_pred = pred[:num_samples]
            self.val_gt = gt[:num_samples]

    def on_validation_epoch_end(self, trainer, pl_module):
        # only log every 100 steps
        step = trainer.global_step
        steps_per_epoch = trainer.num_training_batches
        log_every_step = self.log_every * steps_per_epoch
        if step % log_every_step != 0 and trainer.estimated_stepping_batches != step:  # save after the last batch
            # reset
            self.val_pred = None
            self.val_gt = None
            return

        # check that the predictions are saved
        if self.val_pred is None or self.val_gt is None:
            return

        self.plot_predictions(trainer, step, self.val_pred, self.val_gt, self.plot_error_val)

        # reset
        self.val_pred = None
        self.val_gt = None

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if (trainer.logger is not None and (self.test_gt is None or len(self.test_gt) < self.num_samples)
                and (outputs is not None and 'pred' in outputs.keys() and 'target' in outputs.keys())):
            pred = outputs['pred'].detach().cpu().numpy()
            gt = outputs['target'].detach().cpu().numpy()

            curr_len = len(self.test_gt) if self.test_gt is not None else 0
            max_len = self.num_samples - curr_len
            num_samples = max_len if len(pred) > max_len else len(pred)

            if self.test_pred is None:
                self.test_pred = pred[:num_samples]
                self.test_gt = gt[:num_samples]
            elif num_samples > 0:
                self.test_pred = np.concatenate([self.test_pred, pred[:num_samples]], axis=0)
                self.test_gt = np.concatenate([self.test_gt, gt[:num_samples]], axis=0)

    def on_test_epoch_end(self, trainer, pl_module):
        step = trainer.global_step

        # check that the predictions are saved
        if self.test_pred is None or self.test_gt is None:
            return

        self.plot_predictions(trainer, step, self.test_pred, self.test_gt, self.plot_error_test)

        # reset
        self.test_pred = None
        self.test_gt = None

    def plot_predictions(self, trainer, step, pred, gt, plot_error):
        n_samples = len(pred)
        for i in range(n_samples):
            pred_i = pred[i]
            target_i = gt[i]
            n_vars = pred_i.shape[-1]
            n_cols = 3 if plot_error else 2

            fig, axs = plt.subplots(n_vars, n_cols, figsize=(3 * n_cols, 3 * n_vars), squeeze=False, sharex=True,
                                    sharey=True)
            for j in range(n_vars):
                ax = axs[j, 0]
                im1 = ax.imshow(pred_i[..., j:j + 1].transpose(1, 0, 2), cmap='jet')
                set_colorbar(fig, ax, im1, self.add_colorbar)

                ax = axs[j, 1]
                im2 = ax.imshow(target_i[..., j:j + 1].transpose(1, 0, 2), cmap='jet')
                set_colorbar(fig, ax, im2, self.add_colorbar)

                if plot_error:
                    ax = axs[j, 2]
                    im3 = ax.imshow(np.abs(pred_i - target_i)[..., j:j + 1].transpose(1, 0, 2), cmap='Greys')
                    set_colorbar(fig, ax, im3, self.add_colorbar)

                if j == 0:
                    axs[j, 0].set_title(f'pred {j}')
                    axs[j, 1].set_title(f'target {j}')

            plt.subplots_adjust(wspace=0.35, hspace=0.05)
            trainer.logger.experiment.log({f'prediction_{str(i).zfill(2)}': wandb.Image(fig)}, step=step)
            plt.close(fig)


class PlotDiffusionTrajectory(Callback):
    def __init__(self, num_samples=5, log_every=100):
        super().__init__()
        self.num_samples = num_samples
        self.log_every = log_every
        self.plot_every = 100
        self.val_traj = None
        self.val_gt = None
        self.test_traj = None
        self.test_gt = None
        self.plot_error_val = False
        self.plot_error_test = True
        self.add_colorbar = True

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0 and trainer.logger is not None and 'traj' in outputs.keys() and 'gt' in outputs.keys():
            traj = outputs['traj'].detach().cpu().numpy()  # b, t, h, w, c
            gt = outputs['gt'].detach().cpu().numpy()
            num_samples = self.num_samples if len(traj) > self.num_samples else len(traj)

            self.val_traj = traj[:num_samples]
            self.val_gt = gt[:num_samples]

        if batch_idx == 0 and trainer.logger is not None and 'traj_h' in outputs.keys() and 'gt_h' in outputs.keys():
            # add samples for h
            traj = outputs['traj_h'].detach().cpu().numpy()  # b, t, h, w, c
            gt = outputs['gt_h'].detach().cpu().numpy()
            num_samples = self.num_samples if len(traj) > self.num_samples else len(traj)

            self.val_traj = traj[:num_samples]
            self.val_gt = gt[:num_samples]

            # add samples for u
            traj = outputs['traj_u'].detach().cpu().numpy()  # b, t, h, w, c
            gt = outputs['gt_u'].detach().cpu().numpy()
            num_samples = self.num_samples if len(traj) > self.num_samples else len(traj)

            self.val_traj = np.concatenate([self.val_traj, traj[:num_samples]], axis=0)
            self.val_gt = np.concatenate([self.val_gt, gt[:num_samples]], axis=0)
        return

    def on_validation_epoch_end(self, trainer, pl_module):
        # check that the predictions are saved
        if self.val_traj is None or self.val_gt is None:
            return

        self.plot_trajectories(trainer, self.val_traj, self.val_gt, self.plot_error_val)

        # reset
        self.val_traj = None
        self.val_gt = None

        return

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if (trainer.logger is not None and (self.test_gt is None or len(self.test_gt) < self.num_samples)
                and (outputs is not None and 'traj' in outputs.keys() and 'gt' in outputs.keys())):
            traj = outputs['traj'].detach().cpu().numpy()  # b, t, h, w, c
            gt = outputs['gt'].detach().cpu().numpy()

            curr_len = len(self.test_gt) if self.test_gt is not None else 0
            max_len = self.num_samples - curr_len
            self.add_test_samples(traj, gt, max_len)

        elif (trainer.logger is not None and (self.test_gt is None or len(self.test_gt) < 2 * self.num_samples)
                and (outputs is not None and 'traj_h' in outputs.keys() and 'gt_h' in outputs.keys())):
            traj = outputs['traj_h'].detach().cpu().numpy()  # b, t, h, w, c
            gt = outputs['gt_h'].detach().cpu().numpy()

            curr_len = len(self.test_gt) if self.test_gt is not None else 0
            max_len = self.num_samples - curr_len // 2
            self.add_test_samples(traj, gt, max_len)

            traj = outputs['traj_u'].detach().cpu().numpy()  # b, t, h, w, c
            gt = outputs['gt_u'].detach().cpu().numpy()

            curr_len = len(self.test_gt) if self.test_gt is not None else 0
            max_len = self.num_samples - curr_len // 2
            self.add_test_samples(traj, gt, max_len)

    def add_test_samples(self, traj, gt, max_len):
        num_samples = max_len if len(traj) > max_len else len(traj)
        if self.test_traj is None:
            self.test_traj = traj[:num_samples]
            self.test_gt = gt[:num_samples]
        elif num_samples > 0:
            self.test_traj = np.concatenate([self.test_traj, traj[:num_samples]], axis=0)
            self.test_gt = np.concatenate([self.test_gt, gt[:num_samples]], axis=0)

    def on_test_epoch_end(self, trainer, pl_module):
        # check that the predictions are saved
        if self.test_traj is None or self.test_gt is None:
            return

        self.plot_trajectories(trainer, self.test_traj, self.test_gt, self.plot_error_test)

        # reset
        self.test_traj = None
        self.test_gt = None

    def plot_trajectories(self, trainer, traj, gt, plot_error):
        n_samples = len(traj)
        n_steps = traj.shape[1]
        for i in range(n_samples):
            target = gt[i]

            for step in np.arange(n_steps, -1, -self.plot_every):
                t_idx = step - 1
                pred = traj[i, t_idx]

                if len(pred.shape) < 4:
                    pred = pred[:, :, None, :]  # add repeats dimension

                n_vars = pred.shape[-1]
                n_repeats = pred.shape[2]
                n_cols = n_repeats + 2 if plot_error else n_repeats + 1  # n_repeats + gt + error

                fig, axs = plt.subplots(n_vars, n_cols, figsize=(3.5 * n_cols, 3 * n_vars), squeeze=False,
                                        sharex=True, sharey=True)
                for j in range(n_vars):
                    vmin = np.min([np.min(pred[..., j:j + 1]), np.min(target[..., j:j + 1])])
                    vmax = np.max([np.max(pred[..., j:j + 1]), np.max(target[..., j:j + 1])])
                    # vmin = np.min(target[..., j:j + 1])
                    # vmax = np.max(target[..., j:j + 1])

                    for k in range(n_repeats):
                        ax = axs[j, k]
                        im1 = ax.imshow(pred[..., k, j:j + 1].transpose(1, 0, 2), vmin=vmin, vmax=vmax, cmap='jet')
                        set_colorbar(fig, ax, im1, self.add_colorbar)
                        # ax.imshow(pred[..., k, j:j + 1].transpose(1, 0, 2), cmap='jet')

                    ax = axs[j, n_repeats]
                    im2 = ax.imshow(target[..., j:j + 1].transpose(1, 0, 2), vmin=vmin, vmax=vmax, cmap='jet')
                    set_colorbar(fig, ax, im2, self.add_colorbar)

                    if plot_error:
                        ax = axs[j, n_repeats + 1]
                        pred0 = pred[..., -1, :]  # if we have several repeats, take the last one
                        im3 = ax.imshow(np.abs(pred0 - target)[..., j:j + 1].transpose(1, 0, 2), cmap='Greys')
                        set_colorbar(fig, ax, im3, self.add_colorbar)

                    if j == 0:
                        axs[j, 0].set_title(f'pred {j}')
                        axs[j, n_repeats].set_title(f'target {j}')

                plt.subplots_adjust(wspace=0.05, hspace=0.15)
                trainer.logger.experiment.log({f'prediction_{i}_{step}': wandb.Image(fig)})
                plt.close(fig)


class SaveGeneratedSamples(Callback):
    def __init__(self, num_samples=5, dirpath=None, traj_name="traj", gt_name="gt"):
        super().__init__()
        self.num_samples = num_samples
        self.plot_every = 100
        self.val_traj = None
        self.val_gt = None
        self.test_traj = None
        self.test_gt = None
        self.dirpath = dirpath
        self.traj_name = traj_name
        self.gt_name = gt_name

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0 and trainer.logger is not None and self.traj_name in outputs.keys():
            if self.traj_name in outputs.keys() and self.gt_name in outputs.keys():
                traj = outputs[self.traj_name].detach().cpu().numpy()  # b, t, h, w, c
                gt = outputs[self.gt_name].detach().cpu().numpy()
                num_samples = self.num_samples if len(traj) > self.num_samples else len(traj)

                self.val_traj = traj[:num_samples]
                self.val_gt = gt[:num_samples]
        return

    def on_validation_epoch_end(self, trainer, pl_module):
        # check that the predictions are saved
        if self.val_traj is None or self.val_gt is None:
            return

        self.save_samples(self.val_traj, self.val_gt, "val.npy")

        # reset
        self.val_traj = None
        self.val_gt = None

        return

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if (trainer.logger is not None and (self.test_gt is None or len(self.test_gt) < self.num_samples)
                and (outputs is not None and self.traj_name in outputs.keys() and self.gt_name in outputs.keys())):
            traj = outputs[self.traj_name].detach().cpu().numpy()  # b, t, h, w, c
            gt = outputs[self.gt_name].detach().cpu().numpy()

            curr_len = len(self.test_gt) if self.test_gt is not None else 0
            max_len = self.num_samples - curr_len
            num_samples = max_len if len(traj) > max_len else len(traj)

            if self.test_traj is None:
                self.test_traj = traj[:num_samples]
                self.test_gt = gt[:num_samples]
            elif num_samples > 0:
                self.test_traj = np.concatenate([self.test_traj, traj[:num_samples]], axis=0)
                self.test_gt = np.concatenate([self.test_gt, gt[:num_samples]], axis=0)

    def on_test_epoch_end(self, trainer, pl_module):
        # check that the predictions are saved
        if self.test_traj is None or self.test_gt is None:
            return

        self.save_samples(self.test_traj, self.test_gt, "test.npy")

        # reset
        self.test_traj = None
        self.test_gt = None

    def save_samples(self, traj, gt, filename):
        file_gen = filename.split('.')[0] + '_gen.npy'
        file_gt = filename.split('.')[0] + '_gt.npy'

        if self.dirpath is not None:
            if not os.path.exists(self.dirpath):
                os.makedirs(self.dirpath)

            file_path_gt = os.path.join(self.dirpath, file_gt)
            file_path_gen = os.path.join(self.dirpath, file_gen)
        else:
            file_path_gt = file_gt
            file_path_gen = file_gen

        np.save(file_path_gt, gt)
        np.save(file_path_gen, traj)


class SaveFullGeneratedSamples(Callback):
    def __init__(self, dirpath=None, traj_name="traj", gt_name="gt"):
        super().__init__()
        self.test_traj = None
        self.test_gt = None
        self.dirpath = dirpath
        self.traj_name = traj_name
        self.gt_name = gt_name

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if (trainer.logger is not None and (outputs is not None and self.traj_name in outputs.keys()
                                            and self.gt_name in outputs.keys())):
            traj = outputs[self.traj_name].detach().cpu().numpy()  # b, t, h, w, c
            gt = outputs[self.gt_name].detach().cpu().numpy()

            if self.test_traj is None:
                self.test_traj = traj
                self.test_gt = gt
            else:
                self.test_traj = np.concatenate([self.test_traj, traj], axis=0)
                self.test_gt = np.concatenate([self.test_gt, gt], axis=0)

    def on_test_epoch_end(self, trainer, pl_module):
        # check that the predictions are saved
        if self.test_traj is None or self.test_gt is None:
            return

        self.save_samples(self.test_traj, self.test_gt, "test.npy")

        # reset
        self.test_traj = None
        self.test_gt = None

    def save_samples(self, traj, gt, filename):
        file_gen = filename.split('.')[0] + '_gen.npy'
        file_gt = filename.split('.')[0] + '_gt.npy'

        if self.dirpath is not None:
            if not os.path.exists(self.dirpath):
                os.makedirs(self.dirpath)

            file_path_gt = os.path.join(self.dirpath, file_gt)
            file_path_gen = os.path.join(self.dirpath, file_gen)
        else:
            file_path_gt = file_gt
            file_path_gen = file_gen

        np.save(file_path_gt, gt)
        np.save(file_path_gen, traj)

