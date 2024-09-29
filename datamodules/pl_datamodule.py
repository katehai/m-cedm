'''
Adapted from: https://github.com/jaggbow/magnet/blob/main/datamodule/h5_datamodule.py
'''
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datamodules.h5_dataset import *
from utils import DotDict


class HDF5Datamodule(pl.LightningDataModule):
    def __init__(
            self,
            name='h5_datamodule_norm',
            train_path="data/train.h5",
            val_path="data/val.h5",
            test_path="data/test.h5",
            return_abs_coords=False,
            return_grid=False,
            norm_x=False,
            norm_t=False,
            norm_input=True,
            norm_target=True,
            flip_xy=False,
            const_norm_stats=True,
            use_theta=False,
            use_tar_ic=False,
            num_workers=2,
            batch_size=32,
            test_batch_size=None,
            down_factor: int = 1,
            down_interp: bool = True,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.name = name
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.return_abs_coords = return_abs_coords
        self.return_grid = return_grid
        self.norm_x = norm_x
        self.norm_t = norm_t
        self.norm_input = norm_input
        self.norm_target = norm_target
        self.flip_xy = flip_xy
        self.const_norm_stats = const_norm_stats
        self.use_theta = use_theta
        self.use_tar_ic = use_tar_ic
        self.eps = 1e-6
        self.down_factor = down_factor
        self.down_interp = down_interp

        self.batch_size = batch_size

        if test_batch_size is None or test_batch_size == 0:
            test_batch_size = batch_size
        self.test_batch_size = test_batch_size

        self.num_workers = num_workers

        # get mean and std from the train set for inputs and targets
        mean_std, min_max = self.get_stats()
        input_mean, input_std, target_mean, target_std = mean_std
        self.input_mean = input_mean
        self.input_std = input_std + self.eps
        self.target_mean = target_mean
        self.target_std = target_std + self.eps

        input_min, input_max, target_min, target_max = min_max
        self.input_min = input_min
        self.input_min_max = input_max - input_min + self.eps
        self.target_min = target_min
        self.target_min_max = target_max - target_min + self.eps

    def get_stats(self):
        if self.const_norm_stats:
            # calculate mean and std across the whole train set for inputs and targets
            # values for the stats are scalars
            with h5py.File(self.train_path, 'r') as f:
                input_mean = torch.tensor(f.attrs['inp_mean'], dtype=torch.float32)
                input_std = torch.tensor(f.attrs['inp_std'], dtype=torch.float32)
                target_mean = torch.tensor(f.attrs['tar_mean'], dtype=torch.float32)
                target_std = torch.tensor(f.attrs['tar_std'], dtype=torch.float32)

                input_min = torch.tensor(f.attrs['inp_min'], dtype=torch.float32)
                input_max = torch.tensor(f.attrs['inp_max'], dtype=torch.float32)
                target_min = torch.tensor(f.attrs['tar_min'], dtype=torch.float32)
                target_max = torch.tensor(f.attrs['tar_max'], dtype=torch.float32)

                f.close()
        else:
            # calculate mean and std across batch dimension for inputs and targets
            with h5py.File(self.train_path, 'r') as f:
                inputs, targets = [], []
                for key in f.keys():
                    sample = f[key]
                    inp = sample['data']['input'][:]
                    target = sample['data']['target'][:]
                    inputs.append(inp)
                    targets.append(target)
                f.close()

            inputs = torch.tensor(np.stack(inputs, axis=0), dtype=torch.float32).squeeze(dim=-1)
            targets = torch.tensor(np.stack(targets, axis=0), dtype=torch.float32).squeeze(dim=-1)

            input_mean = torch.mean(inputs, dim=0)
            input_std = torch.std(inputs, dim=0)
            target_mean = torch.mean(targets, dim=0)
            target_std = torch.std(targets, dim=0)

            input_min = torch.min(inputs, dim=0)[0]
            input_max = torch.max(inputs, dim=0)[0]
            target_min = torch.min(targets, dim=0)[0]
            target_max = torch.max(targets, dim=0)[0]

        mean_std = [input_mean, input_std, target_mean, target_std]
        min_max = [input_min, input_max, target_min, target_max]

        return mean_std, min_max

    def setup(self, stage=None):
        self.train_dataset = HDF5Dataset(
            datapath=self.train_path,
            return_abs_coords=self.return_abs_coords,
            return_grid=self.return_grid,
            input_mean=self.input_mean,
            input_std=self.input_std,
            target_mean=self.target_mean,
            target_std=self.target_std,
            norm_x=self.norm_x,
            norm_t=self.norm_t,
            norm_input=self.norm_input,
            norm_target=self.norm_target,
            flip_xy=self.flip_xy,
            use_theta=self.use_theta,
            use_tar_ic=self.use_tar_ic,
            dtype=torch.float32)

        self.val_dataset = HDF5Dataset(
            datapath=self.val_path,
            return_abs_coords=self.return_abs_coords,
            return_grid=self.return_grid,
            input_mean=self.input_mean,
            input_std=self.input_std,
            target_mean=self.target_mean,
            target_std=self.target_std,
            norm_x=self.norm_x,
            norm_t=self.norm_t,
            norm_input=self.norm_input,
            norm_target=self.norm_target,
            flip_xy=self.flip_xy,
            use_theta=self.use_theta,
            use_tar_ic=self.use_tar_ic,
            dtype=torch.float32,
            down_factor=self.down_factor,
            down_interp=self.down_interp)

        self.test_dataset = HDF5Dataset(
            datapath=self.test_path,
            return_abs_coords=self.return_abs_coords,
            return_grid=self.return_grid,
            input_mean=self.input_mean,
            input_std=self.input_std,
            target_mean=self.target_mean,
            target_std=self.target_std,
            norm_x=self.norm_x,
            norm_t=self.norm_t,
            norm_input=self.norm_input,
            norm_target=self.norm_target,
            flip_xy=self.flip_xy,
            use_theta=self.use_theta,
            use_tar_ic=self.use_tar_ic,
            dtype=torch.float32,
            down_factor=self.down_factor,
            down_interp=self.down_interp)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=True)

    def get_norm_stats(self):
        # if inputs and targets are changing places than we need to change places for the corresponding stats too
        if self.flip_xy:
            return DotDict({
                "norm_target": self.norm_input,
                "target_mean": self.input_mean,
                "target_std": self.input_std,
                "target_min": self.input_min,
                "target_min_max": self.input_min_max,
                "norm_input": self.norm_target,
                "input_mean": self.target_mean,
                "input_std": self.target_std,
                "input_min": self.target_min,
                "input_min_max": self.target_min_max,
            })
        else:
            return DotDict({
                    "norm_target": self.norm_target,
                    "target_mean": self.target_mean,
                    "target_std": self.target_std,
                    "target_min": self.target_min,
                    "target_min_max": self.target_min_max,
                    "norm_input": self.norm_input,
                    "input_mean": self.input_mean,
                    "input_std": self.input_std,
                    "input_min": self.input_min,
                    "input_min_max": self.input_min_max,
            })


class HDF5MaskDatamodule(HDF5Datamodule):
    def __init__(self,
            name='h5_mask_datamodule_norm',
            train_path="data/train.h5",
            val_path="data/val.h5",
            test_path="data/test.h5",
            return_abs_coords=False,
            return_grid=False,
            norm_x=False,
            norm_t=False,
            norm_input=True,
            norm_target=True,
            flip_xy=False,
            const_norm_stats=True,
            use_theta=False,
            use_tar_ic=False,
            num_workers=2,
            batch_size=32,
            test_batch_size=None,
            down_factor=1,
            down_interp=True):
        super().__init__(name,
            train_path,
            val_path,
            test_path,
            return_abs_coords,
            return_grid,
            norm_x,
            norm_t,
            norm_input,
            norm_target,
            flip_xy,
            const_norm_stats,
            use_theta,
            use_tar_ic,
            num_workers,
            batch_size,
            test_batch_size,
            down_factor,
            down_interp)

    def setup(self, stage=None):
        self.train_dataset = HDF5MaskDataset(
            datapath=self.train_path,
            return_abs_coords=self.return_abs_coords,
            return_grid=self.return_grid,
            input_mean=self.input_mean,
            input_std=self.input_std,
            target_mean=self.target_mean,
            target_std=self.target_std,
            norm_x=self.norm_x,
            norm_t=self.norm_t,
            norm_input=self.norm_input,
            norm_target=self.norm_target,
            flip_xy=self.flip_xy,
            use_theta=self.use_theta,
            use_tar_ic=self.use_tar_ic,
            dtype=torch.float32,
            is_train=True)

        self.val_dataset = HDF5MaskDataset(
            datapath=self.val_path,
            return_abs_coords=self.return_abs_coords,
            return_grid=self.return_grid,
            input_mean=self.input_mean,
            input_std=self.input_std,
            target_mean=self.target_mean,
            target_std=self.target_std,
            norm_x=self.norm_x,
            norm_t=self.norm_t,
            norm_input=self.norm_input,
            norm_target=self.norm_target,
            flip_xy=self.flip_xy,
            use_theta=self.use_theta,
            use_tar_ic=self.use_tar_ic,
            dtype=torch.float32,
            down_factor=self.down_factor,
            down_interp=self.down_interp)

        self.test_dataset = HDF5MaskDataset(
            datapath=self.test_path,
            return_abs_coords=self.return_abs_coords,
            return_grid=self.return_grid,
            input_mean=self.input_mean,
            input_std=self.input_std,
            target_mean=self.target_mean,
            target_std=self.target_std,
            norm_x=self.norm_x,
            norm_t=self.norm_t,
            norm_input=self.norm_input,
            norm_target=self.norm_target,
            flip_xy=self.flip_xy,
            use_theta=self.use_theta,
            use_tar_ic=self.use_tar_ic,
            dtype=torch.float32,
            down_factor=self.down_factor,
            down_interp=self.down_interp)


class HDF5TimeMaskDatamodule(HDF5MaskDatamodule):
    def __init__(self,
            name='h5_mask_datamodule_norm',
            train_path="data/train.h5",
            val_path="data/val.h5",
            test_path="data/test.h5",
            return_abs_coords=False,
            return_grid=False,
            norm_x=False,
            norm_t=False,
            norm_input=True,
            norm_target=True,
            flip_xy=False,
            const_norm_stats=True,
            use_theta=False,
            use_tar_ic=False,
            num_workers=2,
            batch_size=32,
            test_batch_size=None,
            down_factor=1,
            down_interp=True,
            add_time_masks=False):
        super().__init__(name,
            train_path,
            val_path,
            test_path,
            return_abs_coords,
            return_grid,
            norm_x,
            norm_t,
            norm_input,
            norm_target,
            flip_xy,
            const_norm_stats,
            use_theta,
            use_tar_ic,
            num_workers,
            batch_size,
            test_batch_size,
            down_factor,
            down_interp)
        self.add_time_masks = add_time_masks

    def setup(self, stage=None):
        self.train_dataset = HDF5TimeMaskDataset(
            datapath=self.train_path,
            return_abs_coords=self.return_abs_coords,
            return_grid=self.return_grid,
            input_mean=self.input_mean,
            input_std=self.input_std,
            target_mean=self.target_mean,
            target_std=self.target_std,
            norm_x=self.norm_x,
            norm_t=self.norm_t,
            norm_input=self.norm_input,
            norm_target=self.norm_target,
            flip_xy=self.flip_xy,
            use_theta=self.use_theta,
            use_tar_ic=self.use_tar_ic,
            dtype=torch.float32,
            is_train=True)

        self.val_dataset = HDF5TimeMaskDataset(
            datapath=self.val_path,
            return_abs_coords=self.return_abs_coords,
            return_grid=self.return_grid,
            input_mean=self.input_mean,
            input_std=self.input_std,
            target_mean=self.target_mean,
            target_std=self.target_std,
            norm_x=self.norm_x,
            norm_t=self.norm_t,
            norm_input=self.norm_input,
            norm_target=self.norm_target,
            flip_xy=self.flip_xy,
            use_theta=self.use_theta,
            use_tar_ic=self.use_tar_ic,
            dtype=torch.float32,
            down_factor=self.down_factor,
            down_interp=self.down_interp,
            add_time_masks=self.add_time_masks)

        self.test_dataset = HDF5TimeMaskDataset(
            datapath=self.test_path,
            return_abs_coords=self.return_abs_coords,
            return_grid=self.return_grid,
            input_mean=self.input_mean,
            input_std=self.input_std,
            target_mean=self.target_mean,
            target_std=self.target_std,
            norm_x=self.norm_x,
            norm_t=self.norm_t,
            norm_input=self.norm_input,
            norm_target=self.norm_target,
            flip_xy=self.flip_xy,
            use_theta=self.use_theta,
            use_tar_ic=self.use_tar_ic,
            dtype=torch.float32,
            down_factor=self.down_factor,
            down_interp=self.down_interp,
            add_time_masks=self.add_time_masks)


class HDF5SparseMaskDatamodule(HDF5MaskDatamodule):
    def __init__(self,
            name='h5_mask_datamodule_norm',
            train_path="data/train.h5",
            val_path="data/val.h5",
            test_path="data/test.h5",
            return_abs_coords=False,
            return_grid=False,
            norm_x=False,
            norm_t=False,
            norm_input=True,
            norm_target=True,
            flip_xy=False,
            const_norm_stats=True,
            use_theta=False,
            use_tar_ic=False,
            num_workers=2,
            batch_size=32,
            test_batch_size=None,
            down_factor=1,
            down_interp=True,
            add_res_masks=False):
        super().__init__(name,
            train_path,
            val_path,
            test_path,
            return_abs_coords,
            return_grid,
            norm_x,
            norm_t,
            norm_input,
            norm_target,
            flip_xy,
            const_norm_stats,
            use_theta,
            use_tar_ic,
            num_workers,
            batch_size,
            test_batch_size,
            down_factor,
            down_interp)
        self.add_res_masks = add_res_masks

    def setup(self, stage=None):
        self.train_dataset = HDF5SparseMaskDataset(
            datapath=self.train_path,
            return_abs_coords=self.return_abs_coords,
            return_grid=self.return_grid,
            input_mean=self.input_mean,
            input_std=self.input_std,
            target_mean=self.target_mean,
            target_std=self.target_std,
            norm_x=self.norm_x,
            norm_t=self.norm_t,
            norm_input=self.norm_input,
            norm_target=self.norm_target,
            flip_xy=self.flip_xy,
            use_theta=self.use_theta,
            use_tar_ic=self.use_tar_ic,
            dtype=torch.float32,
            is_train=True)

        self.val_dataset = HDF5SparseMaskDataset(
            datapath=self.val_path,
            return_abs_coords=self.return_abs_coords,
            return_grid=self.return_grid,
            input_mean=self.input_mean,
            input_std=self.input_std,
            target_mean=self.target_mean,
            target_std=self.target_std,
            norm_x=self.norm_x,
            norm_t=self.norm_t,
            norm_input=self.norm_input,
            norm_target=self.norm_target,
            flip_xy=self.flip_xy,
            use_theta=self.use_theta,
            use_tar_ic=self.use_tar_ic,
            dtype=torch.float32,
            down_factor=self.down_factor,
            down_interp=self.down_interp,
            add_res_masks=self.add_res_masks)

        self.test_dataset = HDF5SparseMaskDataset(
            datapath=self.test_path,
            return_abs_coords=self.return_abs_coords,
            return_grid=self.return_grid,
            input_mean=self.input_mean,
            input_std=self.input_std,
            target_mean=self.target_mean,
            target_std=self.target_std,
            norm_x=self.norm_x,
            norm_t=self.norm_t,
            norm_input=self.norm_input,
            norm_target=self.norm_target,
            flip_xy=self.flip_xy,
            use_theta=self.use_theta,
            use_tar_ic=self.use_tar_ic,
            dtype=torch.float32,
            down_factor=self.down_factor,
            down_interp=self.down_interp,
            add_res_masks=self.add_res_masks)
