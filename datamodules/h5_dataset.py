'''
Adapted from: https://github.com/jaggbow/magnet/blob/main/datamodule/dataset.py
'''

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from utils import *


class HDF5Dataset(Dataset):
    """
    Load samples of an PDE Dataset, get items according to PDE.
    """

    def __init__(self,
                 datapath: str,
                 return_abs_coords: bool,
                 return_grid: bool,
                 input_mean: np.array,
                 input_std: np.array,
                 target_mean: np.array,
                 target_std: np.array,
                 norm_x: bool = False,
                 norm_t: bool = False,
                 norm_input: bool = True,
                 norm_target: bool = True,
                 flip_xy: bool = False,
                 use_theta: bool = False,
                 use_tar_ic: bool = False,
                 dtype=torch.float32,
                 down_factor: int = 1,
                 down_interp: bool = True):
        """Initialize the dataset object.
        Args:
            datapath: path to dataset
            dtype: floating precision of data
        """
        super().__init__()
        self.dtype = dtype
        self.datapath = datapath
        self.return_abs_coords = return_abs_coords
        self.return_grid = return_grid
        self.input_mean = input_mean if torch.is_tensor(input_mean) else torch.tensor(input_mean, dtype=dtype)
        self.input_std = input_std if torch.is_tensor(input_std) else torch.tensor(input_std, dtype=dtype)
        self.target_mean = target_mean if torch.is_tensor(target_mean) else torch.tensor(target_mean, dtype=dtype)
        self.target_std = target_std if torch.is_tensor(target_std) else torch.tensor(target_std, dtype=dtype)
        self.norm_x = norm_x
        self.norm_t = norm_t
        self.norm_input = norm_input
        self.norm_target = norm_target
        self.flip_xy = flip_xy
        self.use_theta = use_theta
        self.use_tar_ic = use_tar_ic
        self.down_factor = down_factor
        self.down_interp = down_interp

        print("Loading data from {}".format(datapath))
        # print(f"Input mean: {self.input_mean}")
        # print(f"Input std: {self.input_std}")
        # print(f"Target mean: {self.target_mean}")
        # print(f"Target std: {self.target_std}")

        # Extract list of seeds
        with h5py.File(datapath, 'r') as f:
            data_list = sorted(f.keys())
            self.data_list = np.array(data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int):
        """
        Returns data a batch item.
        Args:
            idx: data index in the h5 file
        Returns:
            torch.Tensor: input state
            torch.Tensor: dx or x
            torch.Tensor: dt or t
            torch.Tensor: target state
        """
        with h5py.File(self.datapath, 'r') as f:
            seed_group = f[self.data_list[idx]]

            # data dim = [t, x1, ..., xd, v]
            sample = seed_group["data"]
            inp = torch.tensor(sample["input"][:], dtype=self.dtype)
            target = torch.tensor(sample["target"][:], dtype=self.dtype)

            # normalize
            if self.norm_input:
                inp = (inp - self.input_mean) / self.input_std
            if self.norm_target:
                target = (target - self.target_mean) / self.target_std

            if self.flip_xy:
                # use inputs as targets and the other way around
                inp_cp = inp.clone()
                inp = target
                target = inp_cp

            if self.use_theta:
                const_keys = list(seed_group['const'].keys())
                theta = torch.ones(inp.shape[0], inp.shape[1], len(const_keys), dtype=self.dtype)
                for i, c in enumerate(const_keys):
                    theta[..., i] = torch.tensor(seed_group['const'][c][0], dtype=self.dtype)

                # stack theta and input
                inp = torch.cat([inp, theta], dim=-1)

            if self.use_tar_ic:
                n_times = inp.shape[0]
                ic = target[0:1].repeat(n_times, 1, 1)

                # stack ic to the input
                inp = torch.cat([inp, ic], dim=-1)

            x = torch.tensor(seed_group["grid"]["x"][:], dtype=self.dtype)
            t = torch.tensor(seed_group["grid"]["t"][:], dtype=self.dtype)

            if len(t) > len(inp):
                t = t[:-1]  # some simulators store an extra step in the end

            if self.norm_x:
                x = (x - x.min()) / (x.max() - x.min())

            if self.norm_t:
                t = (t - t.min()) / (t.max() - t.min())

            # downsample the input
            if self.down_factor > 1:
                each_x = 2 ** (self.down_factor - 1)
                # inp = inp[::each_x, ::each_x]
                # x = x[::each_x]
                # t = t[::each_x]
                # target = target[::each_x, ::each_x]

                if self.down_interp:
                    ## downsample the input and preserve the dimensionality by interpolation
                    inp1 = inp[::each_x, ::each_x]
                    target1 = target[::each_x, ::each_x]  # downsample and then interpolate

                    inp1 = F.interpolate(inp1.permute(2, 1, 0).unsqueeze(0), scale_factor=each_x, mode='bilinear',
                                         align_corners=False)
                    inp1 = inp1.squeeze(0).permute(2, 1, 0)
                    inp = inp1

                    target1 = F.interpolate(target1.permute(2, 1, 0).unsqueeze(0), scale_factor=each_x, mode='bilinear',
                                            align_corners=False)
                    target1 = target1.squeeze(0).permute(2, 1, 0)
                    target = target1
                else:
                    # feed inputs of smaller resolutions (64 x 64 or 32 x 32)
                    inp = F.interpolate(inp.permute(2, 1, 0).unsqueeze(0), scale_factor=1/each_x, mode='bilinear',
                                        align_corners=False)
                    inp = inp.squeeze(0).permute(2, 1, 0)

                    x = F.interpolate(x.unsqueeze(0).unsqueeze(0), scale_factor=1/each_x, mode='linear',
                                      align_corners=False).squeeze(0).squeeze(0)
                    t = F.interpolate(t.unsqueeze(0).unsqueeze(0), scale_factor=1/each_x, mode='linear',
                                      align_corners=False).squeeze(0).squeeze(0)

                    target = F.interpolate(target.permute(2, 1, 0).unsqueeze(0), scale_factor=1/each_x,
                                           mode='bilinear', align_corners=False)
                    target = target.squeeze(0).permute(2, 1, 0)

                # print(f"Input size is {inp.shape}")
                # print(f"x shape is {x.shape}")
                # print(f"t shape is {t.shape}")
                # print(f"Target size is {target.shape}")

            if self.return_abs_coords:
                if self.return_grid:
                    t_grid, x_grid = torch.meshgrid(t, x, indexing='ij')
                    t_grid = t_grid.unsqueeze(-1)
                    x_grid = x_grid.unsqueeze(-1)
                    return inp, t_grid, x_grid, target
                else:
                    return inp, x, t, target
            else:
                dx = torch.diff(x)[0]
                dt = torch.diff(t)[0]

                return inp, dx, dt, target


class HDF5MaskDataset(HDF5Dataset):
    def __init__(self,
                 datapath: str,
                 return_abs_coords: bool,
                 return_grid: bool,
                 input_mean: np.array,
                 input_std: np.array,
                 target_mean: np.array,
                 target_std: np.array,
                 norm_x: bool = False,
                 norm_t: bool = False,
                 norm_input: bool = True,
                 norm_target: bool = True,
                 flip_xy: bool = False,
                 use_theta: bool = False,
                 use_tar_ic: bool = False,
                 dtype=torch.float32,
                 down_factor: int = 1,
                 down_interp: bool = True,
                 is_train=False):
        super().__init__(
                 datapath,
                 return_abs_coords,
                 return_grid,
                 input_mean,
                 input_std,
                 target_mean,
                 target_std,
                 norm_x,
                 norm_t,
                 norm_input,
                 norm_target,
                 flip_xy,
                 use_theta,
                 use_tar_ic,
                 dtype,
                 down_factor,
                 down_interp)

        self.is_train = is_train

    def sample_mask(self, inp, target):
        # mask = 1 -> missing element
        # mask = 0 -> available element
        if self.is_train:
            if torch.rand(1) > 0.5:  # 0.8:
                inp_mask = torch.zeros_like(inp)
                tar_mask = torch.ones_like(target)
            else:
                inp_mask = torch.ones_like(inp)
                tar_mask = torch.zeros_like(target)

            mask = torch.cat([inp_mask, tar_mask], dim=-1)
        else:
            inp_mask = torch.zeros_like(inp)
            tar_mask = torch.ones_like(target)
            mask1 = torch.cat([inp_mask, tar_mask], dim=-1)

            inp_mask = torch.ones_like(inp)
            tar_mask = torch.zeros_like(target)
            mask2 = torch.cat([inp_mask, tar_mask], dim=-1)

            mask = {"u": mask1, "h": mask2}

        return mask

    def __getitem__(self, idx: int):
        inp, dx, dt, target = super().__getitem__(idx)

        mask = self.sample_mask(inp, target)
        return inp, dx, dt, target, mask


class HDF5TimeMaskDataset(HDF5MaskDataset):
    def __init__(self,
                 datapath: str,
                 return_abs_coords: bool,
                 return_grid: bool,
                 input_mean: np.array,
                 input_std: np.array,
                 target_mean: np.array,
                 target_std: np.array,
                 norm_x: bool = False,
                 norm_t: bool = False,
                 norm_input: bool = True,
                 norm_target: bool = True,
                 flip_xy: bool = False,
                 use_theta: bool = False,
                 use_tar_ic: bool = False,
                 dtype=torch.float32,
                 down_factor: int = 1,
                 down_interp: bool = True,
                 is_train: bool = False,
                 add_time_masks: bool = False):
        super().__init__(
                 datapath,
                 return_abs_coords,
                 return_grid,
                 input_mean,
                 input_std,
                 target_mean,
                 target_std,
                 norm_x,
                 norm_t,
                 norm_input,
                 norm_target,
                 flip_xy,
                 use_theta,
                 use_tar_ic,
                 dtype,
                 down_factor,
                 down_interp,
                 is_train)
        self.add_time_masks = add_time_masks

    def get_train_mask(self, inp, target):
        inp_ch = inp.shape[-1]
        target_ch = target.shape[-1]

        var = torch.rand(1)
        if var <= 0.4:
            # target is missing
            inp_mask = torch.zeros_like(inp, dtype=torch.bool)
            tar_mask = torch.ones_like(target, dtype=torch.bool)
        elif var <= 0.8:
            # input is missing
            inp_mask = torch.ones_like(inp, dtype=torch.bool)
            tar_mask = torch.zeros_like(target, dtype=torch.bool)
        else:
            # both vars are available
            inp_mask = torch.zeros_like(inp, dtype=torch.bool)
            tar_mask = torch.zeros_like(target, dtype=torch.bool)

        mask_var = torch.cat([inp_mask, tar_mask], dim=-1)

        # select t max
        res = inp.shape[0]
        t_max1 = res // 2 + torch.randint(res // 2 + 1, (1,))
        t_max2 = res // 2 + torch.randint(res // 2 + 1, (1,))

        mask_res = torch.ones_like(mask_var, dtype=torch.bool)
        mask_res[:t_max1, :, :inp_ch] = False
        mask_res[:t_max2, :, inp_ch:] = False
        mask = mask_var | mask_res

        mask = mask.float()
        return mask

    def sample_mask(self, inp, target):
        # mask = 1 -> missing element
        # mask = 0 -> available element
        if self.is_train:
            mask = self.get_train_mask(inp, target)
        else:
            inp_mask = torch.zeros_like(inp)
            tar_mask = torch.ones_like(target)
            mask1 = torch.cat([inp_mask, tar_mask], dim=-1)

            inp_mask = torch.ones_like(inp)
            tar_mask = torch.zeros_like(target)
            mask2 = torch.cat([inp_mask, tar_mask], dim=-1)

            mask = {"u": mask1, "h": mask2}

            # mask some of the time points for
            if self.add_time_masks:
                res = inp.shape[0]

                inp_mask = torch.zeros_like(inp)
                tar_mask = torch.zeros_like(target)
                inp_mask[int(0.5 * res):] = 1
                tar_mask[int(0.5 * res):] = 1
                mask00 = torch.cat([inp_mask, tar_mask], dim=-1)

                inp_mask = torch.zeros_like(inp)
                tar_mask = torch.ones_like(target)
                inp_mask[int(0.5 * res):] = 1
                mask11 = torch.cat([inp_mask, tar_mask], dim=-1)

                inp_mask = torch.ones_like(inp)
                tar_mask = torch.zeros_like(target)
                tar_mask[int(0.5 * res):] = 1
                mask21 = torch.cat([inp_mask, tar_mask], dim=-1)

                inp_mask = torch.zeros_like(inp)
                tar_mask = torch.ones_like(target)
                inp_mask[int(0.75 * res):] = 1
                mask12 = torch.cat([inp_mask, tar_mask], dim=-1)

                inp_mask = torch.ones_like(inp)
                tar_mask = torch.zeros_like(target)
                tar_mask[int(0.75 * res):] = 1  # observe only each 4th element
                mask22 = torch.cat([inp_mask, tar_mask], dim=-1)

                # mask = {"u": mask1, "h": mask2,
                #         "u_t2": mask11, "h_t2": mask21,
                #         "u_t4": mask12, "h_t4": mask22
                #         }

                mask = {"hu": mask00, "u": mask11, "h": mask21
                        }

        return mask


class HDF5SparseMaskDataset(HDF5MaskDataset):
    def __init__(self,
                 datapath: str,
                 return_abs_coords: bool,
                 return_grid: bool,
                 input_mean: np.array,
                 input_std: np.array,
                 target_mean: np.array,
                 target_std: np.array,
                 norm_x: bool = False,
                 norm_t: bool = False,
                 norm_input: bool = True,
                 norm_target: bool = True,
                 flip_xy: bool = False,
                 use_theta: bool = False,
                 use_tar_ic: bool = False,
                 dtype=torch.float32,
                 down_factor: int = 1,
                 down_interp: bool = True,
                 is_train: bool = False,
                 add_res_masks: bool = False):
        super().__init__(
                 datapath,
                 return_abs_coords,
                 return_grid,
                 input_mean,
                 input_std,
                 target_mean,
                 target_std,
                 norm_x,
                 norm_t,
                 norm_input,
                 norm_target,
                 flip_xy,
                 use_theta,
                 use_tar_ic,
                 dtype,
                 down_factor,
                 down_interp,
                 is_train)
        self.add_res_masks = add_res_masks

    def get_train_mask(self, inp, target):
        inp_ch = inp.shape[-1]
        target_ch = target.shape[-1]

        var = torch.rand(1)
        if var <= 0.33:
            # target is missing
            inp_mask = torch.zeros_like(inp, dtype=torch.bool)
            tar_mask = torch.ones_like(target, dtype=torch.bool)
        elif var <= 0.66:
            # input is missing
            inp_mask = torch.ones_like(inp, dtype=torch.bool)
            tar_mask = torch.zeros_like(target, dtype=torch.bool)
        else:
            # both vars are available
            inp_mask = torch.zeros_like(inp, dtype=torch.bool)
            tar_mask = torch.zeros_like(target, dtype=torch.bool)

        mask_var = torch.cat([inp_mask, tar_mask], dim=-1)

        # select resolution
        res_rand1 = torch.randint(3, ()) + 1
        res_rand2 = torch.randint(3, ()) + 1

        each_x1 = 2 ** (res_rand1 - 1)
        each_x2 = 2 ** (res_rand2 - 1)

        # select t max
        res = inp.shape[0]
        res_cur1 = res // 2 ** (res_rand1 - 1)
        res_cur2 = res // 2 ** (res_rand2 - 1)
        t_max1 = res // 2 + res_rand1 * torch.randint(res_cur1 // 2 + 1, (1,))
        t_max2 = res // 2 + res_rand2 * torch.randint(res_cur2 // 2 + 1, (1,))

        mask_res = torch.ones_like(mask_var, dtype=torch.bool)
        mask_res[:t_max1:each_x1, ::each_x1, :inp_ch] = False
        mask_res[:t_max2:each_x2, ::each_x2, inp_ch:] = False
        mask = mask_var | mask_res

        mask = mask.float()
        return mask

    def sample_mask(self, inp, target):
        # mask = 1 -> missing element
        # mask = 0 -> available element
        if self.is_train:
            mask = self.get_train_mask(inp, target)
        else:
            inp_mask = torch.zeros_like(inp)
            tar_mask = torch.ones_like(target)
            mask1 = torch.cat([inp_mask, tar_mask], dim=-1)

            inp_mask = torch.ones_like(inp)
            tar_mask = torch.zeros_like(target)
            mask2 = torch.cat([inp_mask, tar_mask], dim=-1)

            mask = {"u": mask1, "h": mask2}

            # smaller resolution masks
            if self.add_res_masks:
                # inp_mask = torch.zeros_like(inp)
                # tar_mask = torch.ones_like(target)
                # inp_mask[::2, ::2] = 1
                # mask11 = torch.cat([inp_mask, tar_mask], dim=-1)
                #
                # inp_mask = torch.ones_like(inp)
                # tar_mask = torch.zeros_like(target)
                # tar_mask[::2, ::2] = 1
                # mask21 = torch.cat([inp_mask, tar_mask], dim=-1)
                #
                # inp_mask = torch.zeros_like(inp)
                # tar_mask = torch.ones_like(target)
                # inp_mask[::4, ::4] = 1
                # mask12 = torch.cat([inp_mask, tar_mask], dim=-1)
                #
                # inp_mask = torch.ones_like(inp)
                # tar_mask = torch.zeros_like(target)
                # tar_mask[::4, ::4] = 1
                # mask22 = torch.cat([inp_mask, tar_mask], dim=-1)

                inp_mask = torch.ones_like(inp)
                tar_mask = torch.ones_like(target)
                inp_mask[::2, ::2] = 0
                mask11 = torch.cat([inp_mask, tar_mask], dim=-1)

                inp_mask = torch.ones_like(inp)
                tar_mask = torch.ones_like(target)
                tar_mask[::2, ::2] = 0
                mask21 = torch.cat([inp_mask, tar_mask], dim=-1)

                inp_mask = torch.ones_like(inp)
                tar_mask = torch.ones_like(target)
                inp_mask[::4, ::4] = 0
                mask12 = torch.cat([inp_mask, tar_mask], dim=-1)

                inp_mask = torch.ones_like(inp)
                tar_mask = torch.ones_like(target)
                tar_mask[::4, ::4] = 0  # observe only each 4th element
                mask22 = torch.cat([inp_mask, tar_mask], dim=-1)

                # mask = {"u": mask1, "h": mask2,
                #         "u_d2": mask11, "h_d2": mask21,
                #         "u_d4": mask12, "h_d4": mask22
                #         }

                mask = {"u": mask12, "h": mask22,
                        # "u_d2": mask11, "h_d2": mask21,
                        # "u_d4": mask12, "h_d4": mask22
                        }

        return mask
