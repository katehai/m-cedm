import h5py
import numpy as np
import torch
import torch.nn.functional as F

from datamodules.h5_dataset import HDF5Dataset


class SwpDataset(HDF5Dataset):
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
                 add_t: bool = False,
                 flip_xy: bool = False,
                 use_theta: bool = False,
                 use_tar_ic: bool = False,
                 train_2d: bool = False,
                 dtype=torch.float32,
                 down_factor: int = 1,
                 down_interp: bool = True):
        super().__init__(datapath,
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
        self.add_t = add_t
        self.train_2d = train_2d

        if train_2d:
            self.add_t = True

    def __getitem__(self, idx: int):
        """
        Returns data a batch item.
        Args:
            idx: data index in the h5 file
        Returns:
            torch.Tensor: input state and position of measurements
            torch.Tensor: target state
            torch.Tensor: node_type
            torch.Tensor: positions
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
                x_norm = (x - x.min()) / (x.max() - x.min())  # if irregular data, should pass in x_min and x_max
            else:
                x_norm = x

            if self.norm_t:
                t_norm = (t - t.min()) / (t.max() - t.min())
            else:
                t_norm = t

            if not self.return_abs_coords:
                x = torch.diff(x)[0]
                t = torch.diff(t)[0]  # not used for this model as an input

            t_grid, x_grid = torch.meshgrid(t_norm, x_norm, indexing='ij')
            if self.add_t:
                t_grid = t_grid.unsqueeze(-1)
                inp = torch.cat([inp, t_grid], dim=-1)

            x_grid = x_grid.unsqueeze(-1)
            inp = torch.cat([inp, x_grid], dim=-1)

            # downsample the input
            if self.down_factor > 1:
                each_x = 2 ** (self.down_factor - 1)

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
                    inp = F.interpolate(inp.permute(2, 1, 0).unsqueeze(0), scale_factor=1 / each_x, mode='bilinear',
                                        align_corners=False)
                    inp = inp.squeeze(0).permute(2, 1, 0)

                    x = F.interpolate(x.unsqueeze(0).unsqueeze(0), scale_factor=1 / each_x, mode='linear',
                                      align_corners=False).squeeze(0).squeeze(0)
                    t = F.interpolate(t.unsqueeze(0).unsqueeze(0), scale_factor=1 / each_x, mode='linear',
                                      align_corners=False).squeeze(0).squeeze(0)

                    target = F.interpolate(target.permute(2, 1, 0).unsqueeze(0), scale_factor=1 / each_x,
                                           mode='bilinear', align_corners=False)
                    target = target.squeeze(0).permute(2, 1, 0)

            if self.train_2d:
                inp = inp.reshape(1, -1, inp.shape[-1])  # 1, n_time x n_x, in_channels
                target = target.reshape(1, -1, target.shape[-1])  # 1, n_time x n_x, out_channels

                t_grid_offset, x_grid_offset = torch.meshgrid(t - t.min(), x - x.min(), indexing='ij')
                offset_pos = torch.stack([t_grid_offset, x_grid_offset], dim=-1)  # n_time, n_x, 2
                offset_pos = offset_pos.reshape(-1, offset_pos.shape[-1])

                node_type = torch.zeros_like(t_grid_offset, dtype=torch.long)  # 0: domain point, 1 - a boundary point
                node_type[0] = 1
                node_type[-1] = 1
                node_type[:, 0] = 1
                node_type[:, -1] = 1
                node_type = node_type.reshape(-1, 1)
            else:
                offset_pos = x - x.min()
                offset_pos = offset_pos.unsqueeze(-1)

                node_type = torch.zeros_like(x, dtype=torch.long)  # 0: domain point, 1 - a boundary point
                node_type[0] = 1
                node_type[-1] = 1
                node_type = node_type.unsqueeze(-1)

            n_time = len(t)
            return inp, target, node_type, offset_pos, n_time


class SwpTimePredDataset(HDF5Dataset):
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
                 add_t: bool = False,
                 flip_xy: bool = False,
                 use_theta: bool = False,
                 use_tar_ic: bool = False,
                 dtype=torch.float32,
                 down_factor: int = 1,
                 down_interp: bool = True,
                 n_history: int = 64):
        super().__init__(datapath,
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
        self.add_t = add_t
        self.n_history = n_history

    def __getitem__(self, idx: int):
        """
        Returns data a batch item.
        Args:
            idx: data index in the h5 file
        Returns:
            torch.Tensor: input state and position of measurements
            torch.Tensor: target state
            torch.Tensor: node_type for input
            torch.Tensor: node_type for target
            torch.Tensor: positions for inputs
            torch.Tensor: positions for targets
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

            x = torch.tensor(seed_group["grid"]["x"][:], dtype=self.dtype)
            t = torch.tensor(seed_group["grid"]["t"][:], dtype=self.dtype)

            if len(t) > len(inp):
                t = t[:-1]  # some simulators store an extra step in the end

            if self.norm_x:
                x_norm = (x - x.min()) / (x.max() - x.min())  # if irregular data, should pass in x_min and x_max
            else:
                x_norm = x

            if self.norm_t:
                t_norm = (t - t.min()) / (t.max() - t.min())
            else:
                t_norm = t

            if not self.return_abs_coords:
                x = torch.diff(x)[0]
                t = torch.diff(t)[0]  # not used for this model as an input

            t_grid, x_grid = torch.meshgrid(t_norm, x_norm, indexing='ij')
            inp_dim = inp.shape[-1]
            tar_dim = target.shape[-1]
            if self.add_t:
                t_grid = t_grid.unsqueeze(-1)
                inp = torch.cat([inp, t_grid], dim=-1)

            x_grid = x_grid.unsqueeze(-1)
            inp = torch.cat([inp, x_grid], dim=-1)

            # downsample the input
            if self.down_factor > 1:
                each_x = 2 ** (self.down_factor - 1)

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
                    inp = F.interpolate(inp.permute(2, 1, 0).unsqueeze(0), scale_factor=1 / each_x, mode='bilinear',
                                        align_corners=False)
                    inp = inp.squeeze(0).permute(2, 1, 0)

                    x = F.interpolate(x.unsqueeze(0).unsqueeze(0), scale_factor=1 / each_x, mode='linear',
                                      align_corners=False).squeeze(0).squeeze(0)
                    t = F.interpolate(t.unsqueeze(0).unsqueeze(0), scale_factor=1 / each_x, mode='linear',
                                      align_corners=False).squeeze(0).squeeze(0)

                    target = F.interpolate(target.permute(2, 1, 0).unsqueeze(0), scale_factor=1 / each_x,
                                           mode='bilinear', align_corners=False)
                    target = target.squeeze(0).permute(2, 1, 0)

            # Concatenate the input and target
            n_history = self.n_history
            state = torch.cat([inp[..., :inp_dim], target, inp[..., inp_dim:]], dim=-1)
            inp = state[:n_history]
            target = state[n_history:, :, :inp_dim+tar_dim]
            n_time = len(target)

            inp = inp.reshape(1, -1, inp.shape[-1])  # 1, n_time x n_x, in_channels
            target = target.reshape(1, -1, target.shape[-1])  # 1, n_time x n_x, out_channels

            t_grid_offset, x_grid_offset = torch.meshgrid(t - t.min(), x - x.min(), indexing='ij')
            offset_pos = torch.stack([t_grid_offset, x_grid_offset], dim=-1)  # n_time, n_x, 2
            offset_pos_inp = offset_pos[:n_history]
            offset_pos_target = offset_pos[n_history:]
            offset_pos_inp = offset_pos_inp.reshape(-1, offset_pos_inp.shape[-1])
            offset_pos_target = offset_pos_target.reshape(-1, offset_pos_target.shape[-1])

            node_type = torch.zeros_like(t_grid_offset, dtype=torch.long)  # 0: domain point, 1 - a boundary point
            node_type[0] = 1
            node_type[-1] = 1
            node_type[:, 0] = 1
            node_type[:, -1] = 1
            node_type_inp = node_type[:n_history]
            node_type_target = node_type[n_history:]
            node_type_inp = node_type_inp.reshape(-1, 1)
            node_type_target = node_type_target.reshape(-1, 1)

            return inp, target, node_type_inp, node_type_target, offset_pos_inp, offset_pos_target, n_time
