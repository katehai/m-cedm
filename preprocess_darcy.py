import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch

from models.losses import LpLoss


def darcy_loss1(a, u):
    batchsize = u.shape[0]
    size = u.shape[1]
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    lploss = LpLoss()

    Du = darcy_loss(a, u)
    # f = torch.ones(Du.shape, device=u.device)
    Du = torch.tensor(Du, dtype=torch.float32)
    f = torch.ones_like(Du)
    abs_loss = torch.mean(torch.abs(Du - f)).numpy()
    loss_f = lploss(Du, f)

    return abs_loss, loss_f


def darcy_loss(a, u, D=1):
    batchsize = u.shape[0]
    size = u.shape[1]
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    dx = D / (size - 1)
    dy = dx

    # ux: (batch, size-2, size-2)
    ux = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
    uy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dy)

    a = a[:, 1:-1, 1:-1]
    aux = a * ux
    auy = a * uy
    auxx = (aux[:, 2:, 1:-1] - aux[:, :-2, 1:-1]) / (2 * dx)
    auyy = (auy[:, 1:-1, 2:] - auy[:, 1:-1, :-2]) / (2 * dy)
    Du = - (auxx + auyy)
    return Du


def darcy_loss_pde_bench(a, u, D=1, mean=True, clip=False):
    batchsize = u.shape[0]
    size = u.shape[1]
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    dx = D / size
    dy = dx

    # ux: (batch, size-2, size-2)
    ux = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
    uy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dy)

    a = a[:, 1:-1, 1:-1]
    aux = a * ux
    auy = a * uy
    auxx = (aux[:, 2:, 1:-1] - aux[:, :-2, 1:-1]) / (2 * dx)
    auyy = (auy[:, 1:-1, 2:] - auy[:, 1:-1, :-2]) / (2 * dy)
    Du = - (auxx + auyy)

    pde_loss = np.abs(Du - 1.) # ** 2

    if clip:
        pde_loss = np.clip(pde_loss, 0, 1.)

    if mean:
        pde_loss = np.mean(pde_loss)

    return pde_loss


def plot_darcy_sample(consts, targets, folder='plots/darcy', name='darcy.png'):
    n_samples = consts.shape[0]

    fig, axs = plt.subplots(n_samples, 2, figsize=(2*2, 2*n_samples), squeeze=False, sharex=True, sharey=True)
    for i in range(n_samples):
        const = consts[i, :, :]
        target = targets[i, :, :]
        axs[i, 0].imshow(const, cmap='jet')
        axs[i, 1].imshow(target, cmap='jet')
        # print("Const min/max: ", np.min(const), np.max(const))
        # print("Target min/max: ", np.min(target), np.max(target))

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(os.path.join(folder, name), bbox_inches='tight')
    plt.close(fig)


def plot_darcy_sample_pde_loss(consts, targets, pde_losses, folder='plots/darcy', name='darcy_pde.png'):
    n_samples = consts.shape[0]

    fig, axs = plt.subplots(n_samples, 3, figsize=(2*3, 2*n_samples), squeeze=False, sharex=True, sharey=True)
    for i in range(n_samples):
        const = consts[i, :, :]
        target = targets[i, :, :]
        axs[i, 0].imshow(const, cmap='jet')
        axs[i, 1].imshow(target, cmap='jet')
        axs[i, 2].imshow(pde_losses[i, :, :], cmap='jet')
        print("Const min/max: ", np.min(const), np.max(const))
        print("Target min/max: ", np.min(target), np.max(target))

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(os.path.join(folder, name), bbox_inches='tight')
    plt.close(fig)


def load_darcy_pdebench():
    # load darcy flow data from PDEBench
    filepath = 'data/2D_DarcyFlow_beta1.0_Train.hdf5'
    with h5py.File(filepath, 'r') as f:
        const = f['nu'][:]
        target = f['tensor'][:][:, 0]
        # x = f['x-coordinate'][:]
        # y = f['y-coordinate'][:]

        # print(f.keys())
        # print(target.shape)
        # print(const.shape)
        # print(x.shape, y.shape)
        # print(np.min(x), np.max(x))
        # print(np.min(y), np.max(y))

        idx = np.random.randint(0, const.shape[0], 10)
        plot_darcy_sample(const[idx], target[idx])
        # plot_darcy_sample(const[:10], target[:10])

    # print(const.shape, target.shape)
    return const, target


def load_darcy_fno(filepath='data/darcy_train.h5'):
    with h5py.File(filepath, 'r') as f:
        inputs = []
        targets = []
        for seed in f.keys():
            inp = f[seed]['data']["input"][:]
            target = f[seed]['data']["target"][:]
            inputs.append(inp)
            targets.append(target)

        inputs = np.stack(inputs, axis=0)
        targets = np.stack(targets, axis=0)

        print("Shapes: ", inputs.shape, targets.shape)

        idx = np.random.randint(0, inputs.shape[0], 10)
        plot_darcy_sample(inputs[idx], targets[idx], name='darcy_fno.png')
    return inputs, targets


def analyze_pde_error(consts_a, targets_u, pde_loss):
    print('PDE loss')
    pde_loss_mean = np.mean(pde_loss, axis=(1, 2))
    idx_max = np.argmax(pde_loss_mean)
    print("Max idx: ", idx_max)
    pde_loss_max = pde_loss[idx_max]
    print("Max loss value is ", np.min(pde_loss_max), np.max(pde_loss_max))
    plot_darcy_sample(consts_a[idx_max:idx_max + 1], targets_u[idx_max:idx_max + 1], name='darcy_max.png')
    plot_darcy_sample_pde_loss(consts_a[idx_max:idx_max + 1], targets_u[idx_max:idx_max + 1],
                               pde_loss[idx_max:idx_max + 1],
                               name='darcy_max_pde.png')
    print("PDE loss mean: ", np.min(pde_loss_mean), np.max(pde_loss_mean))
    print(np.unique(consts_a[idx_max:idx_max + 1]))
    print("Mean: ", np.mean(pde_loss))


def analyze_pde_error_clip(consts_a, targets_u, pde_loss):
    pde_loss = np.clip(pde_loss, -1, 1)

    print("Clipped PDE loss analysis")
    analyze_pde_error(consts_a, targets_u, pde_loss)


def load_data_analyze_pde_error():
    consts_a, targets_u = load_darcy_pdebench()
    pde_loss = np.abs(darcy_loss(consts_a, targets_u) - 1.)
    analyze_pde_error(consts_a, targets_u, pde_loss)
    analyze_pde_error_clip(consts_a, targets_u, pde_loss)

    abs_loss, pde_loss_lp = darcy_loss1(consts_a, targets_u)
    print("PDE loss lp: ", pde_loss_lp)
    print("PDE abs loss:", abs_loss)

    # ## FNO pde loss
    consts_a_fno, targets_u_fno = load_darcy_fno(filepath='data/darcy_train_full.h5')
    pde_loss_fno = np.abs(darcy_loss(consts_a_fno, targets_u_fno) - 1.)

    print("FNO")
    analyze_pde_error(consts_a_fno, targets_u_fno, pde_loss_fno)

    abs_loss_fno, pde_loss_lp_fno = darcy_loss1(consts_a_fno, targets_u_fno)
    print("PDE loss lp FNO: ", pde_loss_lp_fno)
    print("PDE abs loss FNO:", abs_loss_fno)


def add_stats(f, stats, tag):
    for key, value in stats.items():
        f.attrs[f"{tag}_{key}"] = value


def get_stats(values):
    stats = {'mean': np.mean(values), 'std': np.std(values), 'min': np.min(values), 'max': np.max(values)}
    return stats


def create_h5py(inp, target, inp_stats, tar_stats, x, t, save_path):
    with h5py.File(save_path, 'w') as f:
        n_samples = inp.shape[0]
        for i in range(n_samples):
            f.create_dataset(f"{i}/data/input", data=inp[i])
            f.create_dataset(f"{i}/data/target", data=target[i])

            f.create_dataset(f"{i}/grid/x", data=x)
            f.create_dataset(f"{i}/grid/t", data=t)

        add_stats(f, inp_stats, "inp")
        add_stats(f, tar_stats, "tar")

        f.close()


def change_format_h5py_darcy(n_train=1000, save_file=True):
    # load data from PDE bench, split by train and test sets
    filepath = 'data/2D_DarcyFlow_beta1.0_Train.hdf5'
    with h5py.File(filepath, 'r') as f:
        const = f['nu'][:][..., None]
        target = f['tensor'][:][:][:, 0][..., None]
        x = f['x-coordinate'][:]
        y = f['y-coordinate'][:]

    # train / test split
    train_idx = np.arange(0, n_train, dtype=np.int64)
    test_idx = np.arange(9000, 9100, dtype=np.int64)
    const_train = const[train_idx]
    target_train = target[train_idx]

    const_test = const[test_idx]
    target_test = target[test_idx]

    print(f"Train shape: ", const_train.shape, target_train.shape)
    print(f"Test shape: ", const_test.shape, target_test.shape)

    # check PDE error for train and for test
    pde_loss_train = darcy_loss_pde_bench(const_train, target_train)
    pde_loss_test = darcy_loss_pde_bench(const_test, target_test)

    print(f"PDE loss train: {pde_loss_train}")
    print(f"PDE loss test: {pde_loss_test}")

    if save_file:
        inp_stats = get_stats(const_train)
        tar_stats = get_stats(target_train)

        name_postfix = "" if n_train == 1000 else f"_{n_train}"
        create_h5py(const_train, target_train, inp_stats, tar_stats, x, y, f'data/1D_darcy_128/darcy_train{name_postfix}.h5')
        create_h5py(const_test, target_test, inp_stats, tar_stats, x, y, f'data/1D_darcy_128/darcy_test{name_postfix}.h5')

    return const, target


if __name__ == "__main__":
    # change the format of the file to be the same as shallow water
    # load_data_analyze_pde_error()

    change_format_h5py_darcy(save_file=False)
