import os


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def override_data_folders(cfg_datamodule, dataroot, system, res=128, n_train=1000):
    print(f"PDE error: system = {system}")
    train_res = 128
    if system == "swe":
        if n_train == 1000:
            train_file = f"1D_swp_{train_res}/1D_swp_{train_res}_train.h5"
        else:
            train_file = f"1D_swp_{train_res}/1D_swp_{train_res}_train_{n_train}.h5"
        val_file = f"1D_swp_{res}/1D_swp_{res}_test.h5"
        test_file = f"1D_swp_{res}/1D_swp_{res}_test.h5"
    elif system == "swe_per":
        train_file = f"1D_swp_{train_res}_per/1D_swp_{train_res}_per_train.h5"
        val_file = f"1D_swp_{res}_per/1D_swp_{res}_per_test.h5"
        test_file = f"1D_swp_{res}_per/1D_swp_{res}_per_test.h5"
    elif system == "darcy":
        train_file = f"1D_darcy_128/darcy_train.h5"
        val_file = f"1D_darcy_128/darcy_test.h5"
        test_file = f"1D_darcy_128/darcy_test.h5"
    else:
        # use swe system by default
        train_file = f"1D_swp_{train_res}/1D_swp_{train_res}_train.h5"
        val_file = f"1D_swp_{res}/1D_swp_{res}_test.h5"
        test_file = f"1D_swp_{res}/1D_swp_{res}_test.h5"

    cfg_datamodule.train_path = os.path.join(dataroot, train_file)
    cfg_datamodule.val_path = os.path.join(dataroot, val_file)
    cfg_datamodule.test_path = os.path.join(dataroot, test_file)

    return cfg_datamodule
