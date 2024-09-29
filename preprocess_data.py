import numpy as np
import os
import h5py
import argparse


def calc_stats_from_list(values):
    v_array = np.array(values, dtype=np.float64)
    s_dim = v_array.shape[-1]
    v_array = v_array.reshape(-1, s_dim)
    stats = {'mean': np.mean(v_array, axis=0), 'std': np.std(v_array, axis=0),
             'min': np.min(v_array, axis=0), 'max': np.max(v_array, axis=0)}
    return stats


def calc_stats(filepath):
    f = h5py.File(filepath, 'r')
    inps, tars = [], []
    keys = list(f.keys())
    mean0 = 0.0
    for key in keys:
        sample = f[key]
        inp = sample['data']['input'][:]
        tar = sample['data']['target'][:]
        inps.append(inp)
        tars.append(tar)
        mean0 += np.mean(inp[..., 0])

    f.close()

    inp_stats = calc_stats_from_list(inps)
    tar_stats = calc_stats_from_list(tars)

    return inp_stats, tar_stats


def get_std(filepath, inp_mean, tar_mean):
    s_dim = len(inp_mean)
    inp_var, tar_var = np.zeros(s_dim), np.zeros(s_dim)
    f = h5py.File(filepath, 'r')

    keys = list(f.keys())
    n_samples = len(keys)
    for key in keys:
        sample = f[key]
        inp = sample['data']['input'][:]
        tar = sample['data']['target'][:]

        inp = inp.reshape(-1, s_dim)
        tar = tar.reshape(-1, s_dim)

        # mean over (x - mean)**2
        inp_var += np.mean((inp - inp_mean)**2, axis=0)
        tar_var += np.mean((tar - tar_mean)**2, axis=0)

    f.close()

    inp_std = np.sqrt(inp_var / n_samples)
    tar_std = np.sqrt(tar_var / n_samples)
    return inp_std, tar_std


def init_stats(data):
    s_dim = data.shape[-1]
    sample = data.reshape(-1, s_dim)
    d_mean = np.zeros(s_dim)
    d_std = np.zeros_like(d_mean)
    d_min = np.amin(sample, axis=0)
    d_max = np.amax(sample, axis=0)

    inp_stats = {'mean': d_mean, 'std': d_std, 'min': d_min, 'max': d_max}
    return inp_stats


def calc_stats_sequential(filepath):
    f = h5py.File(filepath, 'r')
    keys = list(f.keys())
    n_samples = len(keys)

    sample0 = f[keys[0]]
    inp0 = sample0['data']['input'][:]
    tar0 = sample0['data']['target'][:]
    inp_stats = init_stats(inp0)
    tar_stats = init_stats(tar0)

    for key in keys:
        sample = f[key]
        inp = sample['data']['input'][:]
        tar = sample['data']['target'][:]

        upd_stats_parallel(inp_stats, inp)
        upd_stats_parallel(tar_stats, tar)

    f.close()

    inp_stats['mean'] /= n_samples
    tar_stats['mean'] /= n_samples

    inp_stats['std'], tar_stats['std'] = get_std(filepath, inp_stats['mean'], tar_stats['mean'])

    return inp_stats, tar_stats


def upd_stats_parallel(stats, sample):
    s_dim = sample.shape[-1]
    sample = sample.reshape(-1, s_dim)

    stats['mean'] += np.mean(sample, axis=0)
    stats['min'] = np.minimum(np.amin(sample, axis=0), stats['min'])
    stats['max'] = np.maximum(np.amax(sample, axis=0), stats['max'])


def compare_stats(stats1, stats2):
    for key, value1 in stats1.items():
        value2 = stats2[key]
        if np.any(np.abs(value1 - value2) > 1e-5):
            print(f"Values for {key} are not equal: {value1} != {value2}")
            print(f"difference is {np.abs(value1 - value2)}")
        else:
            print(f"Values for {key} are equal")
    return


def check_sequential_stats_calc(train_filepath):
    print("Calculate stats for train data")
    inp_stats1, tar_stats1 = calc_stats(train_filepath)
    print("Stats calculated by loading the whole set")
    print("Input stats: ", inp_stats1)
    print("Target stats: ", tar_stats1)
    print()
    inp_stats2, tar_stats2 = calc_stats_sequential(train_filepath)
    print("Stats calculated by loading one sample at a time")
    print("Input stats: ", inp_stats2)
    print("Target stats: ", tar_stats2)
    print()
    print("Compare stats")
    print("Input stats: ")
    compare_stats(inp_stats1, inp_stats2)
    print("Target stats: ")
    compare_stats(tar_stats1, tar_stats2)


def load_stats_from_file(filepath):
    f = h5py.File(filepath, 'r')
    inp_stats = {}
    tar_stats = {}
    stats_keys = ['mean', 'std', 'min', 'max']
    for stat_key in stats_keys:
        inp_stats[stat_key] = f.attrs[f"inp_{stat_key}"]
        tar_stats[stat_key] = f.attrs[f"tar_{stat_key}"]

    f.close()
    return inp_stats, tar_stats


def add_stats(f, stats, tag):
    for key, value in stats.items():
        f.attrs[f"{tag}_{key}"] = value


def add_stats_to_file(filepath, inp_stats, tar_stats):
    f = h5py.File(filepath, 'r+')
    add_stats(f, inp_stats, "inp")
    add_stats(f, tar_stats, "tar")

    f.close()


def adjust_num_steps(filepath, num_steps):
    f = h5py.File(filepath, 'r+')
    for key in f.keys():
        sample = f[key]

        # override input and target for each sample to have a truncated number of steps
        # since datasets have different sizes we need to remove the old one first and to add a new one
        inp = sample['data']['input'][:]
        tar = sample['data']['target'][:]

        if len(inp) == inp.shape[1] and len(tar) == tar.shape[1]:
            print("The number of steps is not adjusted because inputs and targets are quadratic")
            print(f"Inp size is {inp.shape} and tar size is {tar.shape}")
            continue

        if num_steps == -1:
            num_steps = len(inp) - 1

        if len(inp) < num_steps:
            print(f"Number of steps is too large for sample {key}: {len(inp)} < {num_steps}")
            continue

        inp = inp[:num_steps]
        tar = tar[:num_steps]

        del sample['data']
        sample.create_dataset('data/input', data=inp)
        sample.create_dataset('data/target', data=tar)

    f.close()


def process_data(datafolder, datafolder_test, trainfile, testfile, num_steps, change_num_steps):
    train_filepath = os.path.join(datafolder, trainfile)

    if change_num_steps:
        # save correct number of steps
        print(f"Adjust the number of time steps to {num_steps}, removing the last one if -1")
        adjust_num_steps(train_filepath, num_steps)

    ##  calculate and save stats
    # check_sequential_stats_calc(train_filepath)

    inp_stats, tar_stats = calc_stats_sequential(train_filepath)
    print("Input stats: ", inp_stats)
    print("Target stats: ", tar_stats)
    print()

    # add stats to the files
    add_stats_to_file(train_filepath, inp_stats, tar_stats)

    if testfile is not None and testfile != "":
        test_filepath = os.path.join(datafolder_test, testfile)

        if change_num_steps:
            adjust_num_steps(test_filepath, num_steps)
        add_stats_to_file(test_filepath, inp_stats, tar_stats)


def process_test_file(datafolder, datafolder_test, trainfile, testfile, num_steps, change_num_steps):
    train_filepath = os.path.join(datafolder, trainfile)

    ## to double-check
    # inp_stats, tar_stats = calc_stats_sequential(train_filepath)
    # print("Input stats: ", inp_stats)
    # print("Target stats: ", tar_stats)
    # print()

    inp_stats, tar_stats = load_stats_from_file(train_filepath)
    print("Input stats: ", inp_stats)
    print("Target stats: ", tar_stats)
    print()

    if testfile is not None and testfile != "":
        test_filepath = os.path.join(datafolder_test, testfile)

        if change_num_steps:
            adjust_num_steps(test_filepath, num_steps)
        add_stats_to_file(test_filepath, inp_stats, tar_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processing of datasets data')
    parser.add_argument('--datafolder', type=str, default="data", help='folder with data files')
    parser.add_argument('--datafolder_test', type=str, default="", help='folder with data files')
    parser.add_argument('--trainfile', type=str, default="1d_swp_train.h5", help='train data file')
    parser.add_argument('--testfile', type=str, default="", help='test data file')
    parser.add_argument('--num_steps', type=int, default=-1, help='number of steps to use')
    parser.add_argument('--change_num_steps', action='store_true', help='change number of steps in the dataset')
    parser.add_argument('--test_only', action='store_true', help='preprocesses only the test file')
    args = parser.parse_args()

    datafolder_test = args.datafolder if args.datafolder_test == "" else args.datafolder_test

    if args.test_only:
        process_test_file(args.datafolder, datafolder_test, args.trainfile, args.testfile, args.num_steps,
                          args.change_num_steps)
    else:
        process_data(args.datafolder, datafolder_test, args.trainfile, args.testfile, args.num_steps,
                     args.change_num_steps)
