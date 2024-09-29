# Mixed conditional EDM 
The code repository for the paper "Diffusion models as probabilistic neural operators for recovering unobserved states of dynamical systems"

**Arxiv**: [link](https://arxiv.org/abs/2405.07097)


## Proposed method

We proposed training a single diffusion-based generative model for several tasks of interest in dynamical system modeling.
During the training we sample which states are observed and which are unobserved as well as a number of time steps available for observed states.
The model is trained to recover the unobserved states conditioned on the observed ones and tested on the tasks presented below.
<img src="img/mcedm.jpg" alt="img/swe_per1.png" width="700"/>

## Data generation


The following commands can be used to generate the data for the Shallow Water Equations (SWE) equation.


1. Periodic initial conditions

```bash 
python generate/gen_swe_period_1d.py
```

or to explicitly set the generation config with hydra

```bash 
python generate/gen_swe_period_1d.py --config-path configs --config-name sw_periodic_1d
```

<img src="img/swe_per1.png" alt="img/swe_per1.png" width="400"/>
<img src="img/swe_per2.png" alt="img/swe_per2.png" width="400"/>

2. Gaussian perturbation initial conditions (or Dam Break scenario)

```bash 
python generate/gen_dam_break_1d.py
```

or 

```bash 
python generate/gen_dam_break_1d.py --config-path configs --config-name sw_perturb_1d
```

Example simulations are shown below:

<img src="img/swe_perturb1.png" alt="img/swe_perturb1.png" width="400"/>
<img src="img/swe_perturb2.png" alt="img/swe_perturb2.png" width="400"/>

The equation is solved using [PyClaw package](https://www.clawpack.org/pyclaw/started.html) that requires a separate installation. The data is saved in the `data` folder.


### Preprocessing

The generated data is preprocessed by removing the last time step in order to make inputs squared as well as by saving training set statistics for normalization of neural network inputs.
The following command can be used to replicate the preprocessing:

```bash
python preprocess_data.py --datafolder data/1D_swp_128 --datafolder_test data/1D_swp_128 --trainfile 1D_swp_128_train.h5 --testfile 1D_swp_128_test.h5 --num_steps 128 --change_num_steps
```

### Darcy flow data

The data for the Darcy flow equation is taken from [PDEBench](https://github.com/pdebench/PDEBench) data collection, file `2D_DarcyFlow_beta1.0_Train.hdf5` from [here](https://darus.uni-stuttgart.de/file.xhtml?fileId=133217&version=8.0).
and preprocessed by the following command:

```bash
python preprocess_darcy.py

```


# Citation

```
@inproceedings{haitsiukevich2024diffusion,
  title={Diffusion models as probabilistic neural operators for recovering unobserved states of dynamical systems},
  author={Haitsiukevich, Katsiaryna and Poyraz, Onur and Marttinen, Pekka and Ilin, Alexander},
  booktitle={2024 IEEE 34th International Workshop on Machine Learning for Signal Processing (MLSP)},
  year={2024},
  organization={IEEE}
}
```