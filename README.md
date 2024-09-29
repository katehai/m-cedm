# Mixed conditional EDM 
The code repository for the paper "Diffusion models as probabilistic neural operators for recovering unobserved states of dynamical systems"

**Arxiv**: [link](https://arxiv.org/abs/2405.07097)

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


# Citation

@inproceedings{haitsiukevich2024diffusion,
  title={Diffusion models as probabilistic neural operators for recovering unobserved states of dynamical systems},
  author={Haitsiukevich, Katsiaryna and Poyraz, Onur and Marttinen, Pekka and Ilin, Alexander},
  booktitle={2024 IEEE 34th International Workshop on Machine Learning for Signal Processing (MLSP)},
  year={2024},
  organization={IEEE}
}