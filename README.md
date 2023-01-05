

<div align="center">

# TILES ECG: In-the-Wild ECG Pre-Training
This repo contains code implementation as well as trained models for ECG data analysis.
  
</div>

## Installation

We recommend using a conda environment with ``Python >= 3.9`` :
```
conda create -n tiles_ecg python=3.9
conda activate tiles_ecg
```
Clone the repository and install the dependencies:
```
git clone https://github.com/klean2050/tiles_ecg_model
cd tiles_ecg_model && pip install -e .
```

## Project Structure

```
tiles_ecg_model/
├── setup.py             # package installation script
├── config/              # configuration files for each train session
└── src/                 # main project directory
    ├── loaders/             # pytorch dataset classes
    ├── models/              # backbone neural network models
    ├── scripts/             # preprocess, training and evaluation
    ├── trainers/            # lightning classes for each train session
    └── utils/               # miscellaneous scripts and methods
```

## TILES Dataset (TBD)



## Pre-Training Framework (TBD)



## Results & Checkpoints (TBD)

To view results in TensorBoard run:
```
tensorboard --logdir ./runs
```

## Acknowledgements

In this study we made use of the following repositories:

* [VCMR](https://github.com/klean2050/VCMR)


## Authors
* [Kleanthis Avramidis](https://klean2050.github.io): PhD Student in Computer Science, USC SAIL
