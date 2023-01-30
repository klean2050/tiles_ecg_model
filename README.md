

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

## TILES Dataset

Tracking Individual Performance with Sensors (TILES) is a project holding multimodal data sets for the analysis of stress, task performance, behavior, and other factors pertaining to professionals engaged in a high-stress workplace environments. Biological, environmental, and contextual data was collected from hospital nurses, staff, and medical residents both in the workplace and at home over time. Labels of human experience were collected using a variety of psychologically validated questionnaires sampled on a daily basis at different times during the day. In this work, we utilize the TILES ECG data from the publicly available dataset, that we download from [here](https://tiles-data.isi.edu/) for a subset of 69 subjects.

## Pre-Training Framework (TBD)

...

## Fine-Tuning Framework (TBD)

...

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
