

<div align="center">

# WildECG: Ubiquitous ECG Pre-Training
This repo contains code implementation as well as trained models for ECG data analysis.
  
</div>

## Installation

We recommend using a conda environment with ``Python >= 3.9`` :

```bash
conda create -n tiles_ecg python=3.9
conda activate tiles_ecg
```

Clone the repository and install the dependencies:

```bash
git clone https://github.com/klean2050/tiles_ecg_model
pip install -e tiles_ecg_model
```

You will also need the ``ecg-augmentations`` library:

```bash
git clone https://github.com/klean2050/ecg-augmentations
pip install -e ecg-augmentations
```

## Project Structure

```bash
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

Tracking Individual Performance with Sensors (TILES) is a project holding multimodal data sets for the analysis of stress, task performance, behavior, and other factors pertaining to professionals engaged in a high-stress workplace environments. Biological, environmental, and contextual data was collected from hospital nurses, staff, and medical residents both in the workplace and at home over time. Labels of human experience were collected using a variety of psychologically validated questionnaires sampled on a daily basis at different times during the day. In this work, we utilize the TILES ECG data from the publicly available dataset, that we download from [here](https://tiles-data.isi.edu/).

## Pre-Training Framework

### Input ECG data

Each TILES participant has their ECG recorded for 15 seconds every 5 minutes during their work hours, for a total of 10 weeks. In this experiment we consider a subset of 69 subjects, and for each of them we extract all available 15-sec ECG segments. We normalize the data per subject before feeding them to the model.

To preprocess TILES ECG, navigate to the root directory and run the following command:

```bash
python src/scripts/preprocess.py
```

### pre-Training \& Fine-tuning

TBD

## Results & Checkpoints

To view results of your experiments in TensorBoard run:

```bash
tensorboard --logdir ./runs
```

## Acknowledgements

In this study we made use of the following repositories:

* [VCMR](https://github.com/klean2050/VCMR) --> repo template
* [ecg-augmentations](https://github.com/klean2050/ecg-augmentations)

## Citation

TBD: The accompanying paper has been submitted to IEEE Journal of Biomedical and Health Informatics (2023).
