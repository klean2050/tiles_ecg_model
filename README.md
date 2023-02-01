

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

Tracking Individual Performance with Sensors (TILES) is a project holding multimodal data sets for the analysis of stress, task performance, behavior, and other factors pertaining to professionals engaged in a high-stress workplace environments. Biological, environmental, and contextual data was collected from hospital nurses, staff, and medical residents both in the workplace and at home over time. Labels of human experience were collected using a variety of psychologically validated questionnaires sampled on a daily basis at different times during the day. In this work, we utilize the TILES ECG data from the publicly available dataset, that we download from [here](https://tiles-data.isi.edu/).

## Pre-Training Framework (TBD)

#### Input ECG data

Each TILES participant has their ECG recorded for 15 seconds every 5 minutes during their work hours, for a total of 10 weeks. In this experiment we consider a subset of 69 subjects, and for each of them we extract all available 15-sec ECG segments. We normalize the data per subject before feeding them to the model.

#### Augmentation Strategy

We pre-train the model in a self-supervised manner, through contrastive learning. In specific, we consider 2 augmented views of an input ECG signal and we train the network to identify these pairings among all possible pairs in a training batch. To augment the ECG samples we use the [PyTorch ECG Augmentations](https://github.com/klean2050/ecg-augmentations) package. First, the input ECG is randomly cropped to 10 seconds and a series of masks and signal transformations are randomly applied based on a set probability. This is applied online, twice during training, to produce the 2 augmented views.

#### Backbone \& Objective

A lightweight ResNet encoder is used to extract latent representations from the augmented data inputs. We use a light architecture of 8 blocks and 16 filters at the first block, in order to abide by the domain literature and make the model applicable to real-time settings. The 256D output embeddings of a pair of augmented samples are projected to a 128D latent space, where all samples within a batch are contrasted using the NT-Xent loss, adapted from the SimCLR study. With this loss, the model is forced to identify the underlying association between augmented versions of the same sample. The network is trained for about 60K steps using an AdamW optimizer (learning rate 0.001 on batches of 128 samples).

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
