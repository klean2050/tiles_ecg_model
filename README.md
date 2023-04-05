

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
pip install -e tiles_ecg_model
```
You will also need the ``ecg-augmentations`` library:
```
git clone https://github.com/klean2050/ecg-augmentations
pip install -e ecg-augmentations
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

## Pre-Training Framework

### Input ECG data

Each TILES participant has their ECG recorded for 15 seconds every 5 minutes during their work hours, for a total of 10 weeks. In this experiment we consider a subset of 69 subjects, and for each of them we extract all available 15-sec ECG segments. We normalize the data per subject before feeding them to the model.

To preprocess TILES ECG, navigate to the root directory and run the following command:
```
python src/scripts/preprocess.py
```

### Augmentation Strategy

We pre-train the model in a self-supervised manner, through contrastive learning. In specific, we consider 2 augmented views of an input ECG signal and we train the network to identify these pairings among all possible pairs in a training batch. To augment the ECG samples we use the [PyTorch ECG Augmentations](https://github.com/klean2050/ecg-augmentations) package. First, the input ECG is randomly cropped to 10 seconds and a series of masks and signal transformations are randomly applied based on a set probability. This is applied online, twice during training, to produce the 2 augmented views.

### Backbone \& Objective

A lightweight ResNet encoder is used to extract latent representations from the augmented data inputs. We use a light architecture of 8 blocks and 16 filters at the first block, in order to abide by the domain literature and make the model applicable to real-time settings. The 256D output embeddings of a pair of augmented samples are projected to a 128D latent space, where all samples within a batch are contrasted using the NT-Xent loss, adapted from the SimCLR study. With this loss, the model is forced to identify the underlying association between augmented versions of the same sample. The network is trained for about 60K steps using an AdamW optimizer.

## Fine-Tuning Framework

We transfer the trained ECG encoder to the downstream tasks in a teacher-student setting, where additional sensor streams are trained from scratch for an estimation task, along with aligning their latent representations to those produced by the (frozen) ECG model. Hence each modality incorporates a separate network and a double objective to train upon. The final state estimation is done using late fusion of the different modalities (i.e., either by prediction fusion or majority voting - TBD).

### Case: DriveDB

The specific dataset contains raw sensor measurements like ECG, EDA, HR and respiration information. Since no annotations or behavioral or environmental variables are given, we model the state of each driver by predicting 5-min averaged EDA from ECG and HR streams. The framework we described is successfully trained to estimate the average EDA value per 5-minute intervals in a subject-independent setting.

### Case: SWELL-KW

### Case: WESAD

The dataset for WEarable Stress and Affect Detection (WESAD) contains ECG data from 15 participants. RespiBAN Professional sensors were used to collect ECG at a sampling rate of 700 Hz. The goal was to study 4 different affective states (neutral, stressed, amused, and meditated). To perform this study, 4 different test scenarios were created. First, 20 minutes of neutral data were collected, during which participants were asked to do normal activities. During the amusement scenario, participants watched 11 funny video clips for a total of 392 seconds. Next, participants went through public speaking and arithmetic tasks for a total of 10 minutes as part of the stress scenario. Finally, participants went through a guided meditation session of 7 minutes in duration. Upon completion of each trial, the ground truth labels for the affect states were collected using PANAS.

### Case: Toyota - MIRISE Dataset

Initial experiment: differentiate between *sunny* and *rainy* conditions.

## Results & Checkpoints

To view results in TensorBoard run:
```
tensorboard --logdir ./runs
```

## Acknowledgements

In this study we made use of the following repositories:

* [VCMR](https://github.com/klean2050/VCMR)


## Authors
* [Kleanthis Avramidis](https://klean2050.github.io): PhD Student in Computer Science, USC SAIL
* [Tiantian Feng](https://github.com/tiantiaf0627): PhD Student in Computer Science, USC SAIL
