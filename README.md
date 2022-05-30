<div align="center">

# VCMR: Video-Conditioned Music Representations
This repository is the official implementation of the VCMR project.
  
</div>

## Installation

We recommend using a conda environment with ``Python >= 3.9`` :
```
conda create -n vcmr python=3.9
conda activate vcmr
```
Clone the repository and install the dependencies:
```
git clone https://github.com/klean2050/VCMR
cd VCMR && pip install -e .
```

You will also need to install the ``libsndfile`` library:
```
conda install -c conda-forge libsndfile
```

## Project Structure

```
VCMR/
├── config/              # configuration files for each train session
├── tests/               # sample scripts to test functionalities
├── vcmr/                # main project directory
│   ├── loaders/             # pytorch dataset classes
│   ├── models/              # backbone neural network models
│   ├── scripts/             # preprocess, training and evaluation
│   ├── trainers/            # lightning classes for each train session
│   └── utils/               # miscellaneous scripts and methods
├── .gitignore           # ignore data/ and runs/ folders
├── main.py              # driver script to run
└── setup.py             # package installation script
```

## Demo Auto-Tagging (TBD)

```
python main.py --audio /path/to/audio/file.wav [--flags]
```

## Getting the Data (TBD)

VCMR is trained on a large-scaled dataset of 4857 music video clips downloaded from YouTube.

## The VCMR Framework

### 1. Music Pre-Training

```
python vcmr/scripts/mus_pretrain.py --dataset_dir /path/to/audio/folder/
```

### 2. Video-Conditioned Pre-Training

```
python vcmr/scripts/vid_pretrain.py --dataset_dir /path/to/data/folder/ --ckpt runs/path/to/mus_checkpoint.ckpt
```

### 3. Supervised Fine-Tuning

```
python vcmr/scripts/supervised.py --dataset <dataset_name> --ckpt runs/path/to/vid_checkpoint.ckpt
```

### 4. Model Evaluation

```
python vcmr/scripts/evaluation.py --dataset <dataset_name> --ckpt runs/path/to/sup_checkpoint.ckpt
```

## Results & Checkpoints (TBD)

...

To view results in TensorBoard run:
```
tensorboard --logdir ./runs
```

## Acknowledgements

In this study we made use of the following repositories:

* [CLMR](https://github.com/Spijkervet/CLMR)
* [Video Feature Extractor](https://github.com/antoine77340/video_feature_extractor)
* [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)

## Citation

TBD
