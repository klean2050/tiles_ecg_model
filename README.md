<div align="center">

# VCMR: Video-Conditioned Music Representations
  
</div>

This repository is the official implementation of the project.

## Installation

Clone the repository and install the dependencies. We recommend using a conda environment with Python 3.9+.
```
git clone https://github.com/klean2050/VCMR
cd VCMR 
pip install -e .
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
├── export.py            # script to export model to ONNX
└── main.py              # driver script to run
```

## Demo Auto-Tagging

```
python main.py --audio /path/to/audio/file.wav [--flags]
```

## Getting the Data

VCMR is trained on a large-scaled dataset of 4857 music video clips downloaded from YouTube.

## The VCMR Framework

### 1. Music Pre-Training

```
python scripts/mus_pretrain.py --dataset audio
```

### 2. Video-Conditioned Pre-Training

```
python scripts/vid_pretrain.py --dataset audio_visual --ckpt runs/path/to/mus_checkpoint.ckpt
```

### 3. Supervised Fine-Tuning

```
python scripts/supervised.py --dataset <dataset_name> --ckpt runs/path/to/vid_checkpoint.ckpt
```

### 4. Model Evaluation

```
python scripts/evaluation.py --dataset <dataset_name> --ckpt runs/path/to/sup_checkpoint.ckpt
```

## Results & Checkpoints

...

To view results in TensorBoard, run:
```
tensorboard --logdir ./runs
```

## Acknowledgements

In this study we made use of the following repositories:

* CLMR
* Video Feature Extractor

## Citation

TBD
