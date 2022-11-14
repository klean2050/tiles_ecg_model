

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
├── setup.py             # package installation script
└── video_list.txt       # list of curated YouTube videos (ID)
```

## Getting the Data

VCMR is trained on a large-scale dataset of 20150 music video clips, downloaded from YouTube in MPEG-4 high-resolution format. The unique codes of the utilized videos can be found at ``video_list.txt``. To reproduce the dataset, run ``preprocess_data.py``.

For each video we isolate the middle 2 minutes of its content. To avoid non-official clips (e.g., amateur covers, lyric videos) we keep track of the scenes and discard those clips that include a scene of more than 30 seconds. For the music encoder we extract the audio component in WAV mono format at 16 kHz and split it at 8 segments of 15 seconds. For the visual encoder we extract CLIP embeddings from frames at 5 fps and average the resulting 512-D feature vectors per second.

## The VCMR Framework

The 2 pre-training phases run on the custom datasets called ``audio`` and ``audio_visual`` respectively, so user needs just to specify the path to the data. The fine-tuning phase requires user to specify the dataset of interest.

### 1. Music Pre-Training

```
python vcmr/scripts/mus_pretrain.py --dataset_dir /path/to/audio/folder/
```

### 2. Video-Conditioned Pre-Training

```
python vcmr/scripts/vid_pretrain.py --dataset_dir /path/to/data/folder/
```

### 3. Supervised Fine-Tuning

```
python vcmr/scripts/supervised.py --dataset <dataset_name>
```

### 4. Model Evaluation

```
python vcmr/scripts/evaluation.py --dataset <dataset_name>
```

## Results & Checkpoints (TBD)

To view results in TensorBoard run:
```
tensorboard --logdir ./runs
```

## Acknowledgements

In this study we made use of the following repositories:

* [CLMR](https://github.com/Spijkervet/CLMR)
* [MoviePy](https://github.com/Zulko/moviepy)
* [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)


## Authors
* [Kleanthis Avramidis](https://klean2050.github.io): PhD Student in Computer Science, USC
* [Shanti Stewart](https://www.linkedin.com/in/shanti-stewart/): MS Student in Machine Learning & Data Science, USC

## Citation

```
@article{avramidis2022role,
  title={On the Role of Visual Context in Enriching Music Representations},
  author={Avramidis, Kleanthis and Stewart, Shanti and Narayanan, Shrikanth},
  journal={arXiv preprint arXiv:2210.15828},
  year={2022}
}
```
