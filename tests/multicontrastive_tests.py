"""Script for testing MultiContrastive PyTorch dataset class."""


import argparse
from torch import Tensor
from pytorch_lightning import Trainer
from vcmr.utils import yaml_config_hook
from vcmr.loaders import get_dataset, MultiContrastive


# script options:
config_file = "config/config_vid_new.yaml"
sample_idx = 9


if __name__ == "__main__":
    print("\n\n")

    # --------------
    # CONFIGS PARSER
    # --------------

    # create args parser and link to PyTorch Lightning trainer:
    parser = argparse.ArgumentParser(description="VCMR")
    parser = Trainer.add_argparse_args(parser)

    # extract args from config file and add to parser:
    config = yaml_config_hook(config_file)
    for key, value in config.items():
        parser.add_argument(f"--{key}", default=value, type=type(value))
    # parse args:
    args = parser.parse_args()

    # ------------
    # DATA LOADERS
    # ------------

    # get training dataset:
    train_dataset = get_dataset("audio_visual", args.dataset_dir, subset="train")

    # set up contrastive learning training dataset:
    contrastive_train_dataset = MultiContrastive(
        train_dataset, n_samples=args.audio_length, sr=args.sample_rate
    )

    # -------
    # TESTING
    # -------

    # test __getitem__() method:
    print("Testing __getitem__() method...")
    audio_crop, video_crop, _ = contrastive_train_dataset[sample_idx]

    assert (
        type(audio_crop) == Tensor and type(video_crop) == Tensor
    ), "Error with return type(s)."

    assert tuple(audio_crop.size()) == (
        1,
        args.audio_length,
    ), "Error with shape of sample cropped audio tensor."

    assert tuple(video_crop.size()) == (
        contrastive_train_dataset.n_seconds,
        contrastive_train_dataset.video_n_features,
    ), "Error with shape of sample cropped video tensor."

    print("\n\n")
