"""Script for testing SongSplitter PyTorch dataset class."""


import argparse
from torch import Tensor
from pytorch_lightning import Trainer
from vcmr.utils import yaml_config_hook
from vcmr.loaders import get_dataset, SongSplitter


# script options:
config_file = "config/config_eval_new.yaml"
dataset_subset = "valid"
sample_indices = [4, 9, 13]


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
    
    # get test dataset:
    dataset = get_dataset(
        args.dataset,
        args.dataset_dir,
        subset=dataset_subset,
        sr=args.sample_rate
    )
    # create wrapper dataset for splitting songs:
    test_dataset = SongSplitter(
        dataset,
        audio_length=args.audio_length,
        overlap_ratio=args.song_split_overlap_ratio
    )


    # -------
    # TESTING
    # -------

    # test __getitem__() method:
    print("Testing __getitem__() method...")
    for sample_idx in sample_indices:
        audio, label = test_dataset[sample_idx]
        assert type(audio) == Tensor and type(label) == Tensor, "Error with return type(s)."
        assert audio.dim() == 3 and audio.size(dim=1) == 1 and audio.size(dim=-1) == args.audio_length, "Error with shape of split song."
    

    print("\n\n")

