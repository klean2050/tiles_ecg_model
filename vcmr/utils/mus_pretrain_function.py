"""Contains function to perform music (audio) pretraining."""


import os
import argparse
from typing import List
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchaudio_augmentations import (ComposeMany, RandomApply, RandomResizedCrop, PolarityInversion, Noise, Gain, HighLowPass, Delay, PitchShift, Reverb)
import torchinfo

from vcmr.utils import yaml_config_hook
from vcmr.loaders import get_dataset, Contrastive
from vcmr.models.sample_cnn_config import SampleCNN
from vcmr.trainers.contrastive_learning import ContrastiveLearning


# path of (overall) directory in which to save training logs:
LOG_DIR = "runs/"
# constants:
NUM_AUG_SAMPLES = 2
AUDIO_CHUNK_LENGTH_SEC = 15
GPUS_DEFAULT = "1, 2, 3"


def mus_pretrain(config_file: str, exp_name: str, exp_run_name : str = None, gpus_to_use: str = None, model_summary_info : List[str] = ["input_size", "output_size", "num_params"], verbose : bool = 1) -> None:
    """Performs music (audio) pretraining.

    Args:
        config_file: Path of config (yaml) file.
        exp_name: Name of experiment (used to name logs subdirectory).
        exp_run_name: Name of experiment run (used to name logs subdirectory).
        gpus_to_use: Which GPUs (as numbers) to use.
        model_summary_info: What information to include in model summary.
        verbose: Level of information to display.
    
    Returns: None
    """

    # -----------
    # ARGS PARSER
    # -----------

    # create args parser and link to PyTorch Lightning trainer:
    parser = argparse.ArgumentParser(description="VCMR")
    parser = Trainer.add_argparse_args(parser)

    # extract args from config file and add to parser:
    config = yaml_config_hook(config_file)
    for key, value in config.items():
        parser.add_argument(f"--{key}", default=value, type=type(value))
    # parse args:
    args = parser.parse_args()
    # MAYBE???:
    # args = Trainer.parse_argparser(parser.parse_args(""))

    # set random seed:
    print()
    pl.seed_everything(args.seed)


    # -------------------
    # AUDIO AUGMENTATIONS
    # -------------------

    # create transform for audio augmentations:
    train_transform = [
        RandomResizedCrop(n_samples=args.audio_length),
        RandomApply([PolarityInversion()], p=args.transforms_polarity),
        RandomApply([Noise()], p=args.transforms_noise),
        RandomApply([Gain()], p=args.transforms_gain),
        RandomApply([HighLowPass(sample_rate=args.sample_rate)], p=args.transforms_filters),
        RandomApply([Delay(sample_rate=args.sample_rate)], p=args.transforms_delay),
        RandomApply([PitchShift(n_samples=args.audio_length, sample_rate=args.sample_rate)], p=args.transforms_pitch),
        RandomApply([Reverb(sample_rate=args.sample_rate)], p=args.transforms_reverb)
    ]


    # ------------
    # DATA LOADERS
    # ------------

    if verbose > 0:
        print("\nSetting up data loaders...")
    
    # get training/validation datasets:
    train_dataset = get_dataset("audio", args.dataset_dir, subset="train")
    valid_dataset = get_dataset("audio", args.dataset_dir, subset="valid")

    # set up contrastive learning training/validation datasets:
    contrastive_train_dataset = Contrastive(
        train_dataset,
        input_shape=(1, args.sample_rate * AUDIO_CHUNK_LENGTH_SEC),
        transform=ComposeMany(train_transform, num_augmented_samples=NUM_AUG_SAMPLES)
    )
    contrastive_valid_dataset = Contrastive(
        valid_dataset,
        input_shape=(1, args.sample_rate * AUDIO_CHUNK_LENGTH_SEC),
        transform=ComposeMany(train_transform, num_augmented_samples=NUM_AUG_SAMPLES)
    )

    # create training/validation PyTorch data loaders:
    train_loader = DataLoader(contrastive_train_dataset, batch_size=args.batch_size, num_workers=args.workers, drop_last=True, shuffle=True)
    valid_loader = DataLoader(contrastive_valid_dataset, batch_size=args.batch_size, num_workers=args.workers, drop_last=True, shuffle=False)


    # --------------
    # MODEL & LOGGER
    # --------------

    if verbose > 0:
        print("\nSetting up model and logger...")
    
    # backbone (audio) encoder:
    encoder = SampleCNN(
        n_blocks=args.n_blocks,
        n_channels=args.n_channels,
        output_size=args.output_size,
        conv_kernel_size=args.conv_kernel_size,
        pool_size=args.pool_size,
        activation=args.activation,
        first_block_params={
            "out_channels": args.first_out_channels,
            "conv_size": args.first_conv_size
        },
        input_size=args.audio_length
    )
    # full model:
    model = ContrastiveLearning(args, encoder)

    # logger (logs are saved to LOG_DIR/exp_name/exp_run_name/):
    logger = TensorBoardLogger(LOG_DIR, name=exp_name, version=exp_run_name)


    # """
    # --------
    # TRAINING
    # --------

    # select GPUs to use:
    if gpus_to_use is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus_to_use
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = GPUS_DEFAULT
    
    # create PyTorch Lightning trainer:
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        sync_batchnorm=True,
        max_epochs=15,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        strategy="ddp_find_unused_parameters_false",
        accelerator="gpu",
        devices="auto"
    )

    # train model:
    trainer.fit(model, train_loader, valid_loader)
    # """


    # ---------------
    # MODEL SUMMARIES
    # ---------------

    if verbose > 0:
        print("\n\n\n\n\nCreating model summaries...")
    
    # batch size of test tensor for model summaries (most likely just keep equal to 1):
    batch_size_summary = 1

    # create model summaries directory:
    summaries_dir = os.path.join(LOG_DIR, exp_name, exp_run_name, "model_summaries", "")
    os.makedirs(summaries_dir, exist_ok=True)

    # create model summaries:
    encoder_summary = str(torchinfo.summary(model.encoder, input_size=(batch_size_summary, 1, args.audio_length), col_names=model_summary_info, depth=3, verbose=0))
    projector_summary = str(torchinfo.summary(model.projector, input_size=(batch_size_summary, model.n_features), col_names=model_summary_info, depth=1, verbose=0))
    
    # save model summaries:
    encoder_summary_file = os.path.join(summaries_dir, "encoder_summary.txt")
    with open(encoder_summary_file, "w") as text_file:
        text_file.write(encoder_summary)
    projector_summary_file = os.path.join(summaries_dir, "projector_summary.txt")
    with open(projector_summary_file, "w") as text_file:
        text_file.write(projector_summary)
    
    # display encoder summary, if selected:
    if verbose >= 2:
        print("\n\nENCODER SUMMARY:\n")
        print(encoder_summary)
        print("\n\n\n\nPROJECTOR SUMMARY:\n")
        print(projector_summary)

