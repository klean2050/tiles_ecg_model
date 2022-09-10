"""Script to perform music (audio) pre-training."""


import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import torchinfo

from vcmr.utils import yaml_config_hook
from vcmr.loaders import get_dataset, MultiContrastive
from vcmr.models.sample_cnn_config import SampleCNN
from vcmr.trainers.contrastive_learning import ContrastiveLearning
from vcmr.trainers import ContrastiveLearning, MultimodalLearning


# script options:
config_file = "config/config_vid_new.yaml"
model_summary_info = ["input_size", "output_size", "num_params"]
verbose = 1


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
    # MAYBE???:
    # args = Trainer.parse_argparser(parser.parse_args(""))
    
    # set random seed if selected:
    if args.seed:
        pl.seed_everything(args.seed, workers=True)
    

    # ------------
    # DATA LOADERS
    # ------------

    if verbose:
        print("\nSetting up dataset and data loaders...")
    
    # get training/validation datasets:
    train_dataset = get_dataset(
        "audio_visual",
        args.dataset_dir,
        subset="train"
    )
    valid_dataset = get_dataset(
        "audio_visual",
        args.dataset_dir,
        subset="valid"
    )

    # set up contrastive learning training/validation datasets:
    contrastive_train_dataset = MultiContrastive(
        train_dataset,
        n_samples=args.audio_length,
        sr=args.sample_rate
    )
    contrastive_valid_dataset = MultiContrastive(
        valid_dataset,
        n_samples=args.audio_length,
        sr=args.sample_rate
    )

    # create training/validation dataloaders:
    train_loader = DataLoader(
        contrastive_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True
    )
    valid_loader = DataLoader(
        contrastive_valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=True,
    )


    # --------------
    # MODEL & LOGGER
    # --------------

    if verbose:
        print("\nSetting up model and logger...")
    
    # """
    # create backbone (audio) encoder:
    audio_encoder = SampleCNN(
        n_blocks=args.n_blocks,
        n_channels=args.n_channels,
        output_size=args.output_size,
        conv_kernel_size=args.conv_kernel_size,
        pool_size=args.pool_size,
        activation=args.activation,
        first_block_params=args.first_block_params,
        input_size=args.audio_length
    )
    # load audio model (encoder + projector) from checkpoint:
    pretrained_audio_model = ContrastiveLearning(args, audio_encoder).load_from_checkpoint(
        args.audio_encoder_ckpt_path,
        encoder=audio_encoder,
        output_dim=train_dataset.n_classes
    )
    # """
    """
    # load audio encoder/projector from checkpoint:
    pretrained_audio_model = ContrastiveLearning.load_from_checkpoint(args.audio_encoder_ckpt_path)
    """

    # create full (multimodal) model:
    full_model = MultimodalLearning(
        args,
        pretrained_audio_model.encoder
    )

    # create logger (logs are saved to /save_dir/name/version/):
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name,
        version=args.experiment_version
    )


    # --------
    # TRAINING
    # --------

    # select GPUs to use:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.n_cuda

    # create PyTorch Lightning trainer:
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        max_epochs=args.m_epochs,
        check_val_every_n_epoch=args.val_freq,
        log_every_n_steps=args.log_freq,
        sync_batchnorm=True,
        strategy="ddp_find_unused_parameters_false",
        accelerator="gpu",
        devices="auto",
        precision=args.bit_precision

        # maybe set this for consistent training runs?:
        # deterministic=True
    )
    
    # train model:
    trainer.fit(
        full_model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=args.ckpt_path
    )


    # ---------------
    # MODEL SUMMARIES
    # ---------------

    if verbose:
        if verbose >= 2:
            print("\n\n")
        print("\n\n\nCreating model summaries...")
    
    # create model summaries directory:
    summaries_dir = os.path.join(args.log_dir, args.experiment_name, "model_summaries", "")
    os.makedirs(summaries_dir, exist_ok=True)

    # create audio model summaries:
    audio_encoder_summary = str(torchinfo.summary(
        full_model.encoder,
        input_size=(args.batch_size, 1, args.audio_length),
        col_names=model_summary_info,
        depth=3,
        verbose=0
    ))
    audio_projector_summary = str(torchinfo.summary(
        full_model.projector1,
        input_size=(args.batch_size, full_model.n_features),
        col_names=model_summary_info,
        depth=1,
        verbose=0
    ))


    # create video model summaries:


    # save model summaries:
    audio_encoder_summary_file = os.path.join(summaries_dir, "audio_encoder_summary.txt")
    with open(audio_encoder_summary_file, "w") as text_file:
        text_file.write(audio_encoder_summary)
    audio_projector_summary_file = os.path.join(summaries_dir, "audio_projector_summary.txt")
    with open(audio_projector_summary_file, "w") as text_file:
        text_file.write(audio_projector_summary)
    
    # display model summaries, if selected:
    if verbose >= 2:
        print("\n\nAUDIO ENCODER SUMMARY:\n")
        print(audio_encoder_summary)
        print("\n\n\n\nAUDIO PROJECTOR SUMMARY:\n")
        print(audio_projector_summary)
    
    print("\n\n")

