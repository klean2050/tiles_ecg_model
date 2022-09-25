"""Script to perform supervised training for music tagging."""


import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchaudio_augmentations import RandomResizedCrop
import torchinfo

from vcmr.utils import yaml_config_hook
from vcmr.loaders import get_dataset, Contrastive
from vcmr.models.sample_cnn_config import SampleCNN
from vcmr.trainers import ContrastiveLearning, MultimodalLearning, SupervisedLearning


# script options:
config_file = "config/config_sup_new.yaml"
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
        args.dataset,
        args.dataset_dir,
        subset="train",
        sr=args.sample_rate
    )
    valid_dataset = get_dataset(
        args.dataset,
        args.dataset_dir,
        subset="valid",
        sr=args.sample_rate
    )

    # set up contrastive learning training/validation datasets:
    contrastive_train_dataset = Contrastive(
        train_dataset,
        transform=RandomResizedCrop(n_samples=args.audio_length)
    )
    contrastive_valid_dataset = Contrastive(
        valid_dataset,
        transform=RandomResizedCrop(n_samples=args.audio_length)
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

    # load pretrained multimodal model from checkpoint, if selected:
    if args.ckpt_model_type == "multimodal":
        pretrained_model = MultimodalLearning.load_from_checkpoint(
            args.pretrained_ckpt_path,
            encoder=audio_encoder,
            video_crop_length_sec=args.video_crop_length_sec,
            video_n_features=args.video_n_features
        )
    # load pretrained audio model from checkpoint, if selected:
    elif args.ckpt_model_type == "audio":
        pretrained_model = ContrastiveLearning.load_from_checkpoint(
            args.pretrained_ckpt_path,
            encoder=audio_encoder
        )
    else:
        raise ValueError("Invalid checkpoint model type.")
    
    # create supervised learning model:
    model = SupervisedLearning(
        args,
        pretrained_model.encoder,
        output_dim=train_dataset.n_classes
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
    model_ckpt_callback = ModelCheckpoint(monitor="Valid/pr_auc", mode="max", save_top_k=2)
    early_stop_callback = EarlyStopping(monitor="Valid/loss", mode="min", patience=10)     # SHOULDN'T MODEL = "MAX"???

    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        max_epochs=args.m_epochs,
        callbacks=[early_stop_callback, model_ckpt_callback],
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
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=args.ckpt_path
    )

