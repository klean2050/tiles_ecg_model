import os, argparse, pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from src.utils import yaml_config_hook
from src.loaders import TILES_ECG
from src.models import ResNet1D, S4Model
from src.trainers import ContrastiveLearning, TransformLearning

from ecg_augmentations import *


if __name__ == "__main__":

    # --------------
    # CONFIGS PARSER
    # --------------

    # create args parser and link to Lightning trainer
    parser = argparse.ArgumentParser(description="tiles_ecg")
    parser = Trainer.add_argparse_args(parser)

    # extract args from config and add to parser
    config_file = "config/config_ssl.yaml"
    config = yaml_config_hook(config_file)
    for key, value in config.items():
        parser.add_argument(f"--{key}", default=value, type=type(value))
    args = parser.parse_args()

    # set random seed if selected
    if args.seed:
        pl.seed_everything(args.seed, workers=True)

    # -------------
    # AUGMENTATIONS
    # -------------

    # create transform for ECG augmentation
    if args.contrastive:
        transforms = [
            RandomCrop(n_samples=1000),
            RandomApply([PRMask(sr=100)], p=0.3),
            RandomApply([QRSMask(sr=100)], p=0.3),
            RandomApply([Scale()], p=0.3),
            RandomApply([Permute()], p=0.5),
            RandomApply([GaussianNoise()], p=0.5),
            RandomApply([Invert()], p=0.1),
            RandomApply([Reverse()], p=0.1),
        ]
        transforms = ComposeMany(transforms, 2)
    else:
        transforms = [
            RandomCrop(n_samples=1000),
            RandomApply([GaussianNoise()], p=0.5),
        ]
        transforms = ComposeMany(transforms, 1)

    # ------------
    # DATA LOADERS
    # ------------

    # define training splits
    valid_sp = os.listdir(args.dataset_dir)[::10]
    train_sp = [p for p in os.listdir(args.dataset_dir) if p not in valid_sp]

    # get training and validation datasets
    train_dataset = TILES_ECG(args.dataset_dir, train_sp, transform=transforms)
    valid_dataset = TILES_ECG(args.dataset_dir, valid_sp, transform=transforms)

    # create the dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=True,
    )

    # --------------
    # MODEL & LOGGER
    # --------------

    # create backbone ECG encoder
    if args.model_type == "resnet":
        encoder = ResNet1D(
            in_channels=args.in_channels,
            base_filters=args.base_filters,
            kernel_size=args.kernel_size,
            stride=args.stride,
            groups=args.groups,
            n_block=args.n_block,
            n_classes=args.n_classes,
        )
    elif args.model_type == "s4":
        encoder = S4Model(
            d_input=args.d_input,
            d_output=args.d_output,
            d_model=args.d_model,
            n_layers=args.n_layers,
            dropout=args.dropout,
            prenorm=True,
        )
    else:
        raise ValueError("Model type not supported.")

    # create full LightningModule
    if args.contrastive:
        model = ContrastiveLearning(args, encoder.float())
    else:
        model = TransformLearning(args, encoder.float())

    # logger that saves to /save_dir/name/version/
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name,
        version=args.experiment_version,
    )

    # --------
    # TRAINING
    # --------

    # GPUs to use
    os.environ["CUDA_VISIBLE_DEVICES"] = args.n_cuda

    # create PyTorch Lightning trainer
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        max_epochs=args.m_epochs,
        check_val_every_n_epoch=args.val_freq,
        log_every_n_steps=args.log_freq,
        sync_batchnorm=True,
        strategy="ddp",
        accelerator="gpu",
        devices="auto",
        precision=args.bit_precision,
    )

    # train and save model
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=args.ckpt_path,
    )
