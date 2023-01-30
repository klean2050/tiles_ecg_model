import os, argparse, torchinfo
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchaudio_augmentations import RandomResizedCrop

from src.utils import yaml_config_hook
from src.loaders import DriveDB, Contrastive
from src.models import SampleCNN, ResNet1D
from src.trainers import ContrastiveLearning, MultimodalLearning, SupervisedLearning


# script options:
verbose = 1
model_summary_info = ["input_size", "output_size", "num_params"]


if __name__ == "__main__":
    print("\n\n")

    # --------------
    # CONFIGS PARSER
    # --------------

    # create args parser and link to PyTorch Lightning trainer:
    parser = argparse.ArgumentParser(description="tiles_ecg")
    parser = Trainer.add_argparse_args(parser)

    # extract args from config file and add to parser:
    config_file = "config/config_ssl.yaml"
    config = yaml_config_hook(config_file)
    for key, value in config.items():
        parser.add_argument(f"--{key}", default=value, type=type(value))
    args = parser.parse_args()

    # set random seed if selected:
    if args.seed:
        pl.seed_everything(args.seed, workers=True)

    # ------------
    # DATA LOADERS
    # ------------

    # define training splits
    valid_sp = os.listdir(args.dataset_dir)[::10]
    train_sp = [p for p in os.listdir(args.dataset_dir) if p not in valid_sp]

    # get training/validation datasets:
    train_dataset = DriveDB(
        args.dataset_dir,
        split=train_sp,
        sr=args.sample_rate,
        streams=args.streams,
    )
    valid_dataset = DriveDB(
        args.dataset_dir,
        split=valid_sp,
        sr=args.sample_rate,
        streams=args.streams,
    )

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
    encoder = ResNet1D(
        in_channels=args.in_channels,
        base_filters=args.base_filters,
        kernel_size=args.kernel_size,
        stride=args.stride,
        groups=args.groups,
        n_block=args.n_block,
        n_classes=args.n_classes,
    )
    # create full LightningModule
    model = ContrastiveLearning(args, encoder.float())

    # load pretrained ECG model from checkpoint
    pretrained_model = ContrastiveLearning.load_from_checkpoint(
        args.pretrained_ckpt_path, encoder=encoder
    )
    # create supervised learning model
    model = SupervisedLearning(
        args, pretrained_model.encoder, output_dim=train_dataset.n_classes
    )

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
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.n_cuda

    # create PyTorch Lightning trainer
    model_ckpt_callback = ModelCheckpoint(
        monitor="Valid/pr_auc", mode="max", save_top_k=1
    )
    early_stop_callback = EarlyStopping(monitor="Valid/loss", mode="min", patience=10)

    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        max_epochs=args.m_epochs,
        check_val_every_n_epoch=args.val_freq,
        log_every_n_steps=args.log_freq,
        # sync_batchnorm=True,
        strategy="ddp",
        accelerator="cpu",
        devices="auto",
        # precision=args.bit_precision,
    )

    # train and save model
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=args.ckpt_path,
    )