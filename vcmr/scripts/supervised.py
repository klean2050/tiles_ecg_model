import argparse, os, pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# Audio Augmentations
from torchaudio_augmentations import (
    ComposeMany,
    RandomResizedCrop,
)

from vcmr.loaders import ContrastiveDataset, get_dataset
from vcmr.models import SampleCNN
from vcmr.trainers import ContrastiveLearning, SupervisedLearning, MultimodalLearning
from vcmr.utils import yaml_config_hook


if __name__ == "__main__":

    # -----------
    # ARGS PARSER
    # -----------
    parser = argparse.ArgumentParser(description="CLMR")
    parser = Trainer.add_argparse_args(parser)

    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    # -------------
    # AUGMENTATIONS
    # -------------
    train_transform = [RandomResizedCrop(n_samples=args.audio_length)]
    # train_transform = []

    # -----------
    # DATALOADERS
    # -----------
    train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="train")
    valid_dataset = get_dataset(args.dataset, args.dataset_dir, subset="valid")

    input_shape = args.audio_length
    contrastive_train_dataset = ContrastiveDataset(
        train_dataset,
        input_shape=(1, input_shape),
        transform=ComposeMany(
            train_transform, num_augmented_samples=1
        ),
    )
    contrastive_valid_dataset = ContrastiveDataset(
        valid_dataset,
        input_shape=(1, input_shape),
        transform=ComposeMany(
            train_transform, num_augmented_samples=1
        ),
    )

    train_loader = DataLoader(
        contrastive_train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=True,
    )
    valid_loader = DataLoader(
        contrastive_valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=False,
    )

    # ---------------------
    # ENCODER & CHECKPOINTS
    # ---------------------
    encoder = SampleCNN(
        strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
        supervised=args.supervised,
        out_dim=train_dataset.n_classes,
    )

    logger = TensorBoardLogger("runs", name="CLMRmusvid-{}-90".format(args.dataset))
    proxy = ContrastiveLearning(args, encoder, pre=True)
    proxy = proxy.load_from_checkpoint(
        args.checkpoint_path1, encoder=encoder, output_dim=train_dataset.n_classes
    )
    pretrained = MultimodalLearning(args, encoder)
    pretrained = pretrained.load_from_checkpoint(
        args.checkpoint_path2, enc1=encoder, output_dim=train_dataset.n_classes
    )

    # --------
    # TRAINING
    # --------
    module = SupervisedLearning(
        args, encoder, pretrained, output_dim=train_dataset.n_classes
    )
    early_stopping = EarlyStopping(monitor="Valid/loss", patience=5)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        sync_batchnorm=True,
        max_epochs=30,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        gpus=[0, 1],
        accelerator="gpu",
        # strategy="ddp",
    )
    trainer.fit(module, train_loader, valid_loader)
