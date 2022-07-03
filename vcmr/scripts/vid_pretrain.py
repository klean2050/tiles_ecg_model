import argparse, os, pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# Audio Augmentations
from torchaudio_augmentations import (
    ComposeMany,
    RandomResizedCrop,
)

from vcmr.loaders import get_dataset, MultiContrastive
from vcmr.models import SampleCNN
from vcmr.trainers import MultimodalLearning, ContrastiveLearning
from vcmr.utils import yaml_config_hook


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VCMR")
    parser = Trainer.add_argparse_args(parser)

    config = yaml_config_hook("config/config_vid.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    # ------------
    # augmentations
    # ------------
    train_transform = [RandomResizedCrop(n_samples=args.audio_length)]
    num_augmented_samples = 1

    # ------------
    # dataloaders
    # ------------"
    train_dataset = get_dataset("audio_visual", args.dataset_dir, subset="train")
    valid_dataset = get_dataset("audio_visual", args.dataset_dir, subset="valid")

    contrastive_train_dataset = MultiContrastive(
        train_dataset,
        input_shape=(1, args.sample_rate * 15),
        transform=ComposeMany(
            train_transform, num_augmented_samples=num_augmented_samples
        ),
    )
    contrastive_valid_dataset = MultiContrastive(
        valid_dataset,
        input_shape=(1, args.sample_rate * 15),
        transform=ComposeMany(
            train_transform, num_augmented_samples=num_augmented_samples
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

    # ------------
    # encoder
    # ------------
    encoder = SampleCNN(
        strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
        supervised=0,
        out_dim=train_dataset.n_classes,
    )

    # ------------
    # model
    # ------------
    checkpoint = "runs/VCMR-audio/" + args.checkpoint_path
    pretrained = ContrastiveLearning(args, encoder, pre=True)
    pretrained = pretrained.load_from_checkpoint(
        checkpoint, encoder=encoder, output_dim=train_dataset.n_classes
    )
    module = MultimodalLearning(args, encoder, ckpt=pretrained)
    logger = TensorBoardLogger("runs", name="VCMR-audio_visual")

    # ------------
    # training
    # ------------
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        sync_batchnorm=True,
        max_epochs=50,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        strategy="ddp_find_unused_parameters_false",
        accelerator="gpu",
        devices="auto"
    )
    trainer.fit(module, train_loader, valid_loader)
