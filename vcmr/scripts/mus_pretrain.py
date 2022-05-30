import argparse, pytorch_lightning as pl, os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.utils.data import DataLoader
from torchaudio_augmentations import (
    RandomApply,
    ComposeMany,
    RandomResizedCrop,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
)
from vcmr.loaders import get_dataset, Contrastive
from vcmr.models import SampleCNN
from vcmr.trainers import ContrastiveLearning
from vcmr.utils import yaml_config_hook

if __name__ == "__main__":

    # -----------
    # ARGS PARSER
    # -----------
    parser = argparse.ArgumentParser(description="VCMR")
    parser = Trainer.add_argparse_args(parser)

    config = yaml_config_hook("config/config_mus.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    # -------------
    # AUGMENTATIONS
    # -------------
    num_augmented_samples = 2
    train_transform = [
        RandomResizedCrop(n_samples=args.audio_length),
        RandomApply([PolarityInversion()], p=args.transforms_polarity),
        RandomApply([Noise()], p=args.transforms_noise),
        RandomApply([Gain()], p=args.transforms_gain),
        RandomApply(
            [HighLowPass(sample_rate=args.sample_rate)], p=args.transforms_filters
        ),
        RandomApply([Delay(sample_rate=args.sample_rate)], p=args.transforms_delay),
        RandomApply(
            [
                PitchShift(
                    n_samples=args.audio_length,
                    sample_rate=args.sample_rate,
                )
            ],
            p=args.transforms_pitch,
        ),
        RandomApply(
            [Reverb(sample_rate=args.sample_rate)], p=args.transforms_reverb
        ),
    ]

    # -----------
    # DATALOADERS
    # -----------
    train_dataset = get_dataset("audio", args.dataset_dir, subset="train")
    valid_dataset = get_dataset("audio", args.dataset_dir, subset="valid")

    contrastive_train_dataset = Contrastive(
        train_dataset,
        input_shape=(1, 220500),
        transform=ComposeMany(
            train_transform, num_augmented_samples=num_augmented_samples
        ),
    )
    contrastive_valid_dataset = Contrastive(
        valid_dataset,
        input_shape=(1, 220500),
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

    # ---------------------
    # ENCODER & CHECKPOINTS
    # ---------------------
    encoder = SampleCNN(
        strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
        supervised=0,
        out_dim=train_dataset.n_classes,
    )
    logger = TensorBoardLogger("runs", name="VCMR-audio")

    # --------
    # TRAINING
    # --------
    module = ContrastiveLearning(args, encoder)
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        sync_batchnorm=True,
        max_epochs=20,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        strategy="ddp_find_unused_parameters_false",
        accelerator="gpu",
        devices="auto"
    )
    trainer.fit(module, train_loader, valid_loader)
