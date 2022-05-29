import argparse, os, pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# Audio Augmentations
from torchaudio_augmentations import (
    ComposeMany,
    RandomResizedCrop,
)

from vcmr.loaders import get_dataset, MultimodalDataset
from vcmr.models import SampleCNN
from vcmr.trainers import MultimodalLearning, ContrastiveLearning
from vcmr.utils import yaml_config_hook


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CLMR")
    parser = Trainer.add_argparse_args(parser)

    config = yaml_config_hook("config/config.yaml")
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
    # ------------
    train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="train")
    contrastive_train_dataset = MultimodalDataset(
        train_dataset,
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

    # ------------
    # encoder
    # ------------
    encoder = SampleCNN(
        strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
        supervised=args.supervised,
        out_dim=train_dataset.n_classes,
    )

    # ------------
    # model
    # ------------
    pretrained = ContrastiveLearning(args, encoder, pre=True)
    pretrained = pretrained.load_from_checkpoint(
        args.checkpoint_path1, encoder=encoder, output_dim=train_dataset.n_classes
    )
    module = MultimodalLearning(args, encoder, ckpt=pretrained)
    logger = TensorBoardLogger("runs", name="CLMRvid-{}".format(args.dataset))

    # ------------
    # training
    # ------------
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        sync_batchnorm=True,
        max_epochs=40,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        gpus=[0,1],
        accelerator="gpu",
        #strategy="ddp"
    )
    trainer.fit(module, train_loader, None)
