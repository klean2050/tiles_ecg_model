import argparse, os, pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

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
    # dataloaders
    # ------------
    train_dataset = get_dataset("audio_visual", args.dataset_dir, subset="train")
    valid_dataset = get_dataset("audio_visual", args.dataset_dir, subset="valid")

    contrastive_train_dataset = MultiContrastive(train_dataset, args.audio_length)
    contrastive_valid_dataset = MultiContrastive(valid_dataset, args.audio_length)

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
    # model
    # ------------
    encoder = SampleCNN(strides=[3, 3, 3, 3, 3, 3, 3, 3, 3])
    checkpoint = "runs/VCMR-audio/" + args.checkpoint_path
    pretrained = ContrastiveLearning(args, encoder).load_from_checkpoint(
        checkpoint, encoder=encoder, output_dim=train_dataset.n_classes
    )
    module = MultimodalLearning(args, pretrained.encoder)
    logger = TensorBoardLogger("runs", name="VCMR-audio_visual")

    # ------------
    # training
    # ------------
    os.environ['CUDA_VISIBLE_DEVICES'] = args.n_cuda
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        sync_batchnorm=True,
        max_epochs=args.m_epochs,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        strategy="ddp_find_unused_parameters_false",
        accelerator="gpu",
        precision=16,
        devices="auto"
    )
    trainer.fit(module, train_loader, valid_loader)
