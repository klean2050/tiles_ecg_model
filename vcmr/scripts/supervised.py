import argparse, os, pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchaudio_augmentations import RandomResizedCrop

from vcmr.loaders import Contrastive, get_dataset
from vcmr.models import SampleCNN
from vcmr.trainers import ContrastiveLearning, SupervisedLearning, MultimodalLearning
from vcmr.utils import yaml_config_hook


if __name__ == "__main__":

    # -----------
    # ARGS PARSER
    # -----------
    parser = argparse.ArgumentParser(description="VCMR")
    parser = Trainer.add_argparse_args(parser)

    config = yaml_config_hook("config/config_sup.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    # -----------
    # DATALOADERS
    # -----------
    train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="train")
    valid_dataset = get_dataset(args.dataset, args.dataset_dir, subset="valid")

    input_shape = args.audio_length
    contrastive_train_dataset = Contrastive(
        train_dataset,
        input_shape=(1, input_shape),
        transform=RandomResizedCrop(n_samples=args.audio_length)
    )
    contrastive_valid_dataset = Contrastive(
        valid_dataset,
        input_shape=(1, input_shape),
        transform=RandomResizedCrop(n_samples=args.audio_length)
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
    encoder = SampleCNN(strides=[3, 3, 3, 3, 3, 3, 3, 3, 3])
    checkpoint = "runs/" + args.checkpoint_path
    pretrained = MultimodalLearning(args, encoder).load_from_checkpoint(
        checkpoint, encoder=encoder, output_dim=train_dataset.n_classes
    ) if "visual" in checkpoint else ContrastiveLearning(args, encoder).load_from_checkpoint(
        checkpoint, encoder=encoder, output_dim=train_dataset.n_classes
    ) 

    module = SupervisedLearning(
        args, encoder=pretrained.encoder, output_dim=train_dataset.n_classes
    )
    logger = TensorBoardLogger("runs", name=f"VCMR-{args.dataset}")

    # --------
    # TRAINING
    # --------
    os.environ["CUDA_VISIBLE_DEVICES"] = args.n_cuda
    stop = EarlyStopping(monitor="Valid/loss", mode="min", patience=10)
    restore = ModelCheckpoint(save_top_k=1, monitor="Valid/loss", mode="min")
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        sync_batchnorm=True,
        max_epochs=100,
        callbacks=[stop, restore],
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        strategy="ddp_find_unused_parameters_false",
        accelerator="gpu",
        devices="auto"
    )
    trainer.fit(module, train_loader, valid_loader)
