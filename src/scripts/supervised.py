import os, argparse, pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import GroupKFold
from random import shuffle

from src.utils import yaml_config_hook, evaluate
from src.loaders import get_dataset
from src.models import ResNet1D, S4Model
from src.trainers import ContrastiveLearning, ECGLearning


if __name__ == "__main__":

    # --------------
    # CONFIGS PARSER
    # --------------

    # create args parser and link to PyTorch Lightning trainer:
    parser = argparse.ArgumentParser(description="tiles_ecg")
    parser = Trainer.add_argparse_args(parser)

    # extract args from config file and add to parser:
    config_file = "config/config_swell.yaml"
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

    # get full fine-tuning dataset
    full_dataset = get_dataset(
        dataset=args.dataset, dataset_dir=args.dataset_dir, sr=args.sr
    )

    # setup cross-validation
    gcv = GroupKFold(n_splits=args.splits)
    a = full_dataset.names.copy()
    if not args.subject_agnostic:
        shuffle(a)
    splits = [s for s in gcv.split(full_dataset, groups=a)]

    # iterate over cross-validation splits
    for i, (train_idx, valid_idx) in enumerate(splits):

        # create train and validation datasets
        train_dataset = Subset(full_dataset, train_idx)
        valid_dataset = Subset(full_dataset, valid_idx)

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

        # create supervised model
        if args.use_pretrained:
            # load pretrained ECG model from checkpoint
            pretrained_model = ContrastiveLearning.load_from_checkpoint(
                args.ssl_ckpt_path, encoder=encoder
            )
            model = ECGLearning(
                args,
                pretrained_model.encoder,
                output_dim=args.output_dim,
            )
        else:
            model = ECGLearning(
                args,
                encoder,
                output_dim=args.output_dim,
            )

        # logger that saves to /save_dir/name/version/
        logger = TensorBoardLogger(
            save_dir=args.log_dir,
            name=f"{args.experiment_name}_split_{i}",
            version=args.experiment_version,
        )

        # --------
        # TRAINING
        # --------

        # GPUs to use
        os.environ["CUDA_VISIBLE_DEVICES"] = args.n_cuda

        # create PyTorch Lightning trainer
        model_ckpt_callback = ModelCheckpoint(
            monitor="Valid/f1", mode="max", save_top_k=1
        )
        early_stop_callback = EarlyStopping(
            monitor="Valid/loss", mode="min", patience=15
        )

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
            callbacks=[model_ckpt_callback, early_stop_callback],
        )

        # train and save model
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
            ckpt_path=args.ckpt_path,
        )

        # ----------
        # EVALUATION
        # ----------

        out = f"results/{args.dataset}_split_{i}.txt"
        metrics = evaluate(
            model,
            dataset=valid_dataset,
            dataset_name=args.dataset,
            output=out,
        )
