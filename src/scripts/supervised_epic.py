import argparse, pytorch_lightning as pl
import torch, os, numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from src.utils import yaml_config_hook, evaluate
from src.loaders import get_dataset
from src.models import S4Model
from src.trainers import TransformLearning, SupervisedLearning


if __name__ == "__main__":

    # --------------
    # CONFIGS PARSER
    # --------------

    # create args parser and link to PyTorch Lightning trainer:
    parser = argparse.ArgumentParser(description="tiles_ecg")
    parser = Trainer.add_argparse_args(parser)

    # extract args from config file and add to parser:
    config_file = "config/config_epic.yaml"
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
    train_dataset = get_dataset(
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        sr=args.sr,
        split="train",
        gtruth=args.gtruth,
        ecg_only=False,
    )
    valid_dataset = get_dataset(
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        sr=args.sr,
        split="dev",
        gtruth=args.gtruth,
        ecg_only=False,
    )
    test_dataset = get_dataset(
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        sr=args.sr,
        split="test",
        gtruth=args.gtruth,
        ecg_only=False,
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
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )

    # --------------
    # MODEL & LOGGER
    # --------------

    encoder = S4Model(
        d_input=args.d_input,
        d_output=args.d_output,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        prenorm=True,
    )
    if args.use_pretrained:
        # load pretrained ECG model from checkpoint
        pretrained_model = TransformLearning.load_from_checkpoint(
            args.ssl_ckpt_path, encoder=encoder
        )
        model = SupervisedLearning(
            args,
            pretrained_model.encoder,
            output_dim=args.output_dim,
        )
    else:
        model = SupervisedLearning(
            args,
            encoder,
            output_dim=args.output_dim,
        )

    # logger that saves to /save_dir/name/version/
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=f"{args.experiment_name}",
        version=f"{len(args.streams)}{args.dataset}_sc={args.scenario}_fold{args.fold}_{args.gtruth}",
    )

    # --------
    # TRAINING
    # --------

    # GPUs to use
    os.environ["CUDA_VISIBLE_DEVICES"] = args.n_cuda
    # create PyTorch Lightning trainer
    monitor = (
        "Valid/cccloss"
        if "avec" in args.dataset_dir or "epic" in args.dataset_dir
        else "Valid/loss"
    )
    model_ckpt_callback = ModelCheckpoint(monitor=monitor, mode="min", save_top_k=1)
    early_stop_callback = EarlyStopping(monitor=monitor, mode="min", patience=11)

    # train and save model
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
        # resume_from_checkpoint='./checkpoints/blahblah.ckpt'
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=args.ckpt_path,
    )

    # ----------
    # EVALUATION
    # ----------

    v = "scratch" if not args.use_pretrained else "init" if args.unfreeze else "frozen"
    metrics, _ = evaluate(
        model.to(torch.device("cuda")),
        dataset=test_loader,
        dataset_name=args.dataset,
        modalities=args.streams,
    )

    if len(args.streams) > 1:
        args.dataset += "_multi"
    if "epic" in args.dataset_dir:
        np.save(
            f"results/{args.dataset}_{args.scenario}_{args.fold}_{args.gtruth}_{v}.npy",
            metrics,
        )
    else:
        output = f"results/{args.dataset}_{args.gtruth}_{v}.txt"
        with open(output, "w") as f:
            for m, v in metrics.items():
                f.write("{}: {:.3f}\n".format(m, v))
