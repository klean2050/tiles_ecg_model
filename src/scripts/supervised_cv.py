import os, argparse, numpy as np
import pytorch_lightning as pl, torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import GroupKFold
from random import shuffle

from src.utils import yaml_config_hook, evaluate
from src.loaders import get_dataset
from src.models import ResNet1D, S4Model
from src.trainers import SupervisedLearning, TransformLearning, ECGLearning

torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    # --------------
    # CONFIGS PARSER
    # --------------

    # create args parser and link to PyTorch Lightning trainer:
    parser = argparse.ArgumentParser(description="tiles_ecg")
    parser = Trainer.add_argparse_args(parser)

    # get dataset from command line:
    parser.add_argument("--this", default="mirise", type=str)
    args = parser.parse_args()

    # extract args from config file and add to parser:
    config_file = f"config/config_{args.this}.yaml"
    config = yaml_config_hook(config_file)
    for key, value in config.items():
        parser.add_argument(f"--{key}", default=value, type=type(value))
    args = parser.parse_args()

    # set random seed if selected:
    if args.seed:
        pl.seed_everything(args.seed, workers=True)

    # set experiment name
    ld = str(args.low_data).strip(".") if args.low_data != 1 else ""
    v = "scratch" if not args.use_pretrained else "init" if args.unfreeze else "frozen"
    sa = f"sa{ld}" if args.subject_agnostic else f"sd{ld}"
    exp_name = f"{args.dataset}_{sa}_{v}_{'_'.join(args.streams)}_{args.gtruth}"

    # ------------
    # DATA LOADERS
    # ------------

    # get full fine-tuning dataset
    full_dataset = get_dataset(
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        sr=args.sr,
        gtruth=args.gtruth,
        args=args
    )
    # calculate random threshold for F1-macro
    count_labels = np.unique(full_dataset.labels, return_counts=True)
    f1_labels = []
    for j in range(len(count_labels[0])):
        r = count_labels[1][j] / count_labels[1].sum()
        f1_labels.append(2 * r / (1 + r))
    print(f"\nRandom F1-macro: {np.mean(f1_labels):.3f}\n")

    # setup k-fold cross-validation
    gcv = GroupKFold(n_splits=args.splits)
    # separate subjects into splits
    a = full_dataset.names.copy()
    if not args.subject_agnostic and not 'case'in args.dataset_dir:
        shuffle(a)
    # get training splits
    splits = [s for s in gcv.split(full_dataset, groups=a)]

    # iterate over training splits
    all_metrics, all_metrics_agg = {}, {}
    for i, (train_idx, valid_idx) in enumerate(splits):
 
        # apply low-data settings
        if args.low_data != 1:
            times = int(1 / args.low_data)
            shuffle(train_idx)
            train_idx = train_idx[:: times]

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
            drop_last=False,
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
            pretrained_model = TransformLearning.load_from_checkpoint(
                args.ssl_ckpt_path, encoder=encoder
            )
            encoder = pretrained_model.encoder

        if args.streams == ["ecg"]:
            model = ECGLearning(args, encoder)
        else:
            model = SupervisedLearning(args, encoder)

        # logger that saves to /save_dir/name/version/
        logger = TensorBoardLogger(
            save_dir=args.log_dir,
            name=f"{exp_name}_split_{i}",
            version=args.experiment_version,
        )

        # --------
        # TRAINING
        # --------

        # GPUs to use
        os.environ["CUDA_VISIBLE_DEVICES"] = args.n_cuda

        # create PyTorch Lightning trainer
        model_ckpt_callback = ModelCheckpoint(
            monitor="Valid/ccc" if args.type == "regression" else "Valid/f1", mode="max", save_top_k=1
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
        # load best model
        ckpt = torch.load(model_ckpt_callback.best_model_path)
        model.load_state_dict(ckpt['state_dict'])

        # ----------
        # EVALUATION
        # ----------

        metrics, metrics_agg = evaluate(
            model.to(torch.device("cuda")),
            dataset=valid_loader,
            dataset_name=args.dataset,
            modalities=args.streams,
        )
        for m in metrics:
            if m not in all_metrics:
                all_metrics[m] = []
            all_metrics[m].append(metrics[m])
        for m in metrics_agg:
            if m not in all_metrics_agg:
                all_metrics_agg[m] = []
            all_metrics_agg[m].append(metrics_agg[m])

    # -----------
    # LOG RESULTS
    # -----------
    os.makedirs("results", exist_ok=True)
    with open(f"results/{exp_name}.txt", "w") as f:
        for m, v in all_metrics.items():
            f.write("Chunk-wise {}: {:.3f} ({:.3f})\n".format(m, np.mean(v), np.std(v)))
        for m, v in all_metrics_agg.items():
            f.write("Aggregated {}: {:.3f} ({:.3f})\n".format(m, np.mean(v), np.std(v)))
