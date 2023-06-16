import argparse, torch, os
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from src.utils import yaml_config_hook, evaluate
from src.loaders import get_dataset
from src.models import S4Model
from src.trainers import TransformLearning, SupervisedLearning


def create_model(args):
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
    return model


def make_dataloaders(args, train_dataset, valid_dataset, test_dataset=None):
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
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            drop_last=False,
        )
        return train_loader, valid_loader, test_loader
    return train_loader, valid_loader


def eval_model_old(args, model, test_loader):
    # ----------
    # EVALUATION
    # ----------
    v = "scratch" if not args.use_pretrained else "init" if args.unfreeze else "frozen"
    metrics, _ = evaluate(
        model.to(torch.device("cuda")),
        dataset=test_loader,
        dataset_name=f"{args.dataset}_{args.downstream_task_class}",
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
    return 


def eval_model(args, model, test_loader):
    metrics, metrics_agg = evaluate(
        model,
        dataset=test_loader,
        dataset_name=f"{args.dataset}_{args.downstream_task_class}",
    )
    return metrics, metrics_agg    


def update_metrics_dicts(metrics, all_metrics, metrics_agg=None, all_metrics_agg=None):
    for m in metrics:
        if m not in all_metrics:
            all_metrics[m] = []
        all_metrics[m].append(metrics[m])
    if metrics_agg and all_metrics_agg:
        for m in metrics_agg:
            if m not in all_metrics_agg:
                all_metrics_agg[m] = []
            all_metrics_agg[m].append(metrics_agg[m])


def log_results(args, all_metrics, all_metrics_agg=None):
    output = f"results/{args.dataset}_{args.scenario}_{args.gtruth}_scratch_050.txt"
    with open(output, "w") as f:
        for m, v in all_metrics.items():
            f.write("Chunk-wise {}: {:.3f} ({:.3f})\n".format(m, np.mean(v), np.std(v)))
        if all_metrics_agg:
            for m, v in all_metrics_agg.items():
                f.write("Aggregated {}: {:.3f} ({:.3f})\n".format(m, np.mean(v), np.std(v)))


def non_cv_training(args):
    train_dataset = get_dataset(
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        sr=args.sr,
        split="train",
        split_strategy='train-val-test',
        gtruth=args.gtruth,
        downstream_task=args.downstream_task_class,
        ecg_only=args.ecg_only,
        separate_windows=args.separate_windows
    )
    valid_dataset = get_dataset(
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        sr=args.sr,
        split="dev",
        split_strategy='train-val-test',
        gtruth=args.gtruth,
        downstream_task=args.downstream_task_class,
        ecg_only=args.ecg_only,
        separate_windows=args.separate_windows
    )
    test_dataset = get_dataset(
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        sr=args.sr,
        split="test",
        split_strategy='train-val-test',
        gtruth=args.gtruth,
        downstream_task=args.downstream_task_class,
        ecg_only=args.ecg_only,
        separate_windows=args.separate_windows
    )
    train_dataloader, valid_dataloader, test_dataloader = make_dataloaders(args, train_dataset, valid_dataset, test_dataset)
    model = create_model(args)
    # logger that saves to /save_dir/name/version/
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=f"{args.experiment_name}",
        version=f"{args.dataset}_{args.scenario}_{args.fold}_{args.gtruth}",
    )
    # --------
    # TRAINING
    # --------

    # GPUs to use
    os.environ["CUDA_VISIBLE_DEVICES"] = args.n_cuda
    # create PyTorch Lightning trainer
    monitor = "Valid/cccloss"
    model_ckpt_callback = ModelCheckpoint(monitor=monitor, mode="min", save_top_k=1)
    early_stop_callback = EarlyStopping(monitor=monitor, mode="min", patience=15)

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
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        ckpt_path=args.ckpt_path,
    )
    all_metrics = dict()
    metrics, _ = eval_model(args, model, test_dataloader)
    update_metrics_dicts(metrics, all_metrics)
    log_results(args, all_metrics)

# def cv_training(args):
    
#     all_metrics, all_metrics_agg = {}, {}
#     for i, (train_idx, valid_idx) in enumerate(splits):
#         pass



if __name__ == "__main__":

    # --------------
    # CONFIGS PARSER
    # --------------

    # create args parser and link to PyTorch Lightning trainer:
    parser = argparse.ArgumentParser(description="tiles_ecg")
    parser = Trainer.add_argparse_args(parser)

    # extract args from config file and add to parser:
    config_file = "config/config_case.yaml"
    config = yaml_config_hook(config_file)
    for key, value in config.items():
        parser.add_argument(f"--{key}", default=value, type=type(value))
    args = parser.parse_args()

    if len(args.streams) == 1:
        if 'ecg' in args.streams:
            args.ecg_only = True
        else:
            raise NotImplementedError("Use either ecg or all signals")

    # set random seed if selected:
    if args.seed:
        pl.seed_everything(args.seed, workers=True)

    # ------------
    # DATA LOADERS
    # ------------
    if args.scenario == 'crossval':
        pass
    elif args.scenario == 'non-crossval':
        non_cv_training(args)
    else:
        raise NotImplementedError(f"{args.scenario} training not implemented")
    # --------------
    # MODEL & LOGGER
    # --------------

    

    # --------
    # TRAINING
    # --------

    
    # ----------
    # EVALUATION
    # ----------

