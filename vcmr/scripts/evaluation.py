import argparse, pytorch_lightning as pl
import os, pickle, warnings, numpy as np
from pytorch_lightning import Trainer

warnings.filterwarnings("ignore")

from vcmr.loaders import get_dataset, Contrastive
from vcmr.models import SampleCNN
from vcmr.trainers import SupervisedLearning
from vcmr.utils import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VCMR")
    parser = Trainer.add_argparse_args(parser)

    config = yaml_config_hook("config/config_eval.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    dataset = get_dataset(
        args.dataset, args.dataset_dir, subset="test", sr=args.sample_rate
    )
    test_dataset = Contrastive(dataset, input_shape=(1, args.audio_length))
    encoder = SampleCNN(
        n_blocks=args.n_blocks,
        n_channels=args.n_channels,
        output_size=args.output_size,
        conv_kernel_size=args.conv_kernel_size,
        pool_size=args.pool_size,
        activation=args.activation,
        first_block_params=args.first_block_params,
        input_size=args.audio_length,
    )

    mus_path = f"results/{args.dataset}/{args.checkpoint_mus}"
    vid_path = f"results/{args.dataset}/{args.checkpoint_vid}"
    os.makedirs(mus_path, exist_ok=True)
    os.makedirs(vid_path, exist_ok=True)

    # music model checkpoint
    checkpoint = f"runs/VCMR-{args.dataset}/{args.checkpoint_mus}"
    module_mus = SupervisedLearning.load_from_checkpoint(
        checkpoint, encoder=encoder, output_dim=dataset.n_classes
    )
    evaluate(
        module_mus,
        test_dataset,
        mus_path,
        args.audio_length,
        device=f"cuda:{args.n_cuda}",
    )

    # music+video model checkpoint
    checkpoint = f"runs/VCMR-{args.dataset}/{args.checkpoint_vid}"
    module_vid = SupervisedLearning.load_from_checkpoint(
        checkpoint, encoder=encoder, output_dim=dataset.n_classes
    )
    evaluate(
        module_vid,
        test_dataset,
        vid_path,
        args.audio_length,
        device=f"cuda:{args.n_cuda}",
    )

    # bars for label-wise performance
    with open(f"{mus_path}/classes_dict.pickle", "rb") as fp:
        a = pickle.load(fp)
        a0 = {k: v[0] for k, v in a.items()}
        a1 = {k: v[1] for k, v in a.items()}

    with open(f"{vid_path}/classes_dict.pickle", "rb") as fp:
        b = pickle.load(fp)
        b0 = {k: v[0] for k, v in b.items()}
        b1 = {k: v[1] for k, v in b.items()}

    _ = make_graphs(args.dataset, a0, b0, name="prs")
    tops = make_graphs(args.dataset, a1, b1, name="rcs")

    # t-SNE plots of pretrained features
    indices = [i for i, k in enumerate(b.keys()) if k in tops]
    lbs = np.load(f"{vid_path}/labels.npy")
    lbs = np.stack([v[indices] for v in lbs])

    fts1 = np.load(f"{mus_path}/features.npy")
    fts2 = np.load(f"{vid_path}/features.npy")
    visualize(args.dataset, fts1, lbs, name=f"{mus_path}/test_mus.png")
    visualize(args.dataset, fts2, lbs, name=f"{vid_path}/test_vid.png")
