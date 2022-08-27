import argparse, pytorch_lightning as pl, warnings
import matplotlib.pyplot as plt, pickle, numpy as np
from pytorch_lightning import Trainer
warnings.filterwarnings("ignore")

from vcmr.loaders import get_dataset, Contrastive
from vcmr.models import SampleCNN
from vcmr.trainers import SupervisedLearning
from vcmr.utils import *


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

    dataset = get_dataset(args.dataset, args.dataset_dir, subset="test", sr=args.sample_rate)
    test_dataset = Contrastive(dataset, input_shape=(1, args.audio_length))
    encoder = SampleCNN(strides=[3, 3, 3, 3, 3, 3, 3, 3, 3])
    
    # ----------
    # EVALUATION
    # ----------
    
    # music model checkpoint
    checkpoint = f"runs/VCMR-{args.dataset}/" + args.checkpoint_mus
    module_mus = SupervisedLearning(
        args, encoder, output_dim=dataset.n_classes
    ).load_from_checkpoint(
        checkpoint, encoder=encoder, output_dim=dataset.n_classes
    )
    evaluate(
        module_mus,
        test_dataset,
        args.dataset + "_mus",
        args.audio_length,
        device=f"cuda:{args.n_cuda}"
    )

    # music+video model checkpoint
    checkpoint = f"runs/VCMR-{args.dataset}/" + args.checkpoint_vid
    module_vid = SupervisedLearning(
        args, encoder, output_dim=dataset.n_classes
    ).load_from_checkpoint(
        checkpoint, encoder=encoder, output_dim=dataset.n_classes
    )
    evaluate(
        module_vid,
        test_dataset,
        args.dataset + "_vid",
        args.audio_length,
        device=f"cuda:{args.n_cuda}"
    )
    
    with open(f"data/{args.dataset}_mus_classes_dict.pickle", "rb") as fp:
        a = pickle.load(fp)
        a0 = {k: v[0] for k, v in a.items()}
        a1 = {k: v[1] for k, v in a.items()}

    with open(f"data/{args.dataset}_vid_classes_dict.pickle", "rb") as fp:
        b = pickle.load(fp)
        b0 = {k: v[0] for k, v in b.items()}
        b1 = {k: v[1] for k, v in b.items()}

    _ = make_graphs(args.dataset, a0, b0, name="prs")
    tops = make_graphs(args.dataset, a1, b1, name="rcs")

    indices = [i for i, k in enumerate(b.keys()) if k in tops]
    lbs = np.load(f"data/{args.dataset}_mus_lbs.npy")
    lbs = np.stack([v[indices] for v in lbs])

    fts1 = np.load(f"data/{args.dataset}_mus_fts.npy")
    fts2 = np.load(f"data/{args.dataset}_vid_fts.npy")
    visualize(args.dataset, fts1, lbs, name="test_mus.png")
    visualize(args.dataset, fts2, lbs, name="test_vid.png")