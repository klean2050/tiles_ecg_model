import argparse, pytorch_lightning as pl
import matplotlib.pyplot as plt, pickle
from pytorch_lightning import Trainer
from torchaudio_augmentations import RandomResizedCrop

from vcmr.loaders import get_dataset, Contrastive
from vcmr.models import SampleCNN
from vcmr.trainers import ContrastiveLearning, SupervisedLearning
from vcmr.utils import yaml_config_hook, evaluate, visualize


def save_marginal(dct, name):
    plt.figure(figsize=(25, 8), dpi=200)
    plt.rcParams["font.size"] = 12
    new_dct = dct.copy()
    for i, key in enumerate(dct):
        if i > 9 and i < len(dct.keys()) - 10:
            del new_dct[key]
    plt.bar(*zip(*new_dct.items()))
    plt.grid(axis="y")
    plt.title("PR-AUC" if name=="prs" else "ROC-AUC")
    plt.savefig(f"data/{args.dataset}_{name}.png")

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
    train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="train")

    # ----------
    # EVALUATION
    # ----------
    encoder = SampleCNN(strides=[3, 3, 3, 3, 3, 3, 3, 3, 3])
    checkpoint = f"runs/VCMR-{args.dataset}/" + args.checkpoint_eval
    module = SupervisedLearning(
        args, encoder, output_dim=train_dataset.n_classes
    ).load_from_checkpoint(
        checkpoint, encoder=encoder, output_dim=train_dataset.n_classes
    )

    contrastive_test_dataset = Contrastive(
        get_dataset(args.dataset, args.dataset_dir, subset="test", sr=args.sample_rate),
        input_shape=(1, args.audio_length),
        transform=RandomResizedCrop(n_samples=args.audio_length)
    )
    
    all, _, fts, lbs = evaluate(
        module,
        contrastive_test_dataset,
        args.dataset,
        args.audio_length,
        device="cuda"
    )
    print(all)

    # ------
    # GRAPHS
    # ------
    with open(f"data/{args.dataset}_classes_dict.pickle", "rb") as fp:
        dct = pickle.load(fp)
    dct0 = {k: v[0] for k, v in dct.items()}
    save_marginal(
        {k: v for k, v in sorted(dct0.items(), key=lambda i: i[1])},
        name="prs"
    )
    dct1 = {k: v[1] for k, v in dct.items()}
    save_marginal(
        {k: v for k, v in sorted(dct1.items(), key=lambda i: i[1])},
        name="rcs"
    )
    # make t-SNE visuals
    visualize(fts, lbs, name="test.png")