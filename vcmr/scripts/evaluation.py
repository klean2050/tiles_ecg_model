import argparse, pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchaudio_augmentations import RandomResizedCrop

from vcmr.loaders import get_dataset, Contrastive
from vcmr.models import SampleCNN
from vcmr.trainers import ContrastiveLearning, SupervisedLearning
from vcmr.utils import yaml_config_hook, evaluate


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
    encoder = SampleCNN(
        strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
        supervised=1,
        out_dim=train_dataset.n_classes,
    )
    #pretrained = ContrastiveLearning(args, encoder, pre=True)

    checkpoint = f"runs/VCMR-{args.dataset}/" + args.checkpoint_path3
    module = SupervisedLearning(
        args, encoder, output_dim=train_dataset.n_classes
    ).load_from_checkpoint(
        checkpoint, enc1=encoder, output_dim=train_dataset.n_classes
    )

    contrastive_test_dataset = Contrastive(
        get_dataset(args.dataset, args.dataset_dir, subset="test"),
        input_shape=(1, args.audio_length),
        transform=RandomResizedCrop(n_samples=args.audio_length)
    )
    results = evaluate(
        module,
        contrastive_test_dataset,
        args.dataset,
        args.audio_length,
        device="cuda"
    )
    print(results)
