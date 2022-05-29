import argparse, pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchaudio_augmentations import RandomResizedCrop, ComposeMany

from vcmr.loaders import get_dataset, ContrastiveDataset
from vcmr.models import SampleCNN
from vcmr.trainers import ContrastiveLearning, SupervisedLearning, MultimodalLearning
from vcmr.utils import yaml_config_hook, evaluate, get_ckpt


if __name__ == "__main__":

    # -----------
    # ARGS PARSER
    # -----------
    parser = argparse.ArgumentParser(description="CLMR")
    parser = Trainer.add_argparse_args(parser)

    config = yaml_config_hook("config/config.yaml")
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
        supervised=args.supervised,
        out_dim=train_dataset.n_classes,
    )
    pretrained = ContrastiveLearning(args, encoder, pre=True)

    ckpt_path = "/data/avramidi/music-videos/clmr/runs/"
    module = SupervisedLearning(args, encoder, pretrained, output_dim=train_dataset.n_classes)
    module = module.load_from_checkpoint(
        ckpt_path + args.checkpoint_path3, enc1=encoder, output_dim=train_dataset.n_classes
     )

    transform = [RandomResizedCrop(n_samples=args.audio_length)]
    contrastive_test_dataset = ContrastiveDataset(
        get_dataset(args.dataset, args.dataset_dir, subset="test"),
        input_shape=(1, args.audio_length),
        transform=ComposeMany(transform, num_augmented_samples=1),
    )
    results = evaluate(
        module,
        contrastive_test_dataset,
        args.dataset,
        args.audio_length,
        device="cuda",
    )
    print(results)
