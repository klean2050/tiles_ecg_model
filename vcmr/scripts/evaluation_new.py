"""Script to perform evaluation of supervised models on music tagging."""


import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
import numpy as np
import torchinfo
import pickle
import warnings

from vcmr.loaders import get_dataset, Contrastive
from vcmr.models.sample_cnn_config import SampleCNN
from vcmr.trainers import SupervisedLearning
from vcmr.utils import yaml_config_hook, evaluate


# script options:
config_file = "config/config_eval_new.yaml"
model_summary_info = ["input_size", "output_size", "num_params"]
verbose = 1


if __name__ == "__main__":
    print("\n\n")
    # ignore warnings:
    warnings.filterwarnings("ignore")

    # --------------
    # CONFIGS PARSER
    # --------------

    # create args parser and link to PyTorch Lightning trainer:
    parser = argparse.ArgumentParser(description="VCMR")
    parser = Trainer.add_argparse_args(parser)

    # extract args from config file and add to parser:
    config = yaml_config_hook(config_file)
    for key, value in config.items():
        parser.add_argument(f"--{key}", default=value, type=type(value))
    # parse args:
    args = parser.parse_args()
    # MAYBE???:
    # args = Trainer.parse_argparser(parser.parse_args(""))

    # set random seed if selected:
    if args.seed:
        pl.seed_everything(args.seed, workers=True)
    

    # -------
    # DATASET
    # -------

    # get test dataset:
    dataset = get_dataset(
        args.dataset,
        args.dataset_dir,
        subset="test",
        sr=args.sample_rate
    )
    test_dataset = Contrastive(
        dataset
    )

    if verbose:
        print("\nUsing test subset of {} with {} examples...".format(args.dataset, len(dataset)))
    

    # ------
    # MODELS
    # ------

    if verbose:
        print("\nLoading models...")
    
    # create backbone (audio) encoder:
    encoder = SampleCNN(
        n_blocks=args.n_blocks,
        n_channels=args.n_channels,
        output_size=args.output_size,
        conv_kernel_size=args.conv_kernel_size,
        pool_size=args.pool_size,
        activation=args.activation,
        first_block_params=args.first_block_params,
        input_size=args.audio_length
    )

    # load supervised model pretrained on audio only from checkpoint:
    audio_model = SupervisedLearning.load_from_checkpoint(
        args.ckpt_path_audio,
        encoder=encoder,
        output_dim=dataset.n_classes
    )
    # load supervised model pretrained on audio + video from checkpoint:
    multimodal_model = SupervisedLearning.load_from_checkpoint(
        args.ckpt_path_multimodal,
        encoder=encoder,
        output_dim=dataset.n_classes
    )


    # ----------
    # EVALUATION
    # ----------

    # select single GPU to use:
    device = torch.device(f"cuda:{args.n_cuda}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.n_cuda

    # create directories for saving results:
    # note: main results directory = results_dir/dataset/model_name/aggregation_method/
    main_results_dir = os.path.join(args.results_dir, args.dataset, args.model_name, args.aggregation_method, "")
    audio_results_dir = os.path.join(main_results_dir, "music_only", "")
    multimodal_results_dir = os.path.join(main_results_dir, "multimodal", "")
    os.makedirs(main_results_dir, exist_ok=True)
    os.makedirs(audio_results_dir, exist_ok=True)
    os.makedirs(multimodal_results_dir, exist_ok=True)

    # evaluate supervised model pretrained on audio only:
    if verbose:
        print("\nRunning evaluation for music only model...")
    evaluate(
        audio_model,
        test_dataset=test_dataset,
        dataset_name=args.dataset,
        audio_length=args.audio_length,
        output_dir=audio_results_dir,
        aggregation_method=args.aggregation_method,
        device=device
    )

    # evaluate supervised model pretrained on audio + video:
    if verbose:
        print("\nRunning evaluation for multimodal model...")
    evaluate(
        multimodal_model,
        test_dataset=test_dataset,
        dataset_name=args.dataset,
        audio_length=args.audio_length,
        output_dir=multimodal_results_dir,
        aggregation_method=args.aggregation_method,
        device=device
    )

    # load results:
    with open(os.path.join(audio_results_dir, "overall_dict.pickle"), "rb") as pickle_file:
        results_audio = pickle.load(pickle_file)
    with open(os.path.join(multimodal_results_dir, "overall_dict.pickle"), "rb") as pickle_file:
        results_multimodal = pickle.load(pickle_file)
    
    # print results:
    print()
    print("\nResults for music only model:")
    for key in results_audio.keys():
        print(f"{key}: {100 * np.around(results_audio[key], 3) : .1f} %")
    print("\nResults for multimodal model:")
    for key in results_multimodal.keys():
        print(f"{key}: {100 * np.around(results_multimodal[key], 3) : .1f} %")
    
    print("\n\n")

