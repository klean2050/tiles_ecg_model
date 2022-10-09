"""Script to perform evaluation of supervised models on music tagging."""


import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
import warnings

from vcmr.loaders import get_dataset, Contrastive
from vcmr.models.sample_cnn_config import SampleCNN
from vcmr.trainers import SupervisedLearning
from vcmr.utils import yaml_config_hook, evaluate


# script options:
config_file = "config/config_eval_new.yaml"
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

    # validate datset subset:
    if args.dataset_subset not in ["train", "valid", "test"]:
        raise ValueError("Invalid dataset subset.")
    
    # get dataset:
    dataset = get_dataset(
        args.dataset,
        args.dataset_dir,
        subset=args.dataset_subset,
        sr=args.sample_rate
    )
    
    if verbose:
        print("\nUsing {} subset of {} with {} examples...".format(args.dataset_subset, args.dataset, len(dataset)))
    

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
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.n_cuda     # produces a CUDA error for some reason

    # create directories for saving results:
    # note: main results directory = results_dir/dataset/model_name/
    main_results_dir = os.path.join(args.results_dir, args.dataset, args.model_name, "")
    # note: results subdirectory of modality x = results_dir/dataset/model_name/modality_x/x_model_version/
    audio_results_dir = os.path.join(main_results_dir, "music_only", args.audio_model_version, "")
    multimodal_results_dir = os.path.join(main_results_dir, "multimodal", args.multimodaL_model_version, "")
    os.makedirs(main_results_dir, exist_ok=True)
    os.makedirs(audio_results_dir, exist_ok=True)
    os.makedirs(multimodal_results_dir, exist_ok=True)

    # evaluate supervised model pretrained on audio only:
    if verbose:
        print("\n\nRunning evaluation for music only model...\n")
    evaluate(
        audio_model,
        dataset=dataset,
        dataset_name=args.dataset,
        audio_length=args.audio_length,
        overlap_ratios=args.song_split_overlap_ratios,
        output_dir=audio_results_dir,
        agg_methods=args.aggregation_methods,
        device=device,
        verbose=verbose
    )

    # evaluate supervised model pretrained on audio + video:
    if verbose:
        print("\n\nRunning evaluation for multimodal model...\n")
    evaluate(
        multimodal_model,
        dataset=dataset,
        dataset_name=args.dataset,
        audio_length=args.audio_length,
        overlap_ratios=args.song_split_overlap_ratios,
        output_dir=multimodal_results_dir,
        agg_methods=args.aggregation_methods,
        device=device,
        verbose=verbose
    )


    print("\n\n")

