"""Script to perform evaluation of supervised models on music tagging."""


import os, argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch, json, warnings

from vcmr.loaders import get_dataset
from vcmr.models.sample_cnn import SampleCNN
from vcmr.trainers import SupervisedLearning
from vcmr.utils import yaml_config_hook, evaluate


# script options:
config_file = "experiments/final_evaluation/magnatagatune/config_final_eval.yaml"
mtag_tags_file = "config/mtat_tag_categories.yaml"
final_eval = True
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
    args = parser.parse_args()

    # set random seed if selected:
    if args.seed:
        pl.seed_everything(args.seed, workers=True)

    # -------
    # DATASET
    # -------

    # validate dataset subset:
    if args.dataset_subset not in ["train", "valid", "test"]:
        raise ValueError("Invalid dataset subset.")
    # get dataset:
    dataset = get_dataset(
        args.dataset, args.dataset_dir, subset=args.dataset_subset, sr=args.sample_rate
    )
    if verbose:
        print(
            "\nUsing {} subset of {} with {} examples...".format(
                args.dataset_subset, args.dataset, len(dataset)
            )
        )

    # ------
    # MODELS
    # ------

    if verbose:
        print("\nLoading models...")

    # create backbone audio encoder:
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

    # load supervised model pretrained on audio only from checkpoint:
    audio_model = SupervisedLearning.load_from_checkpoint(
        args.ckpt_path_audio, encoder=encoder, output_dim=dataset.n_classes
    )
    # load supervised model pretrained on audio + video from checkpoint:
    multimodal_model = SupervisedLearning.load_from_checkpoint(
        args.ckpt_path_multimodal, encoder=encoder, output_dim=dataset.n_classes
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
    audio_results_dir = os.path.join(
        main_results_dir, "music_only", args.audio_model_version, ""
    )
    multimodal_results_dir = os.path.join(
        main_results_dir, "multimodal", args.multimodal_model_version, ""
    )
    os.makedirs(main_results_dir, exist_ok=True)
    os.makedirs(audio_results_dir, exist_ok=True)
    os.makedirs(multimodal_results_dir, exist_ok=True)

    # extract MagnaTagATune tag categories for final evaluation:
    all_tags = list(dataset.label2idx.keys())
    if final_eval and args.dataset == "magnatagatune":
        mtat_tags = yaml_config_hook(mtag_tags_file)
        # sort and validate tag categories:
        for category, tags in mtat_tags.items():
            mtat_tags[category] = sorted(tags)
            if not (set(tags) <= set(all_tags)):
                raise ValueError("One or more tags in {} category is invalid.")

    # set tag groups:
    if final_eval:
        if args.dataset == "magnatagatune":
            tag_groups = mtat_tags
            tag_groups["all_tags"] = all_tags
        else:
            tag_groups = "all"
            # tag_groups = {"": all_tags}
    else:
        tag_groups = "all"
        # tag_groups = {"": all_tags}

    # evaluate supervised model pretrained on audio only:
    if verbose:
        print("\n\nRunning evaluation for music only model...\n")
    audio_metrics = evaluate(
        audio_model,
        dataset=dataset,
        dataset_name=args.dataset,
        audio_length=args.audio_length,
        overlap_ratios=args.song_split_overlap_ratios,
        output_dir=audio_results_dir,
        agg_methods=args.aggregation_methods,
        tag_groups=tag_groups,
        device=device,
        verbose=verbose,
    )

    # evaluate supervised model pretrained on audio + video:
    if verbose:
        print("\n\nRunning evaluation for multimodal model...\n")
    multimodal_metrics = evaluate(
        multimodal_model,
        dataset=dataset,
        dataset_name=args.dataset,
        audio_length=args.audio_length,
        overlap_ratios=args.song_split_overlap_ratios,
        output_dir=multimodal_results_dir,
        agg_methods=args.aggregation_methods,
        tag_groups=tag_groups,
        device=device,
        verbose=verbose,
    )

    # -------------------
    # METRIC FILE UPDATES
    # -------------------

    if verbose:
        print("\n\nUpdating metric files...")

    # update audio model's parent dictionary (contains metrics for single model, single modality, all model versions):
    audio_parent_metrics_file = os.path.join(
        main_results_dir, "music_only", "global_metrics.json"
    )
    # load dictionary if it already exists:
    if os.path.isfile(audio_parent_metrics_file):
        with open(audio_parent_metrics_file, "r") as json_file:
            audio_parent_metrics = json.load(json_file)
    # else create new dictionary:
    else:
        audio_parent_metrics = {}
    # add entry and write back to file:
    audio_parent_metrics[args.audio_model_version] = audio_metrics
    with open(audio_parent_metrics_file, "w") as json_file:
        json.dump(audio_parent_metrics, json_file, indent=3)

    # update multimodal model's parent dictionary (contains metrics for single model, single modality, all model versions):
    multimodal_parent_metrics_file = os.path.join(
        main_results_dir, "multimodal", "global_metrics.json"
    )
    # load dictionary if it already exists:
    if os.path.isfile(multimodal_parent_metrics_file):
        with open(multimodal_parent_metrics_file, "r") as json_file:
            multimodal_parent_metrics = json.load(json_file)
    # else create new dictionary:
    else:
        multimodal_parent_metrics = {}
    # add entry and write back to file:
    multimodal_parent_metrics[args.multimodal_model_version] = multimodal_metrics
    with open(multimodal_parent_metrics_file, "w") as json_file:
        json.dump(multimodal_parent_metrics, json_file, indent=3)

    # update grandparent dictionary (contains metrics for single model, all modalities, all model versions):
    grandparent_metrics_file = os.path.join(main_results_dir, "global_metrics.json")
    # load dictionary if it already exists:
    if os.path.isfile(grandparent_metrics_file):
        with open(grandparent_metrics_file, "r") as json_file:
            grandparent_metrics = json.load(json_file)
    # else create new dictionary:
    else:
        grandparent_metrics = {}
    # add entries and write back to file:
    grandparent_metrics["audio"] = audio_parent_metrics
    grandparent_metrics["multimodal"] = multimodal_parent_metrics
    with open(grandparent_metrics_file, "w") as json_file:
        json.dump(grandparent_metrics, json_file, indent=3)

    # update great-grandparent dictionary (contains metrics for all models, all modalities, all model versions):
    great_grandparent_metrics_file = os.path.join(
        args.results_dir, args.dataset, "global_metrics.json"
    )
    # load dictionary if it already exists:
    if os.path.isfile(great_grandparent_metrics_file):
        with open(great_grandparent_metrics_file, "r") as json_file:
            great_grandparent_metrics = json.load(json_file)
    # else create new dictionary:
    else:
        great_grandparent_metrics = {}
    # add entry and write back to file:
    great_grandparent_metrics[args.model_name] = grandparent_metrics
    with open(great_grandparent_metrics_file, "w") as json_file:
        json.dump(great_grandparent_metrics, json_file, indent=3)

    print("\n\n")

