"""Contains function to perform evaluation of supervised models on music tagging."""


import os, torch, json
import torch.nn.functional as F
import numpy as np, pandas as pd
from tqdm import tqdm
from sklearn import metrics
from typing import Union, Any, List, Dict, Tuple

from vcmr.loaders import SongSplitter


# supported values of certain parameters:
ALL_DATASET_NAMES = ["magnatagatune", "mtg-jamendo-dataset"]
ALL_AGGREGATION_METHODS = ["average", "max", "majority_vote"]
ALL_MODEL_MODALITIES = ["audio", "multimodal"]
ALL_METRIC_TYPES = ["ROC-AUC", "PR-AUC"]


def evaluate(
    model: Any,
    dataset: Any,
    dataset_name: str,
    audio_length: int,
    overlap_ratios: List,
    output_dir: str,
    agg_methods: Union[List, str] = "all",
    tag_groups: Union[Dict, str] = "all",
    device: torch.device = None,
    verbose: bool = False,
) -> Dict:
    """Performs evaluation of supervised models on music tagging.
    Args:
        model (pytorch_lightning.LightningModule): Supervised model to evaluate.
        dataset (torch.utils.data.Dataset): Test dataset.
        dataset_name (str): Name of dataset.
        audio_length (int): Length of raw audio input (in samples).
        overlap_ratios (list): List of overlap ratios to try for splitting songs.
        output_dir (str): Path of directory for saving results.
        agg_methods (list | str): Method(s) to aggregate instance-level outputs of a song.
        tag_groups (dict | str): Tag groups to evaluate.
        device (torch.device): PyTorch device.
        verbose (str): Verbosity.
    
    Returns:
        global_metrics (dict): Dictionary containing performance metrics for all overlap ratio values and all methods.
    """

    # validate and set parameters:
    if dataset_name not in ALL_DATASET_NAMES:
        raise ValueError("Invalid dataset name.")
    for overlap in overlap_ratios:
        if overlap < 0 or overlap > 0.9:
            raise ValueError("Invalid overlap ratio value.")
    if device is None:
        device = torch.device("cuda")
    if type(agg_methods) == str:
        if agg_methods == "all":
            agg_methods = ALL_AGGREGATION_METHODS
        elif agg_methods in ALL_AGGREGATION_METHODS:
            agg_methods = [agg_methods]
        else:
            raise ValueError("Invalid aggregation method.")
    elif type(agg_methods) == list:
        for method in agg_methods:
            if method not in ALL_AGGREGATION_METHODS:
                raise ValueError("Invalid aggregation method.")
        agg_methods = agg_methods
    else:
        raise ValueError("agg_methods is of an invalid data type.")
    if type(tag_groups) == str:
        if tag_groups == "all":
            tag_groups = {"all_tags": list(dataset.label2idx.keys())}
        else:
            raise ValueError("Invalid tag_groups value.")
    elif type(tag_groups) != dict:
        raise ValueError("tag_groups is of an invalid data type.")
    
    # create subdirectories for saving results:
    for overlap in overlap_ratios:
        for method in agg_methods:
            for category in tag_groups.keys():
                os.makedirs(
                    os.path.join(output_dir, f"overlap={overlap}", method, category, ""),
                    exist_ok=True,
                )
    
    # true labels, features (embeddings), and predicted labels:
    y_true = []
    features = {}
    for overlap in overlap_ratios:
        features[overlap] = []
    y_pred = {}
    for overlap in overlap_ratios:
        y_pred[overlap] = {}
        for method in agg_methods:
            y_pred[overlap][method] = []

    # run inference:
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        # loop over overlap ratio values:
        for overlap in overlap_ratios:
            if verbose:
                print("Running inference for overlap_ratio = {}...".format(overlap))

            # create wrapper dataset for splitting songs:
            test_dataset = SongSplitter(
                dataset, audio_length=audio_length, overlap_ratio=overlap
            )
            # clear true labels (only last overlap iteration is kept):
            y_true = []

            # run inference:
            for idx in tqdm(range(len(test_dataset))):
                # get audio of song (split into segments of length audio_length) and label:
                audio_song, label = test_dataset[idx]
                assert (
                    audio_song.dim() == 3
                    and audio_song.size(dim=1) == 1
                    and audio_song.size(dim=-1) == audio_length
                ), "Error with shape of song audio."
                audio_song = audio_song.to(device)

                # pass song through model to get features (embeddings) and outputs (logits) of each segment:
                feat = model.encoder(audio_song)
                output = model.model(audio_song)

                # transform raw logits to probabilities:
                if dataset_name in ["magnatagatune", "mtg-jamendo-dataset"]:
                    output = torch.sigmoid(output)
                else:
                    output = F.softmax(output, dim=1)

                # average segment features across song = song-level feature:
                feat_song = feat.mean(dim=0)
                assert feat_song.dim() == 1 and feat_song.size(dim=0) == feat.size(
                    dim=-1
                ), "Error with shape of song-level feature."

                # save true label and song-level feature:
                y_true.append(label)
                features[overlap].append(feat_song)

                # loop over aggregation methods:
                for method in agg_methods:
                    # aggregate segment outputs across song = song-level output (prediction):
                    if method == "average":
                        pred_song = torch.mean(output, dim=0)
                    elif method == "max":
                        pred_song = torch.amax(output, dim=0)
                    elif method == "majority_vote":
                        pred_song = torch.mean(torch.round(output), dim=0)

                    # sanity check shape:
                    assert pred_song.dim() == 1 and pred_song.size(
                        dim=0
                    ) == output.size(dim=-1), "Error with shape of song-level output."

                    # save predicted label (song-level output):
                    y_pred[overlap][method].append(pred_song)

    # convert lists to numpy arrays:
    y_true = torch.stack(y_true, dim=0).cpu().numpy()
    for overlap in overlap_ratios:
        features[overlap] = torch.stack(features[overlap], dim=0).cpu().numpy()
        for method in agg_methods:
            y_pred[overlap][method] = (
                torch.stack(y_pred[overlap][method], dim=0).cpu().numpy()
            )

    # sanity check shapes:
    assert y_true.ndim == 2, "Error with shape of true labels."
    for overlap in overlap_ratios:
        assert features[overlap].ndim == 2, "Error with shape of song-level features."
        for method in agg_methods:
            assert (
                y_pred[overlap][method].ndim == 2
            ), "Error with shape of predicted labels."
            assert (
                y_pred[overlap][method].shape == y_true.shape
            ), "Error with shape of true and/or predicted labels."

    # save true labels and song-level features:
    np.save(os.path.join(output_dir, "labels.npy"), y_true)
    for overlap in overlap_ratios:
        np.save(
            os.path.join(output_dir, f"overlap={overlap}", "features.npy"), features
        )

    # compute performance metrics:
    if dataset_name in ["magnatagatune", "mtg-jamendo-dataset"]:

        global_metrics = {}
        # loop over overlap ratio values:
        for overlap in overlap_ratios:
            global_metrics[overlap] = {}
            # loop over aggregation methods:
            for method in agg_methods:
                global_metrics[overlap][method] = {}
                # loop over tag groups:
                for category, tags in tag_groups.items():
                    # convert tag names to tag indices:
                    tag_indices = [dataset.label2idx[tag] for tag in tags]

                    # compute global metrics (average across tags in tag group):
                    try:
                        global_roc = metrics.roc_auc_score(
                            y_true[:, tag_indices], y_pred[overlap][method][:, tag_indices], average="macro"
                        )
                        global_precision = metrics.average_precision_score(
                            y_true[:, tag_indices], y_pred[overlap][method][:, tag_indices], average="macro"
                        )
                    except:
                        if verbose:
                            print("Warning: at least 1 global metric was not able to be computed.")
                        global_roc = np.nan
                        global_precision = np.nan
                    # save to json file (rounded to 4 decimal places):
                    global_metrics_dict = {
                        "ROC-AUC": np.around(global_roc, decimals=4),
                        "PR-AUC": np.around(global_precision, decimals=4),
                    }
                    with open(
                        os.path.join(
                            output_dir, f"overlap={overlap}", method, category, "global_metrics.json"
                        ),
                        "w",
                    ) as json_file:
                        json.dump(global_metrics_dict, json_file, indent=3)
                    # save to higher-level dictionary:
                    global_metrics[overlap][method][category] = global_metrics_dict

                    # compute tag-wise metrics:
                    try:
                        tag_roc = metrics.roc_auc_score(
                            y_true[:, tag_indices], y_pred[overlap][method][:, tag_indices], average=None
                        )
                        tag_precision = metrics.average_precision_score(
                            y_true[:, tag_indices], y_pred[overlap][method][:, tag_indices], average=None
                        )
                        # save to csv file:
                        tag_metrics_dict = {
                            name: {"ROC-AUC": roc, "PR-AUC": precision}
                            for name, roc, precision in zip(tags, tag_roc, tag_precision)
                        }
                        tag_metrics_df = pd.DataFrame.from_dict(
                            tag_metrics_dict, orient="index"
                        )
                        tag_metrics_df.to_csv(
                            os.path.join(
                                output_dir, f"overlap={overlap}", method, category, "tag_metrics.csv"
                            ),
                            index_label="tag",
                        )
                    except:
                        if verbose:
                            print("Warning: at least 1 tag-wise metric was not able to be computed.")
                        pass
                
                # save dictionary containing metrics for single overlap ratio, single method, all tag groups to json file:
                with open(
                    os.path.join(output_dir, f"overlap={overlap}", method, "global_metrics.json"),
                    "w",
                ) as json_file:
                    json.dump(global_metrics[overlap][method], json_file, indent=3)
            
            # save dictionary containing metrics for single overlap ratio, all methods, all tag groups to json file:
            with open(
                os.path.join(output_dir, f"overlap={overlap}", "global_metrics.json"),
                "w",
            ) as json_file:
                json.dump(global_metrics[overlap], json_file, indent=3)
        
        # save dictionary containing metrics for all overlap ratios, all methods, all tag groups to json file:
        with open(os.path.join(output_dir, "global_metrics.json"), "w") as json_file:
            json.dump(global_metrics, json_file, indent=3)
    else:
        raise ValueError("Dataset not supported.")
    
    return global_metrics


def find_best_model(
    global_metrics: Dict,
    model_modality: str,
    metric_type: str
) -> Tuple[Tuple, float]:
    """Performs analysis of music tagging performance.

    Args:
        global_metrics (dict): Dictionary containing performance metrics on a dataset.
        model_modality (str): Modality of model to search over.
        metric_type (str): Performance metric to optimize with respect to.

    Returns:
        best_model (tuple): Tuple describing best model.
        best_score (float): Best score (of specificed metric).
    """

    # validate and set parameters:
    if model_modality not in ALL_MODEL_MODALITIES:
        raise ValueError("Invalid model modality.")
    if metric_type not in ALL_METRIC_TYPES:
        raise ValueError("Invalid performance metric.")

    # find best-performing model for a single model modality and a single performance metric:
    best_model = None
    best_score = 0
    for model in global_metrics.keys():
        # for modality in model_modality     # skip loop since we only consider one model modality
        for version in global_metrics[model][model_modality].keys():
            for overlap in global_metrics[model][model_modality][version].keys():
                for method in global_metrics[model][model_modality][version][
                    overlap
                ].keys():
                    # update best model/score:
                    if (
                        global_metrics[model][model_modality][version][overlap][method][
                            metric_type
                        ]
                        > best_score
                    ):
                        best_model = (model, version, f"overlap={overlap}", method)
                        best_score = global_metrics[model][model_modality][version][
                            overlap
                        ][method][metric_type]

    return best_model, best_score

