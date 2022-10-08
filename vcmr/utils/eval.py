"""Contains function to perform evaluation of supervised models on music tagging."""


import os
import torch
import torch.nn.functional as F
from sklearn import metrics
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from typing import Union, Any, List


# supported values of certain parameters:
ALL_DATASET_NAMES = ["magnatagatune", "mtg-jamendo-dataset"]
ALL_AGGREGATION_METHODS = ["average", "max", "majority_vote"]


def evaluate(model: Any, test_dataset: Any, dataset_name: str, audio_length: int, output_dir: str, agg_methods: Union[List, str] = "all", device: torch.device = None) -> None:
    """Performs evaluation of supervised models on music tagging.

    Args:
        model (pytorch_lightning.LightningModule): Supervised model to evaluate.
        test_dataset (torch.utils.data.Dataset): Test dataset.
        dataset_name (str): Name of dataset.
        audio_length (int): Length of raw audio input (in samples).
        output_dir (str): Path of directory for saving results.
        agg_methods (list | str): Method(s) to aggregate instance-level outputs of a song.
        device (torch.device): PyTorch device.
    
    Returns: None
    """

    # validate and set parameters:
    if dataset_name not in ALL_DATASET_NAMES:
        raise ValueError("Invalid dataset name.")
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
    
    # create subdirectories for saving results:
    for method in agg_methods:
        os.makedirs(os.path.join(output_dir, method, ""), exist_ok=True)
    
    # true labels, features (embeddings), and predicted labels:
    y_true = []
    features = []
    y_pred = {}
    for method in agg_methods:
        y_pred[method] = []
    
    # run inference:
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset))):
            # get label:
            _, label = test_dataset[idx]
            # get raw audio of song, split into non-overlapping segments of length audio_length:
            audio_song = test_dataset.concat_clip(idx, audio_length)
            audio_song = audio_song.squeeze(dim=1)
            assert audio_song.dim() == 3 and audio_song.size(dim=-1) == audio_length, "Error with shape of song audio."
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
            
            # save true label and song-level feature:
            y_true.append(label)
            features.append(feat_song)

            # aggregate segment outputs across song = song-level output (prediction) in various ways:
            for method in agg_methods:
                # aggregate segment outputs:
                if method == "average":
                    pred_song = torch.mean(output, dim=0)
                elif method == "max":
                    pred_song = torch.amax(output, dim=0)
                elif method == "majority_vote":
                    pred_song = torch.mean(torch.round(output), dim=0)
                
                # sanity check shapes:
                assert feat_song.dim() == 1 and feat_song.size(dim=0) == feat.size(dim=-1), "Error with shape of song-level feature."
                assert pred_song.dim() == 1 and pred_song.size(dim=0) == output.size(dim=-1), "Error with shape of song-level output."

                # save predicted label (song-level output):
                y_pred[method].append(pred_song)
    
    # convert lists to numpy arrays:
    y_true = torch.stack(y_true, dim=0).cpu().numpy()
    features = torch.stack(features, dim=0).cpu().numpy()
    for method in agg_methods:
        y_pred[method] = torch.stack(y_pred[method], dim=0).cpu().numpy()
    # sanity check shapes:
    assert y_true.ndim == 2, "Error with shape of true labels."
    assert features.ndim == 2, "Error with shape of song-level features."
    for method in agg_methods:
        assert y_pred[method].ndim == 2, "Error with shape of predicted labels."
        assert y_pred[method].shape == y_true.shape, "Error with shape of true and/or predicted labels."
    
    # save true labels and song-level features:
    np.save(os.path.join(output_dir, "labels.npy"), y_true)
    np.save(os.path.join(output_dir, "features.npy"), features)

    # compute performance metrics:
    if dataset_name in ["magnatagatune", "mtg-jamendo-dataset"]:
        # get label names:
        label_names = list(test_dataset.dataset.label2idx.keys())
        label_names = [name.split("---")[-1] for name in label_names]     # old code, probably not necessary
        # sanity check length:
        assert len(label_names) == y_true.shape[-1], "Error with length of label names."

        # loop over aggregation methods:
        global_metrics = {}
        for method in agg_methods:
            # compute global metrics (average across all tags):
            global_roc = metrics.roc_auc_score(y_true, y_pred[method], average="macro")
            global_precision = metrics.average_precision_score(y_true, y_pred[method], average="macro")
            # save to json file:
            global_metrics_dict = {
                "ROC-AUC": global_roc,
                "PR-AUC": global_precision
            }
            with open(os.path.join(output_dir, method, "global_metrics.json"), "w") as json_file:
                json.dump(global_metrics_dict, json_file)
            # save to parent dictionary:
            global_metrics[method] = global_metrics_dict

            # compute tag-wise metrics:
            tag_roc = metrics.roc_auc_score(y_true, y_pred[method], average=None)
            tag_precision = metrics.average_precision_score(y_true, y_pred[method], average=None)
            # save to csv file:
            tag_metrics_dict = {name: {"ROC-AUC": roc, "PR-AUC": precision} for name, roc, precision in zip(label_names, tag_roc, tag_precision)}
            tag_metrics_df = pd.DataFrame.from_dict(tag_metrics_dict, orient="index")
            tag_metrics_df.to_csv(os.path.join(output_dir, method, "tag_metrics.csv"), index_label="tag")
        # save dictionary containing metrics for all methods to json file:
        with open(os.path.join(output_dir, "global_metrics.json"), "w") as json_file:
            json.dump(global_metrics, json_file)
    else:
        raise ValueError("Invalid dataset name.")
    
    # OLD CODE:
    """
    else:
        y_pred = torch.stack(y_pred, dim=0)
        _, y_pred = torch.max(y_pred, 1)
        accuracy = metrics.accuracy_score(y_true, y_pred)
        print({"Accuracy": accuracy})
    """

