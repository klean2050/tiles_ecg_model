"""Contains function to perform evaluation of supervised models on music tagging."""


import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
import pickle
from tqdm import tqdm
from typing import Any


def evaluate(network: Any, test_dataset: Any, dataset_name: str, audio_length: int, output_dir: str, aggregation_method: str = "average", device: torch.device = None) -> None:
    """Performs evaluation of supervised models on music tagging.

    Args:
        network (pytorch_lightning.LightningModule): Supervised model to evaluate.
        test_dataset (torch.utils.data.Dataset): Test dataset.
        dataset_name (str): Name of dataset.
        audio_length (int): Length of raw audio input (in samples).
        output_dir (str): Path of directory for saving results.
        aggregation_method (str): Method to aggregate instance-level outputs of a song.
            Supported values: "average"
        device (torch.device): PyTorch device.
    
    Returns: None
    """

    if aggregation_method not in ["average"]:
        raise ValueError("Invalid aggregation method.")
    if device is None:
        device = torch.device("cuda")

    est_array, gt_array = [], []
    features = []

    network = network.to(device)
    network.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset))):
            _, label = test_dataset[idx]
            batch = test_dataset.concat_clip(idx, audio_length)
            batch = batch.squeeze(1).to(device)

            output = network.model(batch)
            feat = network.encoder(batch)

            if dataset_name in ["magnatagatune", "mtg-jamendo-dataset"]:
                output = torch.sigmoid(output)
            else:
                output = F.softmax(output, dim=1)

            if aggregation_method == "average":
                track_prediction = output.mean(dim=0)
                features.append(feat.mean(dim=0))
            est_array.append(track_prediction)
            gt_array.append(label)

    features = torch.stack(features, dim=0).cpu().numpy()
    est_array = torch.stack(est_array, dim=0).cpu().numpy()
    gt_array = torch.stack(gt_array, dim=0).cpu().numpy()

    np.save(os.path.join(output_dir, "features.npy"), features)
    np.save(os.path.join(output_dir, "labels.npy"), gt_array)

    if dataset_name in ["magnatagatune", "mtg-jamendo-dataset"]:
        overall_dict = {
            "PR-AUC": metrics.average_precision_score(
                gt_array, est_array, average="macro"
            ),
            "ROC-AUC": metrics.roc_auc_score(gt_array, est_array, average="macro"),
        }
        with open(os.path.join(output_dir, "overall_dict.pickle"), "wb") as fp:
            pickle.dump(overall_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

        labels = test_dataset.dataset.label2idx.keys()
        labels = [name.split("---")[-1] for name in labels]

        prs = metrics.average_precision_score(gt_array, est_array, average=None)
        rcs = metrics.roc_auc_score(gt_array, est_array, average=None)
        classes_dict = {name: [v1, v2] for name, v1, v2 in zip(labels, prs, rcs)}
        with open(os.path.join(output_dir, "classes_dict.pickle"), "wb") as fp:
            pickle.dump(classes_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        est_array = torch.stack(est_array, dim=0)
        _, est_array = torch.max(est_array, 1)
        accuracy = metrics.accuracy_score(gt_array, est_array)
        print({"Accuracy": accuracy})

