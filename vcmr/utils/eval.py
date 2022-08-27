import torch, torch.nn as nn, pickle
import torch.nn.functional as F, numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn import metrics


def evaluate(
    network: nn.Module,
    test_dataset: Dataset,
    dataset_name: str,
    audio_length: int,
    device,
):

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
            
            if dataset_name.split("_")[0] in ["magnatagatune", "mtg-jamendo-dataset"]:
                output = torch.sigmoid(output)
            else:
                output = F.softmax(output, dim=1)

            track_prediction = output.mean(dim=0)
            features.append(feat.mean(dim=0))
            est_array.append(track_prediction)
            gt_array.append(label)

    features = torch.stack(features, dim=0).cpu().numpy()
    est_array = torch.stack(est_array, dim=0).cpu().numpy()
    gt_array = torch.stack(gt_array, dim=0).cpu().numpy()

    np.save(f"data/{dataset_name}_fts.npy", features)
    np.save(f"data/{dataset_name}_lbs.npy", gt_array)

    if dataset_name.split("_")[0] in ["magnatagatune", "mtg-jamendo-dataset"]:
        overall_dict = {
            "PR-AUC": metrics.average_precision_score(gt_array, est_array, average="macro"),
            "ROC-AUC": metrics.roc_auc_score(gt_array, est_array, average="macro"),
        }
        with open(f"data/{dataset_name}_overall_dict.pickle", "wb") as fp:
            pickle.dump(overall_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

        labels = test_dataset.dataset.label2idx.keys()
        labels = [name.split("---")[-1] for name in labels]

        prs = metrics.average_precision_score(gt_array, est_array, average=None)
        rcs = metrics.roc_auc_score(gt_array, est_array, average=None)
        classes_dict = {
            name: [v1, v2] for name, v1, v2 in zip(labels, prs, rcs)
        }
        with open(f"data/{dataset_name}_classes_dict.pickle", "wb") as fp:
            pickle.dump(classes_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        est_array = torch.stack(est_array, dim=0)
        _, est_array = torch.max(est_array, 1)
        accuracy = metrics.accuracy_score(gt_array, est_array)
        print({"Accuracy": accuracy})