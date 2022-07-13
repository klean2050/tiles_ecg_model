import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn import metrics


def evaluate(
    encoder: nn.Module,
    test_dataset: Dataset,
    dataset_name: str,
    audio_length: int,
    device,
) -> dict:

    est_array = []
    gt_array = []

    encoder = encoder.to(device)
    encoder.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset))):
            _, label = test_dataset[idx]
            batch = test_dataset.concat_clip(idx, audio_length)
            batch = batch.squeeze(1).to(device)

            output = encoder.model(batch)
            if dataset_name in ["magnatagatune", "mtg-jamendo-dataset"]:
                output = torch.sigmoid(output)
            else:
                output = F.softmax(output, dim=1)

            track_prediction = output.mean(dim=0)
            est_array.append(track_prediction)
            gt_array.append(label)

    if dataset_name in ["magnatagatune", "mtg-jamendo-dataset"]:
        est_array = torch.stack(est_array, dim=0).cpu().numpy()
        gt_array = torch.stack(gt_array, dim=0).cpu().numpy()
        roc_aucs = metrics.roc_auc_score(gt_array, est_array, average="macro")
        pr_aucs = metrics.average_precision_score(gt_array, est_array, average="macro")
        return {
            "PR-AUC": pr_aucs,
            "ROC-AUC": roc_aucs,
        }
    else:
        est_array = torch.stack(est_array, dim=0)
        _, est_array = torch.max(est_array, 1)
        accuracy = metrics.accuracy_score(gt_array, est_array)
        return {
            "Accuracy": accuracy
        }