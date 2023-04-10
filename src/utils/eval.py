import os, torch, numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def evaluate(model, dataset, dataset_name, aggregate="majority"):
    """
    Performs evaluation of trained supervised models
    Args:
        model (pytorch_lightning.LightningModule): Supervised model to evaluate.
        dataset (torch.utils.data.Dataset): Test dataset.
        dataset_name (str): Name of dataset.
        aggregate (str): Method to aggregate instance-level outputs
    Returns:
        global_metrics (dict): Dictionary containing performance metrics
    """
    model.eval()
    print("Evaluating model...")

    y_true, y_pred, y_name = [], [], []
    for ecg, label, name in tqdm(dataset):
        # move inputs to device
        ecg = ecg.to(model.device)
        label = label.to(model.device)
        with torch.no_grad():
            preds, _ = model(ecg, label)
            if "ptb" not in dataset_name:
                preds = preds.argmax(dim=1).detach()

            # save labels and predictions
            y_true.append(label)
            y_pred.append(preds)
            y_name.extend(name)

    # stack arrays for evaluation
    y_true = torch.cat(y_true, dim=0).squeeze().cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).squeeze().cpu().numpy()
    y_name = np.array(y_name)

    # chunk-wise predictions
    if "ptb" in dataset_name:
        auc = roc_auc_score(y_true, y_pred)
        y_pred = 1 / (1 + np.exp(-y_pred)) > 0.5
        f1 = f1_score(y_true, y_pred * 1, average="macro", zero_division=0)
        print(f"Chunk-wise results for {dataset_name}:")
        print({"AUROC": auc, "F1-macro": f1})
    else:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print(f"Chunk-wise results for {dataset_name}:")
        print({"Accuracy": acc, "F1-macro": f1})

    # aggregate predictions
    y_true_agg, y_pred_agg = [], []
    if "ptb" in dataset_name:
        y_true_agg = y_true
        y_pred_agg = y_pred

        auc = roc_auc_score(y_true_agg, y_pred_agg)
        y_pred_agg = 1 / (1 + np.exp(-y_pred_agg)) > 0.5
        f1 = f1_score(y_true_agg, y_pred_agg * 1, average="macro", zero_division=0)
        print(f"Aggregate results for {dataset_name}:")
        print({"AUROC": auc, "F1-macro": f1})
        return {"AUROC": auc, "F1-macro": f1}
    else:
        for name in np.unique(y_name):
            idx = np.where(y_name == name)[0]
            these_preds = y_pred[idx]
            these_labels = y_true[idx]
            for label in np.unique(these_labels):
                y_true_agg.append(label)
                fidx = np.where(these_labels == label)[0]
                if aggregate == "majority":
                    y_pred_agg.append(np.bincount(these_preds[fidx]).argmax())
                else:
                    raise NotImplementedError

        acc = accuracy_score(y_true_agg, y_pred_agg)
        f1 = f1_score(y_true_agg, y_pred_agg, average="macro", zero_division=0)
        print(f"Aggregate results for {dataset_name}:")
        print({"Accuracy": acc, "F1-macro": f1})
        return {"Accuracy": acc, "F1-macro": f1}
