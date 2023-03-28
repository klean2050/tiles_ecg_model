import os, torch, numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


def evaluate(model, dataset, dataset_name, output=None, aggregate="majority"):
    """
    Performs evaluation of trained supervised models
    Args:
        model (pytorch_lightning.LightningModule): Supervised model to evaluate.
        dataset (torch.utils.data.Dataset): Test dataset.
        dataset_name (str): Name of dataset.
        output_dir (str | None): Path of directory for saving results, if any
        aggregate (str): Method to aggregate instance-level outputs
    Returns:
        global_metrics (dict): Dictionary containing performance metrics
    """
    model.eval()
    with torch.no_grad():
        print("Evaluating model...")
        y_true, y_pred, y_name = [], [], []
        for idx in tqdm(range(len(dataset))):
            ecg, label, name = dataset[idx]
            ecg = torch.Tensor(ecg).to(model.device)
            label = torch.Tensor([label]).to(model.device)

            # pass sample through model
            _, preds = model(ecg.unsqueeze(0), label)
            preds = preds.argmax(dim=1).detach()

            # save labels and predictions
            y_true.append(label)
            y_pred.append(preds)
            y_name.append(name)

    # stack arrays for evaluation
    y_true = torch.stack(y_true, dim=0).squeeze().cpu().numpy()
    y_pred = torch.stack(y_pred, dim=0).squeeze().cpu().numpy()
    y_name = np.array(y_name)

    # aggregate predictions
    y_true_agg, y_pred_agg = [], []
    for name in np.unique(y_name):
        idx = np.where(y_name == name)[0]
        for label in np.unique(y_true[idx]):
            fidx = np.where(y_true[idx] == label)[0]
            y_true_agg.append(label)
            if aggregate == "majority":
                y_pred_agg.append(np.bincount(y_pred[fidx]).argmax())
            else:
                raise NotImplementedError

    # evaluate metrics in dict
    acc = accuracy_score(y_true_agg, y_pred_agg)
    f1 = f1_score(y_true_agg, y_pred_agg, average="macro", zero_division=0)
    print(f"k-fold results for {dataset_name}:")
    print({"Accuracy": acc, "F1-macro": f1})

    # save predictions
    if output is not None:
        os.makedirs(output.split("/")[0], exist_ok=True)
        with open(output, "w") as f:
            string = f"Accuracy: {acc:.4f}\nF1-macro: {f1:.4f}"
            f.write(string)
