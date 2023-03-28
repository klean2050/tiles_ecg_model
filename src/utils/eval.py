import torch, numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


def evaluate(model, dataset, dataset_name, output_dir, aggregate, device):
    """
    Performs evaluation of trained supervised models
    Args:
        model (pytorch_lightning.LightningModule): Supervised model to evaluate.
        dataset (torch.utils.data.Dataset): Test dataset.
        dataset_name (str): Name of dataset.
        output_dir (str | None): Path of directory for saving results, if any
        aggregate (str): Method to aggregate instance-level outputs
        device (torch.device): PyTorch device.
    Returns:
        global_metrics (dict): Dictionary containing performance metrics
    """

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        
        y_true, y_pred = [], []
        for idx in tqdm(range(len(test_dataset))):
            ecg, label, name = test_dataset[idx]
            ecg = ecg.to(device)

            # pass sample through model
            ecg = ecg.unsqueeze(1)
            preds = self.model(ecg).squeeze()

            # transform logits to predictions
            preds = preds.argmax(dim=1).detach()

            # TODO: aggregate per session
            label = ...
            preds = ...

            # save true label and prediction
            y_true.append(label)
            y_pred.append(preds)

    # stack arrays for evaluation
    y_true = torch.stack(y_true, dim=0).cpu().numpy()
    y_pred = torch.stack(y_pred, dim=0).cpu().numpy()

    # evaluate metrics in dict
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print({"Accuracy": acc, "F1-macro": f1})

