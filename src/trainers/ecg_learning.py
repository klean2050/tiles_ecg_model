import torch, torch.nn as nn
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.utils import CCCLoss, mean_ccc


class ECGLearning(LightningModule):
    def __init__(self, args, encoder, output_dim):
        super().__init__()
        self.save_hyperparameters(args)
        self.ground_truth = args.gtruth
        self.accuracy = accuracy_score
        self.bs = args.batch_size
        self.args = args

        # freezing trained ECG encoder
        self.encoder = encoder
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = self.args.unfreeze

        # create cls projector
        self.n_features = self.encoder.output_size
        self.project_cls = nn.Sequential(
            nn.Linear(self.n_features, self.hparams.projection_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hparams.projection_dim, output_dim),
        )
        # putting it all together
        self.model = nn.Sequential(self.encoder, self.project_cls)

        self.validation_true = list()
        self.validation_pred = list()

    def forward(self, x, y):
        preds = self.model(x.unsqueeze(1))
        preds = preds.squeeze() if preds.shape[0] != 1 else preds
        if (
            "ptb" in self.args.dataset_dir
            or "avec" in self.args.dataset_dir
            or "epic" in self.args.dataset_dir
        ):
            y = y.float()
        else:
            y = y.long()
            
        return preds, y

    def training_step(self, batch, _):
        data, y, _ = batch
        preds, y = self.forward(data, y)
        loss = self.compute_loss(preds, y)
        self.log("Train/loss", loss, sync_dist=True, batch_size=self.bs)
        return loss

    def validation_epoch_end(self, _):
        if "avec" in self.args.dataset_dir or "epic" in self.args.dataset_dir:
            cccloss = self.compute_loss(
                torch.stack(self.validation_pred), torch.stack(self.validation_true)
            )
            ccc = mean_ccc(self.validation_pred, self.validation_true)
            self.log("Valid/ccc", ccc, sync_dist=True, batch_size=self.bs)
            self.log("Valid/cccloss", cccloss, sync_dist=True, batch_size=self.bs)
            self.validation_true = list()
            self.validation_pred = list()

    def validation_step(self, batch, _):
        data, y, _ = batch
        preds, y = self.forward(data, y)
        loss = self.compute_loss(preds, y, val=True)

        if "ptb" in self.args.dataset_dir:
            auroc = roc_auc_score(y.cpu(), preds.cpu())
            preds = (torch.sigmoid(preds).detach() > 0.5) * 1
            f1 = f1_score(y.cpu(), preds.cpu(), average="macro", zero_division=0)
            self.log("Valid/f1", f1, sync_dist=True, batch_size=self.bs)
            self.log("Valid/auroc", auroc, sync_dist=True, batch_size=self.bs)
        elif "avec" in self.args.dataset_dir or "epic" in self.args.dataset_dir:
            for idx in range(len(preds.cpu())):
                self.validation_pred.append(preds.cpu()[idx])
                self.validation_true.append(y.cpu()[idx])

        else:
            y, preds = y.long(), preds.argmax(dim=1)
            acc = accuracy_score(y.cpu(), preds.cpu())
            f1 = f1_score(y.cpu(), preds.cpu(), average="macro", zero_division=0)
            self.log("Valid/f1", f1, sync_dist=True, batch_size=self.bs)
            self.log("Valid/acc", acc, sync_dist=True, batch_size=self.bs)

        self.log("Valid/loss", loss, sync_dist=True, batch_size=self.bs)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=float(self.hparams.weight_decay),
        )
        return {"optimizer": optimizer}

    def compute_loss(self, preds, y, val=False):
        if "ptb" in self.args.dataset_dir:
            loss = nn.BCEWithLogitsLoss()
        elif "avec" in self.args.dataset_dir or "epic" in self.args.dataset_dir:
            loss = CCCLoss()
        else:
            weight = None if val else len(y) / torch.bincount(y)
            loss = nn.CrossEntropyLoss(weight=weight)
        return loss(preds, y)
