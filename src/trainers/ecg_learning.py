import torch, torch.nn as nn
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class ECGLearning(LightningModule):
    def __init__(self, args, encoder, output_dim):
        super().__init__()
        self.save_hyperparameters(args)
        self.ground_truth = args.gtruth
        self.accuracy = accuracy_score
        self.bs = args.batch_size
        self.args = args

        # configure criterion
        if "drivedb" in args.dataset_dir:
            self.loss = nn.MSELoss()
        elif "ptb" in args.dataset_dir:
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.CrossEntropyLoss()

        # freezing trained ECG encoder
        self.encoder = encoder
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

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

    def forward(self, x, y):
        preds = self.model(x.unsqueeze(1))
        preds = preds.squeeze() if preds.shape[0] != 1 else preds
        y = y.float() if "ptb" in self.args.dataset_dir else y.long()
        return self.loss(preds, y), preds

    def training_step(self, batch, _):
        data, y, _ = batch
        loss, _ = self.forward(data, y)
        self.log("Train/loss", loss, sync_dist=True, batch_size=self.bs)
        return loss

    def validation_step(self, batch, _):
        data, y, _ = batch
        loss, preds = self.forward(data, y)

        if "ptb" in self.args.dataset_dir:
            auroc = roc_auc_score(y.cpu(), preds.cpu())
            self.log("Valid/auroc", auroc, sync_dist=True, batch_size=self.bs)
        else:
            acc = accuracy_score(y.cpu(), preds.cpu().argmax(dim=1))
            self.log("Valid/acc", acc, sync_dist=True, batch_size=self.bs)

        preds = preds.argmax(dim=1).detach().cpu().numpy()
        y = y.argmax(dim=1) if "ptb" in self.args.dataset_dir else y

        f1 = f1_score(y.cpu(), preds, average="macro", zero_division=0)
        self.log("Valid/f1", f1, sync_dist=True, batch_size=self.bs)
        self.log("Valid/loss", loss, sync_dist=True, batch_size=self.bs)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=float(self.hparams.weight_decay),
        )
        return {"optimizer": optimizer}
