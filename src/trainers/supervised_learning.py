import torch, torchmetrics, torch.nn as nn
from pytorch_lightning import LightningModule


class SupervisedLearning(LightningModule):
    def __init__(self, args, encoder, output_dim=50):
        super().__init__()
        self.save_hyperparameters(args)
        self.criterion = self.configure_criterion()
        self.average_precision = torchmetrics.AveragePrecision(pos_label=1)

        self.encoder = encoder
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.projector = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, self.hparams.projection_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.projection_dim, output_dim),
        )
        self.model = nn.Sequential(self.encoder, self.projector)

    def forward(self, x, y):
        preds = self.model(x).squeeze()
        loss = self.criterion(preds, y)
        return loss, preds

    def training_step(self, batch, _):
        x, y = batch
        loss, preds = self.forward(x, y)
        self.log("Train/pr_auc", self.average_precision(preds, y))
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        loss, preds = self.forward(x, y)
        self.log("Valid/pr_auc", self.average_precision(preds, y))
        self.log("Valid/loss", loss)
        return loss

    def configure_criterion(self):
        return nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.projector.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=float(self.hparams.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=False,
        )
        if scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "Valid/loss",
            }
        else:
            return {"optimizer": optimizer}
