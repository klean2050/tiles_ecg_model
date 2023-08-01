import torch, torch.nn as nn, numpy as np
from torch import Tensor
from pytorch_lightning import LightningModule

class TransformLearning(LightningModule):
    def __init__(self, args, encoder: nn.Module):
        super().__init__()
        self.save_hyperparameters(args)

        # backbone encoder
        self.encoder = encoder
        # dimensionality of representation
        self.n_features = self.encoder.output_size
        # projection head
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, self.hparams.projection_dim, bias=False),
        )
        self.model = nn.Sequential(self.encoder, self.projector)

        # criterion function
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        preds = self.model(x)
        return self.criterion(preds, y)

    def training_step(self, batch, _) -> Tensor:
        x, y = batch
        loss = self.forward(x, y)
        self.log("Train/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, _) -> Tensor:
        x, y = batch
        loss = self.forward(x, y)
        self.log("Valid/loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self) -> dict:
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.hparams.learning_rate)
        else:
            raise NotImplementedError
        return {"optimizer": optimizer}
