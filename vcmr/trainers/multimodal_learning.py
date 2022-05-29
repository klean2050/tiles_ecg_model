import torch, torch.nn as nn
from torch import Tensor
from simclr.modules import NT_Xent, LARS
from pytorch_lightning import LightningModule


class MultimodalLearning(LightningModule):
    def __init__(self, args, enc1: nn.Module, ckpt=None):
        super().__init__()
        self.save_hyperparameters(args)
        self.criterion = self.configure_criterion()

        self.n_features = 512
        self.encoder1 = ckpt.model.encoder if ckpt else enc1
        self.projector1 = nn.Linear(self.n_features, self.hparams.projection_dim)

        # keep first 4 conv blocks frozen
        c0, c1 = 0, 0
        for child in self.encoder1.children():
            for block in child.children():
                for param in block.parameters():
                    param.requires_grad = False
                c1 += 1
                if c1 > 4:
                    break
            c0 += 1
            if c0 > 0:
                break

        self.temporal = nn.LSTM(2048, 512, batch_first=True, dropout=0.2)
        self.encoder2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.n_features),
        )
        self.projector2 = nn.Linear(self.n_features, self.hparams.projection_dim)

    def forward(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        z_i = self.projector1(self.encoder1(x_i))
        x_j, _ = self.temporal(x_j)
        z_j = self.projector2(self.encoder2(x_j))
        return self.criterion(z_i, z_j)

    def training_step(self, batch, _) -> Tensor:
        x_i, x_j, _ = batch
        loss = self.forward(x_i, x_j)
        self.log("Train/loss", loss)
        return loss

    def configure_criterion(self) -> nn.Module:
        # PT lightning aggregates differently in DP mode
        if self.hparams.accelerator == "dp" and self.hparams.gpus:
            batch_size = int(self.hparams.batch_size / self.hparams.gpus)
        else:
            batch_size = self.hparams.batch_size

        criterion = NT_Xent(batch_size, self.hparams.temperature, world_size=1)
        return criterion

    def configure_optimizers(self) -> dict:
        scheduler = None
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4)#self.hparams.learning_rate)
        elif self.hparams.optimizer == "LARS":
            # optimized using LARS with linear learning rate scaling
            # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 1e−6.
            learning_rate = 0.3 * self.hparams.batch_size / 256
            optimizer = LARS(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=self.hparams.weight_decay,
                exclude_from_weight_decay=["batch_normalization", "bias"],
            )

            # "decay the learning rate with the cosine decay schedule without restarts"
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.hparams.max_epochs, eta_min=0, last_epoch=-1
            )
        else:
            raise NotImplementedError

        if scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}
