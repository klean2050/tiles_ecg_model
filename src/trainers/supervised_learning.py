import torch, torch.nn as nn
from pytorch_lightning import LightningModule
from simclr.modules import NT_Xent
from src.models import ResNet1D


class SupervisedLearning(LightningModule):
    def __init__(self, args, encoder, output_dim, gtruth=0):
        super().__init__()
        self.save_hyperparameters(args)
        self.modalities = args.streams
        self.ground_truth = gtruth

        # configure criterion
        self.cls_loss = (
            nn.MSELoss() if "drivedb" in args.dataset_dir else nn.CrossEntropyLoss()
        )
        self.ssl_loss = NT_Xent(
            self.hparams.batch_size, self.hparams.temperature, world_size=1
        )

        # freezing trained ECG encoder
        self.encoder = encoder
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        # create model for other modalities
        self.net = ResNet1D(
            in_channels=args.in_channels,
            base_filters=args.base_filters,
            kernel_size=args.kernel_size,
            stride=args.stride,
            groups=args.groups,
            n_block=args.n_block - 1,
            n_classes=args.n_classes,
        )

        # create 2 branches for the multi-task
        self.project_ssl = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, self.hparams.projection_dim),
        )
        self.project_cls = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, self.hparams.projection_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.projection_dim, output_dim),
        )

        # putting it all together
        models = {m: {} for m in self.modalities}
        for m in self.modalities:
            models[m]["net"] = self.encoder if m == "ECG" else self.net
            models[m]["project_cls"] = self.project_cls
            models[m]["project_ssl"] = self.project_ssl
        self.models = models

    def forward(self, x, y):
        # extract vector representations
        vectors = {m: [] for m in x.keys()}
        for m in x.keys():
            if m == "ECG":
                temp_vec = []
                for i in range(x[m].shape[1]):
                    temp = x[m][:, i, :].unsqueeze(1)
                    temp_vec.append(self.models[m]["net"](temp))
                vectors[m] = torch.mean(torch.stack(temp_vec), dim=0)
            else:
                x[m] = x[m].unsqueeze(1)
                vectors[m] = self.models[m]["net"](x[m])

        # extract cls predictions
        preds = {m: [] for m in x.keys()}
        for m in x.keys():
            preds[m] = self.models[m]["project_cls"](vectors[m])

        # extract ssl representations
        reprs = {m: [] for m in x.keys()}
        for m in x.keys():
            reprs[m] = self.models[m]["project_ssl"](vectors[m])

        # compute the cls objective
        y = torch.mean(y, dim=1).unsqueeze(1).float()
        # ERROR HERE: find good objective for DriveDB
        cls_loss = [self.cls_loss(preds[m], y) for m in x.keys()]
        cls_loss = sum(cls_loss).float() / len(cls_loss)

        # compute the ssl objective
        ssl_keys = [m for m in x.keys() if x != "ECG"]
        ssl_loss = [self.ssl_loss(reprs[m], reprs["ECG"]) for m in ssl_keys]
        ssl_loss = sum(ssl_loss).float() / len(ssl_loss)

        # compute the multi-objective
        return cls_loss + 0.01 * ssl_loss

    def training_step(self, batch, _):
        data, _ = batch
        x = {key: data[key] for key in self.modalities}
        y = data[self.ground_truth]
        loss = self.forward(x, y)
        self.log("Train/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        data, _ = batch
        x = {key: data[key] for key in self.modalities}
        y = data[self.ground_truth]
        loss = self.forward(x, y)
        self.log("Valid/loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
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
