import torch, torch.nn as nn
from pytorch_lightning import LightningModule
from src.models import ResNet1D, S4Model

import torch, torch.nn as nn
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.utils import CCCLoss, mean_ccc


class SupervisedLearning(LightningModule):
    def __init__(self, args, encoder):
        super().__init__()
        self.save_hyperparameters(args)
        self.ground_truth = args.gtruth
        self.accuracy = accuracy_score
        self.bs = args.batch_size
        self.modalities = args.streams
        self.args = args

        # sanity checks
        if self.args.type == "regression":
            assert self.args.output_dim == 1 or "ptb" in args.dataset

        # freezing pre-trained ECG encoder
        self.ecg_encoder = encoder
        self.ecg_encoder.eval()
        for param in self.ecg_encoder.parameters():
            param.requires_grad = self.args.unfreeze

        # create model for other modalities
        if args.model_type == "resnet":
            self.net = ResNet1D(
                in_channels=args.in_channels,
                base_filters=args.base_filters,
                kernel_size=args.kernel_size,
                stride=args.stride,
                groups=args.groups,
                n_block=args.n_block - 1,
                n_classes=args.n_classes,
            )
        elif args.model_type == "s4":
            self.net = S4Model(
                d_input=args.d_input,
                d_output=args.d_output,
                d_model=args.d_model,
                n_layers=args.n_layers - 2,
                dropout=args.dropout,
                prenorm=True,
            )
        else:
            raise ValueError("Model type not supported.")

        # model initialization
        self.models = {}
        for m in self.modalities:
            if m == "ecg":
                self.models[m] = self.ecg_encoder
            elif m != "skt":
                self.models[m] = self.net

        # attention dimension
        if "skt" in self.modalities:
            num = len(self.modalities) - 1
            self.att_dim = int(num * self.hparams.projection_dim + 4)
        else:
            num = len(self.modalities)
            self.att_dim = int(num * self.hparams.projection_dim)

        # attention layer
        self.query = nn.Linear(self.att_dim, self.att_dim)
        self.key = nn.Linear(self.att_dim, self.att_dim)
        self.value = nn.Linear(self.att_dim, self.att_dim)

        # task projector
        self.projector = nn.Linear(self.att_dim, self.args.output_dim)

        # helper modules
        self.validation_true = list()
        self.validation_pred = list()

    def forward(self, x, y, flag=False):
        # extract embeddings
        all_vectors = []
        for m in self.modalities:
            if m != "skt":
                if self.train:
                    x[m] = x[m].unsqueeze(1)
                all_vectors.append(self.models[m](x[m]))
        if "skt" in self.modalities:
            all_vectors.append(x["skt"])

        # multimodal fusion
        fused_vector = torch.cat(all_vectors, dim=-1)
        fused_vector = fused_vector.half() if flag else fused_vector.float()
        fused_vector = fused_vector / fused_vector.norm(dim=1, keepdim=True)

        # attention mechanism
        q = self.query(fused_vector)
        k = self.key(fused_vector)
        v = self.value(fused_vector)

        weights = torch.matmul(q, k.T)
        weights = (weights / q.shape[-1] ** 0.5).softmax(0)
        fused_vector = torch.matmul(weights, v)

        # return predictions
        preds = self.projector(fused_vector).squeeze()
        y = y.float() if self.args.type == "regression" else y.long()
        return preds, y

    def training_step(self, batch, _):
        data, y, _ = batch
        x = {key: data[key] for key in self.modalities}
        preds, y = self.forward(x, y, True)
        loss = self.compute_loss(preds, y)
        self.log("Train/loss", loss, sync_dist=True, batch_size=self.bs)
        return loss

    def validation_epoch_end(self, _):
        if self.args.type == "regression":
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
        x = {key: data[key] for key in self.modalities}
        preds, y = self.forward(x, y, True)
        loss = self.compute_loss(preds, y, val=True)

        if "ptb" in self.args.dataset:
            auroc = roc_auc_score(y.cpu(), preds.cpu())
            self.log("Valid/auroc", auroc, sync_dist=True, batch_size=self.bs)

            preds = (torch.sigmoid(preds).detach() > 0.5) * 1
            f1 = f1_score(y.cpu(), preds.cpu(), average="macro", zero_division=0)
            self.log("Valid/f1", f1, sync_dist=True, batch_size=self.bs)

        elif self.args.type == "regression":
            for idx in range(len(preds.cpu())):
                self.validation_pred.append(preds.cpu()[idx])
                self.validation_true.append(y.cpu()[idx])

        else:
            preds = preds.argmax(dim=1)
            f1 = f1_score(y.cpu(), preds.cpu(), average="macro", zero_division=0)
            self.log("Valid/f1", f1, sync_dist=True, batch_size=self.bs)

            acc = accuracy_score(y.cpu(), preds.cpu())
            self.log("Valid/acc", acc, sync_dist=True, batch_size=self.bs)

        self.log("Valid/loss", loss, sync_dist=True, batch_size=self.bs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=float(self.hparams.weight_decay),
        )
        return {"optimizer": optimizer}

    def compute_loss(self, preds, y, val=False):
        if "ptb" in self.args.dataset:
            loss = nn.BCEWithLogitsLoss()
        elif self.args.type == "regression":
            loss = CCCLoss()
        else:
            weight = None if val else len(y) / torch.bincount(y)
            loss = nn.CrossEntropyLoss(weight=weight)
        return loss(preds, y)
