import os
import glob
import yaml

import torch
import boto3
import lightning as L
from torch.nn import functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch as pl


class MNISTModel(L.LightningModule):
    def __init__(
        self,
        lr=0.02,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, dirpath, **kwargs):
        super().__init__(*args, dirpath=dirpath, **kwargs)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        super().on_train_start(trainer, pl_module)
        with open(os.path.join(self.dirpath, 'hyperparams.yaml'), 'w') as f:
            yaml.dump(trainer.model.hparams, f)
