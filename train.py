import glob
import os
from typing import Optional

import torch
import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import boto3

from model import MNISTModel, MyModelCheckpoint
from s3_utils import upload_files_to_s3, download_files_from_s3

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64

os.environ["AWS_ACCESS_KEY_ID"] = "developer"
os.environ["AWS_SECRET_ACCESS_KEY"] = "developer_password"


def train(s3client, n_epochs: int, bucket: str, s3_checkpoint: Optional[str] = None):
    mnist_model = MNISTModel()

    checkpoint_callback = MyModelCheckpoint(
        dirpath='callbacks',
        save_last=True,
    )

    train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=1)
    trainer = L.Trainer(
        logger=False,
        callbacks=checkpoint_callback,
        accelerator="auto",
        devices=1,
        max_epochs=n_epochs,
    )

    if s3_checkpoint is not None:
        from_checkpoint = download_files_from_s3(s3client, bucket=bucket, s3_src_file=s3_checkpoint)
    else:
        from_checkpoint = None

    trainer.fit(mnist_model, train_loader, ckpt_path=from_checkpoint)
    upload_files_to_s3(
        s3client,
        src_dir='callbacks',
        s3_trg_dir='callbacks',
        bucket=bucket,
    )


if __name__ == '__main__':
    s3client = boto3.resource('s3',
                              endpoint_url='http://localhost:9000',
                              aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                              aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                              )
    s3_checkpoint = 'callbacks/last.ckpt'
    n_epochs = 5
    bucket = 'temp'
    train(s3client, n_epochs=n_epochs, bucket=bucket, s3_checkpoint=s3_checkpoint)
