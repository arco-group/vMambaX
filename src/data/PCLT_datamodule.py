import argparse
from typing import Optional

import pytorch_lightning as pl
from easydict import EasyDict as edict

from torch.utils.data import DataLoader
from src.data.PCLT_dataset import prepare_PETCT_dataset


class PETCTDataModule(pl.LightningDataModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.workers
        self.pin_memory = args.device.startswith("cuda")

        self.train_dataset = None
        self.val_dataset = None
        self.train_steps_per_epoch: int = 0

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return
        train_dataset, val_dataset = prepare_PETCT_dataset(
            self.args, transforms=True)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        if len(self.train_dataset) == 0 or self.batch_size <= 0:
            self.train_steps_per_epoch = 0
        else:
            self.train_steps_per_epoch = max(
                len(self.train_dataset) // self.batch_size, 1)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self):
        num_workers = min(max(self.num_workers, 0), 8)
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=self.val_dataset.collate_fn,
        )

    def test_dataloader(self):
        return self.val_dataloader()
