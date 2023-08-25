import lightning as l
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import YOLODataset
import config

class PascalVOCDataModule(l.LightningDataModule):
    def __init__(self, train_csv_path, test_csv_path) -> None:
        super().__init__()
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        
    def setup(self, stage: str) -> None:
        # Use Multiple GPU if available
        IMAGE_SIZE = config.IMAGE_SIZE
        train_dataset = YOLODataset(
            self.train_csv_path,
            transform=config.train_transforms,
            S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
            img_dir=config.IMG_DIR,
            label_dir=config.LABEL_DIR,
            anchors=config.ANCHORS,
        )
        val_dataset = YOLODataset(
            self.test_csv_path,
            transform=config.test_transforms,
            S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
            img_dir=config.IMG_DIR,
            label_dir=config.LABEL_DIR,
            anchors=config.ANCHORS,
        )
        if stage=="train":
            return train_dataset
        elif stage=="val":
            return val_dataset
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataset = self.setup(stage="train")
        return DataLoader(
            dataset=train_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=True,
            drop_last=False,
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_dataset = self.setup(stage="val")
        return DataLoader(
            dataset=val_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=False,
            drop_last=False,
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass