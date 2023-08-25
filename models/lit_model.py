from typing import Any, Optional
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn.functional as F
from watermark import watermark
from models import model
import config
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from loss import YoloLoss
from lit_dataset import PascalVOCDataModule
from utils import get_loaders

class LightningYolo3(L.LightningModule):
    def __init__(self, model, learning_rate, weight_decay, num_steps, num_epochs, max_lr) -> None:
        super(LightningYolo3, self).__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = model.to(config.DEVICE)
        self.num_steps= num_steps
        self.num_epochs = num_epochs
        self.max_lr = max_lr
        self.criteria = YoloLoss()
        
        self.scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        ).to(config.DEVICE)

        self.datamodule = PascalVOCDataModule(
            train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
        )

        # self.train_loader, self.test_loader, self.train_eval_loader = get_loaders(
        #     train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
        # )

    def forward(self, x):
        return self.model(x)
    
    def shared_step(self, batch, batch_idx):
        features, true_labels = batch
        #y0, y1, y2 = (true_labels[0], true_labels[1], true_labels[2])
        
        output = self.forward(features)

        # if (
        #     true_labels[0].isnan().any()
        #     or true_labels[1].isnan().any()
        #     or true_labels[2].isnan().any()
        # ):
        #     print(true_labels[0].dtype)
        #     print(true_labels)
        #     raise Exception("NAN Detected in target")

        # if features.isnan().any():
        #     print(features.dtype)
        #     print(features)
        #     raise Exception("NAN Detected")

        # #print(output)

        # if (
        #     output[0].isnan().any()
        #     or output[1].isnan().any()
        #     or output[2].isnan().any()
        # ):
        #     print(output.dtype)
        #     print(output)
        #     raise Exception("NAN Detected")

        # loss = (
        #         self.loss_fn(output[0], y0, self.scaled_anchors[0])
        #         + self.loss_fn(output[1], y1, self.scaled_anchors[1])
        #         + self.loss_fn(output[2], y2, self.scaled_anchors[2])
        #     )
        #loss = self.loss_fn.forward(output, true_labels, self.scaled_anchors)

        # if loss.isnan().any():
        #     print(true_labels)
        #     print(output)
        #     raise Exception("NAN Detected in Loss")

        loss = self.calculate_loss(output, true_labels)
        
        return output, loss, true_labels

    def training_step(self, batch, batch_idx):
        output, loss, true_labels = self.shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss # This is pass to the optimizer for training
    
    def on_train_epoch_start(self) -> None:
        plot_couple_examples(self, self.datamodule.val_dataloader(), 0.6, 0.5, self.scaled_anchors)
        #plot_couple_examples(self, self.test_loader, 0.6, 0.5, self.scaled_anchors)

    def on_train_epoch_end(self) -> None:
        print(f"Currently epoch {self.current_epoch}")
        if self.current_epoch <= 2:
            print("Checking Accuracy:")
            check_class_accuracy(self, self.datamodule.train_dataloader(), threshold=config.CONF_THRESHOLD)
        elif self.current_epoch > 2 and self.current_epoch%5==0:
            print("Checking Accuracy:")
            check_class_accuracy(self, self.datamodule.train_dataloader(), threshold=config.CONF_THRESHOLD)
            #check_class_accuracy(self, self.train_loader, threshold=config.CONF_THRESHOLD)
    
    def validation_step(self, batch, batch_idx):
        output, loss, true_labels = self.shared_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        if self.current_epoch > 0 and self.current_epoch %  5 == 0:
            check_class_accuracy(self, self.datamodule.val_dataloader(), threshold=config.CONF_THRESHOLD)
            #check_class_accuracy(self, self.test_loader, threshold=config.CONF_THRESHOLD)
            
        if self.current_epoch == self.num_epochs-1:
            pred_boxes, true_boxes = get_evaluation_bboxes(
                self.datamodule.val_dataloader(),
                #self.test_loader,
                self,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        return self.validation_step(batch, batch_idx)
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # OneCycleLR scheduler 
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr = self.max_lr, steps_per_epoch = self.num_steps, epochs = self.num_epochs, pct_start = 5/self.num_epochs,
            div_factor = 100, three_phase = False, final_div_factor = 100, anneal_strategy = "linear"
        )

        return {
            "optimizer":optimizer,
            "lr_scheduler":{
                "scheduler":scheduler,
                "monitor":"train_loss",
                "interval":"step", # Default = epoch| While using cosine annealing scheduler, epoch means restart LR at every epoch, step means at every batch.
                "frequency":1,   # Default
            },
        }
    
    def calculate_loss(self, output, target):
        # scaled_anchors = torch.tensor(
        #     config.ANCHORS, device=config.DEVICE
        # ) * torch.tensor(config.S, device=config.DEVICE).unsqueeze(1).unsqueeze(1).repeat(
        #     1, 3, 2
        # )

        loss = (
            self.criteria(output[0].clone(), target[0].clone(), self.scaled_anchors[0])
            + self.criteria(output[1].clone(), target[1].clone(), self.scaled_anchors[1])
            + self.criteria(output[2].clone(), target[2].clone(), self.scaled_anchors[2])
        )
        #loss = self.criteria(output, target, scaled_anchors)

        return loss
