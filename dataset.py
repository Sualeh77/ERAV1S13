"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

import config
import numpy as np
import os
import pandas as pd
import torch
from utils import xywhn2xyxy, xyxy2xywhn
import random 

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=20,
        transform=None,
        is_train=False,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.mosaic_border = [image_size // 2, image_size // 2] #[208, 208] for 416 img size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales | so it gives total 9 anchors, 3 anchors per scale
        self.num_anchors = self.anchors.shape[0]    # 9
        self.num_anchors_per_scale = self.num_anchors // 3  # 3
        self.C = C
        self.ignore_iou_thresh = 0.5
        self.is_train = is_train

    def __len__(self):
        return len(self.annotations)
    
    def load_mosaic(self, index):
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4 = []
        s = self.image_size # 416
        yc, xc = (int(random.uniform(x, 2 * s - x)) for x in self.mosaic_border)  # mosaic center x, y (it returns random num between x=208 and 2 * x - x = 624 in this example)
        indices = [index] + random.choices(range(len(self)), k=3)  # 3 additional image indices (Makes list of image at index and 3 random images from dataset. Which makes list of 4 images)
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # i is index of indices list and index is the index of image in dataset
            # Load image
            label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])  # Get the label path from annotations df for the image we are looping for
            # og sequence of data in file is [c, xc, yc, w, h] post np.roll > [xc, yc, w, h, c]
            bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
            img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
            img = np.array(Image.open(img_path).convert("RGB")) # loads image and converts in (h, w, ch)
            

            h, w = img.shape[0], img.shape[1]
            labels = np.array(bboxes)

            # place img in img4 (mosaic grid of 4 images)
            if i == 0:  # top left
                # It creates an image with all pixel as default 114. shape of image will be 832x832,3
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            if labels.size:
                labels[:, :-1] = xywhn2xyxy(labels[:, :-1], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            labels4.append(labels)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, :-1],):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate
        labels4[:, :-1] = xyxy2xywhn(labels4[:, :-1], 2 * s, 2 * s)
        labels4[:, :-1] = np.clip(labels4[:, :-1], 0, 1)
        labels4 = labels4[labels4[:, 2] > 0]
        labels4 = labels4[labels4[:, 3] > 0]
        return img4, labels4 

    def __getitem__(self, index):
        if self.is_train and random.randint(1, 100) > 25:
            image, bboxes = self.load_mosaic(index)
        else:
            # def __getitem__(self, index):
            label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
            bboxes = np.roll(
                np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1
            ).tolist()
            img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
            image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S] # for each scale target=[anchor, s, s, objectness, xc, yc, w, h, c]
        for box in bboxes: # Iterating over boxes in an image | As 1 image can have multiple objects
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors) # It checks IOU of the box (using w, h hence [2:4]) with every 9 (3 per scale) anchors
            anchor_indices = iou_anchors.argsort(descending=True, dim=0) # It sorts index of the iou_anchors in descending order. best being first...so on
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                """ scale_idx : anchor idices are 0 to 8, best iou at 0 and worst at 8. we int divide it by num of anchors per scale 
                3 in our case. so if anchor idx=8 we wil get 8//3=2. this will give us the scale (13 or 26 or 52) which suits best
                for prediction. Basically this choses a scale"""
                scale_idx = anchor_idx // self.num_anchors_per_scale
                """
                anchor_on_scale : This choses which anchor shud be use.
                we have 0 to 8 anchors ordered in descending on the basis of it's iou score. anchors are saved as follows 
                [s1a1, s1a2, s1a3, s2a1, ...., s3a3]. So if anchor_idx= 0 or 3 or 6 a1 will be picked because 0|3|6 % 3 = 0, 
                if anchor_idx= 1|4|7 then a2 will be picked. and if anchor_idx= 2|5|8 then a3 will be picked
                """
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx] # We chose scale base on scale_idx found above
                i, j = int(S * y), int(S * x)  # This gives in which cell object center is present. | EG x=0.5 , y=0.5, s=13 -> i=13*0.5=6.5=6
                """
                    Let say scale_idx=0, so we pick taget values of s1 which is [anchor, s, s, objectness, xc, yc, w, h, c]
                    Then from s1 target values we pick value of objectness in anchor_taken (a1 | a2 | a3).
                    Basically to check if the anchor is already taken for other object or not which is super rare.
                """
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0] 
                if not anchor_taken and not has_anchor[scale_idx]:
                    # We get in this loop when has_anchor[scale_idx] is False, and anchor_taken(Means objectness) is 0
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1 # Here we set objectness = 1 for the selected target , scale and anchor
                    """
                        x_cell, y_cell are the position inside the selected cell where centroid of the object is present 
                    """
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates # Setting up target box value for selected scale and anchor (1:5 is for xc, yc, w,h)
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label) # Set up class label
                    has_anchor[scale_idx] = True # Here we update that scale at scale_idx now has anchor

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    """
                        Here we ignore the prediction for anchor which has higher iou than threshold
                    """
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        if (
            image.isnan().any()
            or targets[0].isnan().any()
            or targets[1].isnan().any()
            or targets[2].isnan().any()
        ):
            raise Exception("Nan Value Detected")
        
        """
            Eventually I will get image and target for the image. A target of an image includes 3 values.
            1 is for scale 13 other for scale 26 and last one for scale 52. in Each scale we will have 
            [anchor, S, S, objectness, xc, yc, w, h, c] : 
            Where :
              anchor is 1 slected anchor which has heighest iou with object box
              S is scale 13|26|52
              objectness it will be 1 only for the cell where object centroid is present
              xc, yc, w, h, c : these values will be 0 for all the cell except for the cell where object centroid is present

              so we have 3 target per object 
              [13x13, 26x26, 52x52] in each of the scale all cell values are 0s except ith,jth cell where centroid of object is present
        """
        return image, tuple(targets)


def test():
    anchors = config.ANCHORS

    transform = config.test_transforms

    dataset = YOLODataset(
        "PASCAL_VOC/train.csv",
        "PASCAL_VOC/images/",
        "PASCAL_VOC/labels/",
        S=[13, 26, 52],
        anchors=anchors,
        transform=transform,
    )
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            print(anchor.shape)
            print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        print(boxes)
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)
        #break      #Keep at the time of testing or else it will plot all the images


if __name__ == "__main__":
    test()