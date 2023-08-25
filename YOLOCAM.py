from typing import List
import torch
import numpy as np
import cv2
import random
from utils import non_max_suppression, cells_to_bboxes

from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class YoloCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(YoloCAM, self).__init__(model,
                                       target_layers,
                                       use_cuda,
                                       reshape_transform,
                                       uses_gradients=False)

    def forward(self,
                input_tensor: torch.Tensor,
                scaled_anchors: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:

        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)
        if targets is None:
            bboxes = [[] for _ in range(1)]
            for i in range(3):
                batch_size, A, S, _, _ = outputs[i].shape
                anchor = scaled_anchors[i]
                boxes_scale_i = cells_to_bboxes(
                    outputs[i], anchor, S=S, is_preds=True
                )
                for idx, (box) in enumerate(boxes_scale_i):
                    bboxes[idx] += box
            
            nms_boxes = non_max_suppression(
                bboxes[0], iou_threshold=0.5, threshold=0.5, box_format="midpoint",
            )
            # target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            target_categories = [box[0] for box in nms_boxes]
            targets = [ClassifierOutputTarget(
                category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output)
                       for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)
    
    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_2d_projection(activations)