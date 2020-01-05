import torch
import torch.nn as nn
from detectron2.modeling import build_backbone
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes


class ObjectDetector(nn.Module):
    def __init__(self, cfg, device):
        super(ObjectDetector, self).__init__()
        self.device = device
        self.cfg = cfg
        self.backbone = build_backbone(self.cfg)
        self.pooler_resolution = 14
        self.canonical_level = 4
        self.canonical_scale_factor = 2 ** self.canonical_level
        self.pooler_scales = (1 / self.canonical_scale_factor,)
        self.sampling_ratio = 0
        self.number_of_rois = 10
        self.roi_pooler = ROIPooler(
            output_size=self.pooler_resolution,
            scales=self.pooler_scales,
            sampling_ratio=self.sampling_ratio,
            pooler_type="ROIAlignV2"
        )

    def _rand_boxes(self, num_boxes, x_max, y_max):
        coords = torch.rand(num_boxes, 4)
        coords[:, 0] *= x_max
        coords[:, 1] *= y_max
        coords[:, 2] *= x_max
        coords[:, 3] *= y_max
        boxes = torch.zeros(num_boxes, 4)
        boxes[:, 0] = torch.min(coords[:, 0], coords[:, 2])
        boxes[:, 1] = torch.min(coords[:, 1], coords[:, 3])
        boxes[:, 2] = torch.max(coords[:, 0], coords[:, 2])
        boxes[:, 3] = torch.max(coords[:, 1], coords[:, 3])

        return boxes

    def forward(self, x):
        """
        Performs the object detection computation.
        :param x: Input image with shape (batch_size, C, H, W)
        :return: A region-feature matrix, for a given image.
        """
        x = x[0]
        cnn_features = self.backbone(x)['p3']
        batch_size = x.shape[0]
        W = x.shape[1]
        H = x.shape[2]
        C = x.shape[3]

        rois = []
        for _ in range(batch_size):
            boxes = self._rand_boxes(
                num_boxes=self.number_of_rois, x_max=W * self.canonical_scale_factor,
                y_max=H * self.canonical_scale_factor
            )
            rois.append(Boxes(boxes).to(self.device))

        region_feature_matrix = self.roi_pooler([cnn_features], rois)
        region_feature_matrix = region_feature_matrix.view(batch_size, self.number_of_rois, -1)

        return region_feature_matrix
