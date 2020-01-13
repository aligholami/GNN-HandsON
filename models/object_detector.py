import torch
import torch.nn as nn
from detectron2.modeling import build_backbone
from detectron2.modeling import build_proposal_generator
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import ImageList
import time

class ObjectDetector(nn.Module):
    def __init__(self, cfg, num_max_regions, device):
        super(ObjectDetector, self).__init__()
        self.device = device
        self.cfg = cfg
        self.backbone = build_backbone(self.cfg)
        self.pooler_resolution = 14
        self.canonical_level = 4
        self.canonical_scale_factor = 2 ** self.canonical_level
        self.pooler_scales = (1 / self.canonical_scale_factor,)
        self.sampling_ratio = 0
        self.proposal_generator = build_proposal_generator(self.cfg, self.backbone.output_shape())
        self.roi_pooler = ROIPooler(
            output_size=self.pooler_resolution,
            scales=self.pooler_scales,
            sampling_ratio=self.sampling_ratio,
            pooler_type="ROIPool"
        )
        self.num_max_regions = num_max_regions

    def forward(self, x):
        """
        Performs the object detection computation.
        :param x: Input image with shape (batch_size, C, H, W)
        :return: A tuple (region_feature_matrix: tensor, batch_indexes: list of tuples). The first element is a tensor
        containing feature map corresponding to each region. The second element is a list of tuples, specifying the
        start and end index of regions associated with each image in the batch.
        """
        cnn_features = self.backbone(x[0])
        cnn_features_p3 = [cnn_features['p3']]
        batch_size = x[0].shape[0]
        W = x[0].shape[1]
        H = x[0].shape[2]
        C = x[0].shape[3]
        image_sizes = [(W, H) for _ in range(batch_size)]
        images = ImageList(x[0], image_sizes)
        proposals, _ = self.proposal_generator(images, cnn_features)
        boxes = [z.proposal_boxes for z in proposals]
        region_feature_matrix = self.roi_pooler(cnn_features_p3, boxes)

        rf_C = region_feature_matrix.shape[1]
        rf_W = region_feature_matrix.shape[2]
        rf_H = region_feature_matrix.shape[3]
        region_feature_matrix = region_feature_matrix.view(-1, rf_C * rf_W * rf_H)

        g_ptr = 0
        batch_indexes = []
        for ix, box in enumerate(boxes):
            batch_indexes.append((g_ptr, g_ptr + len(boxes[ix])))
            g_ptr += len(boxes[ix])

        region_feature_matrix_padded = torch.zeros([batch_size * self.num_max_regions, rf_C * rf_W * rf_H])
        for (ix_s, ix_e) in batch_indexes:
            region_feature_matrix_padded[ix_s:ix_e, :] = region_feature_matrix[ix_s:ix_e, :]

        return region_feature_matrix_padded, batch_indexes
