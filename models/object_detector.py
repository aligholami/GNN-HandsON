import torch
import torch.nn as nn
from detectron2.modeling import build_backbone
from detectron2.modeling import build_proposal_generator
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import ImageList
from torch_geometric.data import Data, Batch


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
        self.proposal_generator = build_proposal_generator(self.cfg, self.backbone.output_shape())
        self.roi_pooler = ROIPooler(
            output_size=self.pooler_resolution,
            scales=self.pooler_scales,
            sampling_ratio=self.sampling_ratio,
            pooler_type="ROIPool"
        )

    def forward(self, x):
        """
        Performs the object detection computation.
        :param x: Input image with shape (batch_size, C, H, W)
        :return: A tuple (region_feature_matrix: tensor, batch_indexes: list of tuples). The first element is a tensor
        containing feature map corresponding to each region. The second element is a list of tuples, specifying the
        start and end index of regions associated with each image in the batch.
        """
        with torch.no_grad():
            cnn_features = self.backbone(x[0])
            cnn_features_p3 = [cnn_features['p3']]
            batch_size = x[0].shape[0]
            W = x[0].shape[1]
            H = x[0].shape[2]
            image_sizes = [(W, H) for _ in range(batch_size)]
            images = ImageList(x[0], image_sizes)
            proposals, _ = self.proposal_generator(images, cnn_features)
            boxes = [z.proposal_boxes for z in proposals]
            region_feature_matrix = self.roi_pooler(cnn_features_p3, boxes)

            rf_C = region_feature_matrix.shape[1]
            rf_W = region_feature_matrix.shape[2]
            rf_H = region_feature_matrix.shape[3]

            g_ptr = 0
            batch = []
            # Create a geometric batch
            for ix, box in enumerate(boxes):
                start_pointer = g_ptr
                end_pointer = start_pointer + len(boxes[ix])
                x = region_feature_matrix[start_pointer:end_pointer, :, :, :]
                x = x.view(-1, rf_C * rf_W * rf_H)
                edge_index = [[0, 1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1, 0]]
                edge_index = torch.tensor(edge_index)
                data = Data(x=x, edge_index=edge_index)
                batch.append(data)

            # Convert to geometric Batch data type
            batch = Batch.from_data_list(batch)

        return batch
