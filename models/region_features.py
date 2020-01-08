from detectron2.modeling import build_backbone
from detectron2.modeling.poolers import ROIPooler


class RegionFeatureGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.build_region_feature_generator()

    def build_region_feature_generator(self):
        backbone = build_backbone(self.cfg)
        pooler = ROIPooler(...)

        return lambda nchw_images: pooler(backbone(nchw_images), ...)

