import torch.nn as nn
from . import RangeEncoder
from .PointDecoder import PointDecoder


class SegFormer(nn.Module):
    def __init__(self, backbone, num_classes=20, embedding_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.backbone = getattr(RangeEncoder, backbone)()

        self.in_channels = self.backbone.embed_dims
        self.decoder = PointDecoder(in_channels=self.in_channels,
                                    embedding_dim=self.embedding_dim, num_classes=self.num_classes)

    def forward(self, x, px, py, points):
        x = self.backbone(x)
        return self.decoder(x, px, py, points)
