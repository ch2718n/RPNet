import torch.nn as nn
import torch
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


def resample_grid(predictions, py, px):
    pypx = torch.stack([px, py], dim=3)
    resampled = F.grid_sample(predictions, pypx, align_corners=False)

    return resampled


class PointDecoder(nn.Module):

    def __init__(self, in_channels=128, embedding_dim=256, num_classes=20):
        super(PointDecoder, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.aux_head4 = nn.Conv2d(embedding_dim, num_classes, kernel_size=1, bias=False)
        self.aux_head3 = nn.Conv2d(embedding_dim, num_classes, kernel_size=1, bias=False)
        self.aux_head2 = nn.Conv2d(embedding_dim, num_classes, kernel_size=1, bias=False)

        self.point_mlp = nn.Sequential(
            nn.Linear(5, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.final = nn.Sequential(
            nn.Conv2d(embedding_dim * 5, embedding_dim, kernel_size=1),
            nn.BatchNorm2d(embedding_dim),
            nn.LeakyReLU(),
            nn.Conv2d(embedding_dim, num_classes, kernel_size=1, bias=False)
        )

    def forward(self, x, px, py, points):
        c1, c2, c3, c4 = x
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resample_grid(_c4, py, px)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resample_grid(_c3, py, px)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resample_grid(_c2, py, px)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c1 = resample_grid(_c1, py, px)

        points = self.point_mlp(points).transpose(1, 2).unsqueeze(2)
        _c = self.final(torch.cat([_c4, _c3, _c2, _c1, points], dim=1))

        if self.training:
            aux4 = self.aux_head4(_c4)
            aux3 = self.aux_head3(_c3)
            aux2 = self.aux_head2(_c2)
            return _c, aux2, aux3, aux4
        else:
            return _c
