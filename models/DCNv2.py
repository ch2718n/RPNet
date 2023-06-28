import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# --------------------------DCNv2 start--------------------------
class DCNv2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.kernel_size = to_2tuple(kernel_size)

        # init weight and bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        # offset conv
        self.conv_offset_mask = nn.Conv2d(in_channels,
                                          3 * self.kernel_size[0] * self.kernel_size[1],
                                          kernel_size=self.kernel_size,
                                          stride=stride,
                                          padding=self.padding,
                                          bias=True)

        # init
        self.reset_parameters()
        self._init_weight()

    def reset_parameters(self):
        n = self.in_channels * (self.kernel_size[0] * self.kernel_size[1])
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def _init_weight(self):
        # init offset_mask conv
        nn.init.constant_(self.conv_offset_mask.weight, 0.)
        nn.init.constant_(self.conv_offset_mask.bias, 0.)

    def forward(self, x):
        out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = F.softmax(mask, dim=1)
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.weight,
                                          bias=self.bias,
                                          padding=self.padding,
                                          mask=mask,
                                          stride=self.stride)
        return x
# ---------------------------DCNv2 end---------------------------
