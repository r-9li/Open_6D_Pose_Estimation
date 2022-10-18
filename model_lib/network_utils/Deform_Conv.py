import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class DeformConv_2d(nn.Module):
    def __init__(self, in_channlels, out_channels, kernel_size, stride, padding):
        super(DeformConv_2d, self).__init__()

        self.offset_conv = nn.Conv2d(in_channlels, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                                     stride=stride, padding=padding)
        self.mask_conv = nn.Conv2d(in_channlels, 1 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride,
                                   padding=padding)

        self.conv = torchvision.ops.DeformConv2d(in_channlels, out_channels, kernel_size, stride=stride,
                                                 padding=padding)

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = F.sigmoid(self.mask_conv(x))
        out = self.conv(x, offset, mask)
        return out
