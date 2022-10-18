import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels):
        super(SEBlock, self).__init__()
        mid_channels = channels // 8
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=mid_channels, kernel_size=1, stride=1, groups=1,
                               bias=True)
        self.activ = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=mid_channels, out_channels=channels, kernel_size=1, stride=1, groups=1,
                               bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)

        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)

        w = self.sigmoid(w)
        x = x * w
        return x
