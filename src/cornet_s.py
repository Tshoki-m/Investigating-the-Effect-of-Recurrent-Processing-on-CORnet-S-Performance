"""CORnet-S with configurable recurrent steps per area.

Re-implemented to mirror the official dicarlolab/cornet `cornet_s.py`
(parameter/module names are kept identical so official pretrained weights load
with strict key matching when times=(2, 4, 2)). Diff this file against the
official one before trusting results -- eval_pretrained_sweep.py checks it end to end.
"""
import math
from collections import OrderedDict

import torch
from torch import nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):
    def forward(self, x):
        return x


class CORblock_S(nn.Module):
    scale = 4  # bottleneck expansion

    def __init__(self, in_channels, out_channels, times=1):
        super().__init__()
        self.times = times
        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale, kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels, kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)
        self.output = Identity()

        # BatchNorm is NOT shared over time steps (paper, Sec. 2.1)
        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)
        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)

            x = x + skip
            x = self.nonlin3(x)
            output = self.output(x)
        return output


def CORnet_S(times=(2, 4, 2), num_classes=1000, v1_pool=True):
    """times = (V2 steps, V4 steps, IT steps). Paper baseline is (2, 4, 2).
    v1_pool=False drops the V1 max-pool (4x less downsampling), used for low-resolution inputs."""
    v2, v4, it = times
    model = nn.Sequential(OrderedDict([
        ('V1', nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('nonlin1', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if v1_pool else Identity()),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('nonlin2', nn.ReLU(inplace=True)),
            ('output', Identity()),
        ]))),
        ('V2', CORblock_S(64, 128, times=v2)),
        ('V4', CORblock_S(128, 256, times=v4)),
        ('IT', CORblock_S(256, 512, times=it)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, num_classes)),
            ('output', Identity()),
        ]))),
    ]))

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    for cfg in [(1, 1, 1), (2, 2, 2), (2, 4, 2), (4, 4, 4)]:
        m = CORnet_S(cfg)
        y = m(torch.randn(2, 3, 224, 224))
        print(cfg, tuple(y.shape), f'{count_params(m)/1e6:.2f}M params')
