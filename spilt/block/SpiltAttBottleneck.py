from torch import nn
import torch
import math
from block.SpiltAtt import *


class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, stride, groups=1):
        super(Bottleneck, self).__init__()
        expansion = 4
        width = int(out_channel / expansion)
        # CA+spiltatt
        self.stride = stride
        if self.stride == 2:
            self.downsample = nn.Conv2d(1024, 2048, 1, 2)
        else:
            self.downsample = None
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=self.stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.soft = nn.Softmax(dim=2)
        self.p = 0.9
        self.w = 0
        self.q = 1 / ((1 - self.p) * self.w)
        self.spilt = spilt_att("0011", self.p, self.w)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """
        out1 = self.spilt(x)
        dirloss = self.avgpool(x - out1)
        dirloss = torch.abs(torch.squeeze(dirloss)).mean(1).mean(0)
        """
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = out + identity
        out = self.relu(out)
        #SpiltAtt loss
        out1 = self.spilt(out)
        dirloss = self.avgpool(out - out1)
        dirloss = torch.abs(torch.squeeze(dirloss)).mean(1).mean(0)

        b, c, h, w = y.shape
        ce = out[:, -int(self.p * c):, :, :]
        ce = ce.view(b, -1, h * w)
        ce = self.soft(ce)
        ce, _ = torch.max(ce, 2)
        celoss = ce.mean(1).mean(0)
        return out, self.q * dirloss+(1-celoss)/3  # or /6