from torch import nn
import torch
from torchvision.models import *
import math

from block.ChannelShuffle import  *
from block.SpiltAtt import *


class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
        expansion = 4
        width = int(out_channel // expansion)
        self.stride = stride
        if self.stride == 2:
            self.downsample = nn.Conv2d(1024, 2048, 1, 2)
        else:
            self.downsample = None
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out1 = self.conv2(out)
        out1 = self.bn2(out1)
        out1 = self.relu(out1)
        out1 = self.conv3(out1)
        out1 = self.bn3(out1)
        out1 += identity
        out1 = self.relu(out1)
        return out1


class mynodel(nn.Module):
    def __init__(self, class_num=7):
        super(mynodel, self).__init__()
        backbone = resnet50(pretrained=True)
        all_base_modules = [m for m in backbone.children()]
        layers = all_base_modules[:-3]
        self.class_num = class_num
        self.backbone = nn.Sequential(*layers)
        self.block1 = Bottleneck(1024, 2048, 2)
        self.block2 = Bottleneck(2048, 2048)
        self.block3 = Bottleneck(2048, 2048)

        self.p = 0.9
        self.q = 1 / self.p
        self.shuffle = ChannelShuffle(8)
        self.spilt = spilt_att("0011", 0.9)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.soft = nn.Softmax(dim=2)

        self.fc = nn.Linear(2048, self.class_num, bias=True)
        self.softmax = nn.Softmax(dim=1)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.shuffle(x)
        y = self.spilt(x)
        #ShuSpilt loss
        dirloss = torch.squeeze(self.avgpool(x - y)).mean(1).mean(0)
        b, c, h, w = y.shape
        ce = y[:, -int(self.p * c):, :, :]
        # ce=self.sig(ce)
        ce = ce.view(b, -1, h * w)
        ce = self.soft(ce)
        ce, _ = torch.max(ce, 2)
        celoss = ce.mean(1).mean(0)

        y = self.avgpool(y)
        y = torch.squeeze(y)
        y = self.fc(y)
        y = self.softmax(y)
        return y, 20 * dirloss + 0.5 - celoss

def remodel():
    return mynodel()