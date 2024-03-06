from torch import nn
import torch
from torchvision.models import *
from block.SpiltAttBottleneck import *

class mynodel(nn.Module):
    def __init__(self,class_num=7):
        super(mynodel, self).__init__()
        backbone=resnet50(pretrained=True)
        all_base_modules = [m for m in backbone.children()]
        layers = all_base_modules[:-3]
        self.class_num=class_num
        self.backbone = nn.Sequential(*layers)
        self.block1=Bottleneck(1024,2048,2)
        self.block2=Bottleneck(2048,2048,1)
        self.block3 = Bottleneck(2048, 2048,1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, self.class_num, bias=True)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x=self.backbone(x)
        x,loss1=self.block1(x)
        x,loss2 = self.block2(x)
        x,loss3= self.block3(x)
        x=self.avgpool(x)
        x=torch.squeeze(x)
        x=self.fc(x)
        loss=loss1+loss2+loss3
        return self.softmax(x),loss

def remodel():
    return mynodel()