import torch
import torch.nn as nn

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups
    def forward(self, x):
        b, c, h, w = x.size()
        #how many channels are there in  each group
        channels_per_group = c // self.groups

        # reshape
        x = x.view(b, self.groups, channels_per_group, h, w)

        x = torch.transpose(x, 1, 2).contiguous()
        out = x.view(b, -1, h, w)

        return out