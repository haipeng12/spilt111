import torch
import torch.nn as nn


class spilt_att(nn.Module):
    def __init__(self, mode, p=0.5,w=0.1):
        super(spilt_att, self).__init__()
        """
        不需要中心点
        """
        self.p = p  # p为通道保留的比率
        self.mode = mode
        self.w=w

    def forward(self, x):
        b, c, h, w = x.shape
        if self.mode == "0110":
            cc = int((c * (1 - self.p)) // 2)
            spilt = torch.ones(b, c // 2, h, w).cuda()
            spilt[:, :cc, :, :] = 0
            spilt1 = torch.zeros(b, c // 2, h, w).cuda()
            spilt1[:, :c // 2 - cc, :, :] = 1
            spilt = torch.cat((spilt, spilt1), dim=1)
            spilt = x.mul(spilt)
            return spilt
        elif self.mode == "1100":
            cc = int(c * self.p)
            spilt = torch.zeros(b, c, h, w).cuda()
            spilt[:, :cc, :, :] = 1
            spilt = x.mul(spilt)
            return spilt
        elif self.mode == "0011":
            cc = int(c * self.p)
            spilt = torch.zeros(b, c, h, w).cuda()
            spilt[:, -cc:, :, :] = 1
            spilt[:,:c-cc,:,:]=self.w
            spilt = x.mul(spilt)
            return spilt
        elif self.mode == "1010":
            cc = int((c * self.p) // 2)
            spilt = torch.zeros(b, c, h, w).cuda()
            spilt[:, :cc, :, :] = 1
            spilt[:, c // 2:c // 2 + cc, :, :] = 1
            spilt = x.mul(spilt)
            return spilt
        elif self.mode == "0101":
            cc = int((c * (1 - self.p)) // 2)
            spilt = torch.ones(b, c, h, w).cuda()
            spilt[:, :cc, :, :] = 0
            spilt[:, c // 2:c // 2 + cc, :, :] = 0
            spilt = x.mul(spilt)
            return spilt
        elif self.mode == "1001":
            cc = int((c * self.p) // 2)
            spilt = torch.zeros(b, c, h, w).cuda()
            spilt[:, :cc, :, :] = 1
            spilt[:, -cc:, :, :] = 1
            spilt = x.mul(spilt)
            return spilt
        else:
            return None