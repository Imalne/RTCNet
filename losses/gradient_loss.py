import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class GradientPriorLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(GradientPriorLoss, self).__init__()
        self.func = nn.L1Loss()
        self.weight = loss_weight

    def forward(self, out_images, target_images):
        map_out = self.gradient_map(out_images)
        map_target = self.gradient_map(target_images)
        return self.func(map_out, map_target) * self.weight

    def gradient_map(self, x):
        batch_size, channel, h_x, w_x = x.size()
        r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
        l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
        t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
        b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
        xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2), 0.5)
        return xgrad
