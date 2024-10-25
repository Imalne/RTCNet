import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from basicsr.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class GramLoss(torch.nn.Module):
    def __init__(self, loss_weight=1):
        super(GramLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, x, y):
        if x.dim() == 4:
            x = x.flatten(2).permute(0,2,1)
            y = y.flatten(2).permute(0,2,1)

        b, hw, c = x.shape

        gmx = x.transpose(1, 2) @ x / (hw)
        gmy = y.transpose(1, 2) @ y / (hw)
    
        return (gmx - gmy).square().mean() * self.loss_weight