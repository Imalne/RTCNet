import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from basicsr.utils.registry import LOSS_REGISTRY



@LOSS_REGISTRY.register()
class UnwindL2Loss(torch.nn.Module):
    def __init__(self, in_channels, out_channels, loss_weight=1):
        super(UnwindL2Loss, self).__init__()
        self.loss_weight = loss_weight
        self.conv_tran_1 = torch.nn.Conv2d(in_channels,out_channels, kernel_size=1)
        self.conv_tran_2 = torch.nn.Conv2d(in_channels,out_channels, kernel_size=1)

    def forward(self, pred, tar):
        return F.mse_loss(self.conv_tran_1(pred), self.conv_tran_2(tar))*self.loss_weight
        