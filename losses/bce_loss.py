import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class BCEWithLogitsLoss(nn.Module):
    def __init__(self, loss_weight=1.0, clip_rate=0):
        super(BCEWithLogitsLoss, self).__init__()
        self.func = nn.BCEWithLogitsLoss(reduction='none')
        self.weight = loss_weight
        self.clip_rate=clip_rate

    def forward(self, pred, target):
        bce_loss = torch.mean(self.func(pred, target),dim=-1)
        if self.clip_rate == 0:
            return torch.mean(bce_loss)
    
        remove_num = int(self.clip_rate * bce_loss.shape[0])
        indices_remove = bce_loss.topk(remove_num, largest=True)[1]
        org_mean = torch.mean(bce_loss).detach()
        bce_loss[indices_remove] = 0
        bce_loss = torch.mean(bce_loss)
        return bce_loss * org_mean / bce_loss.detach()

@LOSS_REGISTRY.register()
class KLLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(KLLoss, self).__init__()
        self.weight = loss_weight

    def forward(self, pred, target):
        B,C,H,W = pred.shape
        pred = pred.permute(0,2,3,1)
        target = target.permute(0,2,3,1)
        kl = F.kl_div(pred.log(), target, reduction='sum')
        return kl * self.weight

