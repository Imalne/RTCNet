import torch
from torch import nn, Tensor
from torch.nn import functional as F
from archs.network_swinir import RSTB
from typing import Optional, List
import math, numpy as np

from functools import partial
from timm.models.vision_transformer import PatchEmbed, Block


class ResBlock(nn.Module):
    def __init__(self, channel, in_norm=False):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channel, eps=1e-6, affine=True),
            nn.SiLU(True),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=channel, eps=1e-6, affine=True),
            nn.SiLU(True),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
        )

    def forward(self, input):
        res = self.conv(input)
        out = res + input
        return out


class SwinLayers(nn.Module):
    def __init__(self, input_resolution=(32, 32), embed_dim=256,
                 blk_depth=6,
                 num_heads=8,
                 window_size=8,
                 num_rstb=4,
                 **kwargs):
        super().__init__()
        self.swin_blks = nn.ModuleList()
        for i in range(num_rstb):
            layer = RSTB(embed_dim, input_resolution, blk_depth, num_heads, window_size, patch_size=1, **kwargs)
            self.swin_blks.append(layer)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).transpose(1, 2)
        for m in self.swin_blks:
            x = m(x, (h, w))
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x


class Quantize2(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, code_book_opt=False):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        self.codebook = nn.Embedding(dim, n_embed)
        self.codebook.weight.data.uniform_(-1.0/self.n_embed, -1.0/self.n_embed)
        self.embed = self.codebook.weight
        self.code_book_opt = code_book_opt


    def forward(self, input_lr, gt_indice=None, rtn_embed_sort=False, prior=None, prior_weight=0.1, prior_trans=nn.Identity()):
        flatten_lr = input_lr.reshape(-1, self.dim)

        
        # upgrade embed
        dist_lc_lr = (
                flatten_lr.pow(2).sum(1, keepdim=True)
                - 2 * flatten_lr @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind_lc_lr = (-dist_lc_lr).max(1)
        embed_onehot_lc_lr = F.one_hot(embed_ind_lc_lr, self.n_embed).type(flatten_lr.dtype)
        embed_ind_lc_lr = embed_ind_lc_lr.view(*input_lr.shape[:-1])
        quantize_lc_lr = self.embed_code(embed_ind_lc_lr)

        if self.training and self.code_book_opt:
            diff_lc_lr = (quantize_lc_lr.detach() - input_lr).pow(2).mean()*0.25 + (quantize_lc_lr - input_lr.detach()).pow(2).mean()
            if prior is not None:
                diff_lc_lr += (prior_trans(quantize_lc_lr.permute(0, 3, 1, 2)) - prior).pow(2).mean() * prior_weight
        else:
            diff_lc_lr = (quantize_lc_lr.detach() - input_lr).pow(2).mean()

        quantize_lc_lr = input_lr + (quantize_lc_lr - input_lr).detach()

        if not rtn_embed_sort:
            return quantize_lc_lr, diff_lc_lr, embed_ind_lc_lr, dist_lc_lr
        else:
            return quantize_lc_lr, diff_lc_lr,  torch.flip((-dist_lc_lr).sort(1)[1], [1]), dist_lc_lr

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

    def embed_logits(self, embed_logits):
        return torch.matmul(embed_logits, self.embed.T)

class Quantize3(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, code_book_opt=False):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        self.codebook = nn.Embedding(dim, n_embed)
        self.codebook.weight.data.uniform_(-1.0/self.n_embed, -1.0/self.n_embed)
        self.embed = self.codebook.weight
        self.code_book_opt = code_book_opt


    def forward(self, input_lr, gt_indice=None, rtn_embed_sort=False, prior=None, prior_weight=0.1, prior_trans=nn.Identity()):
        flatten_lr = input_lr.reshape(-1, self.dim)

        
        # upgrade embed
        dist_lc_lr = (
                flatten_lr.pow(2).sum(1, keepdim=True)
                - 2 * flatten_lr @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        p = (1/dist_lc_lr)
        selected_indices = torch.argsort(p,dim=-1,descending=True)[:,:64]
        p = torch.gather(p,1,selected_indices)
        p = p / torch.sum(p,dim=-1, keepdim=True)
        # embed_ind_lc_lr = torch.tensor([np.random.choice(selected_indices[i].cpu().numpy(), p=p[i].detach().cpu().numpy()) for i in range(len(p))]).long().to(input_lr.device)
        embed_ind_lc_lr= torch.gather(selected_indices, 1, torch.multinomial(p, 1)).squeeze()
        embed_onehot_lc_lr = F.one_hot(embed_ind_lc_lr, self.n_embed).type(flatten_lr.dtype)
        embed_ind_lc_lr = embed_ind_lc_lr.view(*input_lr.shape[:-1])
        quantize_lc_lr = self.embed_code(embed_ind_lc_lr)

        if self.training and self.code_book_opt:
            diff_lc_lr = (quantize_lc_lr.detach() - input_lr).pow(2).mean()*0.25 + (quantize_lc_lr - input_lr.detach()).pow(2).mean()
            if prior is not None:
                diff_lc_lr += (prior_trans(quantize_lc_lr.permute(0, 3, 1, 2)) - prior).pow(2).mean() * prior_weight
        else:
            diff_lc_lr = (quantize_lc_lr.detach() - input_lr).pow(2).mean()

        quantize_lc_lr = input_lr + (quantize_lc_lr - input_lr).detach()

        if not rtn_embed_sort:
            return quantize_lc_lr, diff_lc_lr, embed_ind_lc_lr, dist_lc_lr
        else:
            return quantize_lc_lr, diff_lc_lr,  torch.flip((-dist_lc_lr).sort(1)[1], [1]), dist_lc_lr

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

    def embed_logits(self, embed_logits):
        return torch.matmul(embed_logits, self.embed.T)

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, code_book_opt=False):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())
        self.code_book_opt = code_book_opt

    def forward(self, input, prior=None, prior_weight=0.1, gt_indice=None, no_quantize=False, rtn_embed_sort=False):

        # L2 Nearest Quantization
        flatten = input.reshape(-1, self.dim)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training and self.code_book_opt:
            if prior is None:
                embed_onehot_sum = embed_onehot.sum(0)
                embed_sum = flatten.transpose(0, 1) @ embed_onehot
            else:
                embed_onehot_sum = embed_onehot.sum(0)
                embed_sum = ((1 - prior_weight) * flatten.transpose(0, 1) + prior_weight * prior.permute(0, 2, 3,1).reshape(-1, self.dim).transpose(0, 1).to(input.device)) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        if gt_indice is not None:
            quantize_gt = self.embed_code(gt_indice)
            diff = (quantize_gt.detach() - input).pow(2).mean()
        else:
            diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        if self.LQ_stage and gt_indice is not None:
            diff = 0.25 * diff + self.gram_loss(input, quantize_gt.detach())

        if not rtn_embed_sort:
            return quantize if not no_quantize else input, diff, embed_ind
        else:
            return quantize if not no_quantize else input, diff, torch.flip((-dist).sort(1)[1], [1])

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

    def gram_loss(self, x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        y = y.reshape(b, h * w, c)

        gmx = x.transpose(1, 2) @ x / (h * w)
        gmy = y.transpose(1, 2) @ y / (h * w)

        return (gmx - gmy).square().mean()




class Dual_Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, code_book_opt=False):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed_lr = torch.randn(dim, n_embed).uniform_(-1.0/self.n_embed, -1.0/self.n_embed)
        embed_hr = torch.randn(dim, n_embed).uniform_(-1.0/self.n_embed, -1.0/self.n_embed)
        self.register_buffer("embed_lr", embed_lr)
        self.register_buffer("cluster_size_lr", torch.zeros(n_embed))
        self.register_buffer("embed_avg_lr", embed_lr.clone())
        self.register_buffer("embed_hr", embed_hr)
        self.register_buffer("cluster_size_hr", torch.zeros(n_embed))
        self.register_buffer("embed_avg_hr", embed_hr.clone())
        self.code_book_opt = code_book_opt
        self.hr_weight = 0.5


    def forward(self, input_hr, input_lr, prior=None, prior_weight=0.1, gt_indice=None, rtn_embed_sort=False, forward_type="parallel"):
        flatten_hr = input_hr.reshape(-1, self.dim)
        flatten_lr = input_lr.reshape(-1, self.dim)

        if forward_type == "parallel":
            dist_lr = (
                    flatten_lr.pow(2).sum(1, keepdim=True)
                    - 2 * flatten_lr @ self.embed_lr
                    + self.embed_lr.pow(2).sum(0, keepdim=True)
            )
            dist_hr = (
                    flatten_hr.pow(2).sum(1, keepdim=True)
                    - 2 * flatten_hr @ self.embed_hr
                    + self.embed_hr.pow(2).sum(0, keepdim=True)
            )
            # L2 Nearest Quantization
            flatten = torch.cat([flatten_lr, flatten_hr], dim=1)
            embed = torch.cat([self.embed_lr, self.embed_hr], dim=0)

            dist = (
                    flatten.pow(2).sum(1, keepdim=True)
                    - 2 * flatten @ embed
                    + embed.pow(2).sum(0, keepdim=True)
            )
            _, embed_ind = (-dist).max(1)
            embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
            embed_ind = embed_ind.view(*input_hr.shape[:-1])

            quantize_hr = self.embed_hr_code(embed_ind)
            quantize_lr = self.embed_lr_code(embed_ind)


            if self.training and self.code_book_opt:
                embed_hr_onehot_sum = embed_onehot.sum(0)
                embed_lr_onehot_sum = embed_onehot.sum(0)

                if prior is None:
                    embed_hr_sum = flatten_hr.transpose(0, 1) @ embed_onehot
                    embed_lr_sum = flatten_lr.transpose(0, 1) @ embed_onehot

                self.cluster_size_hr.data.mul_(self.decay).add_(
                    embed_hr_onehot_sum, alpha=1 - self.decay
                )
                self.cluster_size_lr.data.mul_(self.decay).add_(
                    embed_lr_onehot_sum, alpha=1 - self.decay
                )

                self.embed_avg_hr.data.mul_(self.decay).add_(embed_hr_sum, alpha=1 - self.decay)
                self.embed_avg_lr.data.mul_(self.decay).add_(embed_lr_sum, alpha=1 - self.decay)

                n_hr = self.cluster_size_hr.sum()
                cluster_size_hr = (
                        (self.cluster_size_hr + self.eps) / (n_hr + self.n_embed * self.eps) * n_hr
                )
                embed_normalized_hr = self.embed_avg_hr / cluster_size_hr.unsqueeze(0)
                self.embed_hr.data.copy_(embed_normalized_hr)

                n_lr = self.cluster_size_lr.sum()
                cluster_size_lr = (
                        (self.cluster_size_lr + self.eps) / (n_lr + self.n_embed * self.eps) * n_lr
                )
                embed_normalized_lr = self.embed_avg_lr / cluster_size_lr.unsqueeze(0)
                self.embed_lr.data.copy_(embed_normalized_lr)

            diff_hr = (quantize_hr.detach() - input_hr).pow(2).mean()
            diff_lr = (quantize_lr.detach() - input_lr).pow(2).mean()

            quantize_lr = input_lr + (quantize_lr - input_lr).detach()
            quantize_hr = input_hr + (quantize_hr - input_hr).detach()

            if not rtn_embed_sort:
                return quantize_hr, quantize_lr, diff_hr, diff_lr, embed_ind, embed_ind
            else:
                return quantize_hr, quantize_lr, diff_hr, diff_lr, torch.flip((-dist_hr).sort(1)[1], [1]), torch.flip((-dist_lr).sort(1)[1], [1])

        elif forward_type == "lr2hr":
            # L2 Nearest Quantization
            dist_lr = (
                    flatten_lr.pow(2).sum(1, keepdim=True)
                    - 2 * flatten_lr @ self.embed_lr
                    + self.embed_lr.pow(2).sum(0, keepdim=True)
            )
            dist_hr = (
                    flatten_hr.pow(2).sum(1, keepdim=True)
                    - 2 * flatten_hr @ self.embed_hr
                    + self.embed_hr.pow(2).sum(0, keepdim=True)
            )

            _, embed_ind_lr = (-dist_lr).max(1)
            _, embed_ind_hr = (-dist_hr).max(1)
            embed_onehot_lr = F.one_hot(embed_ind_lr, self.n_embed).type(flatten_lr.dtype)
            embed_ind_lr = embed_ind_lr.view(*input_lr.shape[:-1])
            embed_ind_hr = embed_ind_hr.view(*input_hr.shape[:-1])

            quantize_hr = self.embed_hr_code(embed_ind_lr)
            quantize_lr = self.embed_lr_code(embed_ind_lr)

            if self.training and self.code_book_opt:
                embed_hr_onehot_sum = embed_onehot_lr.sum(0)
                embed_lr_onehot_sum = embed_onehot_lr.sum(0)

                if prior is None:
                    embed_hr_sum = flatten_hr.transpose(0, 1) @ embed_onehot_lr
                    embed_lr_sum = flatten_lr.transpose(0, 1) @ embed_onehot_lr


                self.cluster_size_hr.data.mul_(self.decay).add_(
                    embed_hr_onehot_sum, alpha=1 - self.decay
                )
                self.cluster_size_lr.data.mul_(self.decay).add_(
                    embed_lr_onehot_sum, alpha=1 - self.decay
                )
                self.embed_avg_hr.data.mul_(self.decay).add_(embed_hr_sum, alpha=1 - self.decay)
                self.embed_avg_lr.data.mul_(self.decay).add_(embed_lr_sum, alpha=1 - self.decay)

                n_hr = self.cluster_size_hr.sum()
                cluster_size_hr = (
                        (self.cluster_size_hr + self.eps) / (n_hr + self.n_embed * self.eps) * n_hr
                )
                embed_normalized_hr = self.embed_avg_hr / cluster_size_hr.unsqueeze(0)
                self.embed_hr.data.copy_(embed_normalized_hr)

                n_lr = self.cluster_size_lr.sum()
                cluster_size_lr = (
                        (self.cluster_size_lr + self.eps) / (n_lr + self.n_embed * self.eps) * n_lr
                )
                embed_normalized_lr = self.embed_avg_lr / cluster_size_lr.unsqueeze(0)
                self.embed_lr.data.copy_(embed_normalized_lr)

            diff_hr = (quantize_hr.detach() - input_hr).pow(2).mean()
            diff_lr = (quantize_lr.detach() - input_lr).pow(2).mean()

            quantize_lr = input_lr + (quantize_lr - input_lr).detach()
            quantize_hr = input_hr + (quantize_hr - input_hr).detach()

            if not rtn_embed_sort:
                return quantize_hr, quantize_lr, diff_hr, diff_lr, embed_ind_hr, embed_ind_lr
            else:
                return quantize_hr, quantize_lr, diff_hr, diff_lr, torch.flip((-dist_hr).sort(1)[1], [1]), torch.flip((-dist_lr).sort(1)[1], [1])

    
        elif forward_type == "hr2lr":
            # L2 Nearest Quantization
            dist_lr = (
                    flatten_lr.pow(2).sum(1, keepdim=True)
                    - 2 * flatten_lr @ self.embed_lr
                    + self.embed_lr.pow(2).sum(0, keepdim=True)
            )
            dist_hr = (
                    flatten_hr.pow(2).sum(1, keepdim=True)
                    - 2 * flatten_hr @ self.embed_hr
                    + self.embed_hr.pow(2).sum(0, keepdim=True)
            )
            _, embed_ind_hr = (-dist_hr).max(1)
            _, embed_ind_lr = (-dist_lr).max(1)
            embed_onehot_hr = F.one_hot(embed_ind_hr, self.n_embed).type(flatten_hr.dtype)
            embed_ind_hr = embed_ind_hr.view(*input_hr.shape[:-1])
            embed_ind_lr = embed_ind_lr.view(*input_lr.shape[:-1])

            quantize_hr = self.embed_hr_code(embed_ind_hr)
            quantize_lr = self.embed_lr_code(embed_ind_hr)

            if self.training and self.code_book_opt:
                embed_hr_onehot_sum = embed_onehot_hr.sum(0)
                embed_lr_onehot_sum = embed_onehot_hr.sum(0)

                if prior is None:
                    embed_hr_sum = flatten_hr.transpose(0, 1) @ embed_onehot_hr
                    embed_lr_sum = flatten_lr.transpose(0, 1) @ embed_onehot_hr


                self.cluster_size_hr.data.mul_(self.decay).add_(
                    embed_hr_onehot_sum, alpha=1 - self.decay
                )
                self.cluster_size_lr.data.mul_(self.decay).add_(
                    embed_lr_onehot_sum, alpha=1 - self.decay
                )
                self.embed_avg_hr.data.mul_(self.decay).add_(embed_hr_sum, alpha=1 - self.decay)
                self.embed_avg_lr.data.mul_(self.decay).add_(embed_lr_sum, alpha=1 - self.decay)

                n_hr = self.cluster_size_hr.sum()
                cluster_size_hr = (
                        (self.cluster_size_hr + self.eps) / (n_hr + self.n_embed * self.eps) * n_hr
                )
                embed_normalized_hr = self.embed_avg_hr / cluster_size_hr.unsqueeze(0)
                self.embed_hr.data.copy_(embed_normalized_hr)

                n_lr = self.cluster_size_lr.sum()
                cluster_size_lr = (
                        (self.cluster_size_lr + self.eps) / (n_lr + self.n_embed * self.eps) * n_lr
                )
                embed_normalized_lr = self.embed_avg_lr / cluster_size_lr.unsqueeze(0)
                self.embed_lr.data.copy_(embed_normalized_lr)

            diff_hr = (quantize_hr.detach() - input_hr).pow(2).mean()
            diff_lr = (quantize_lr.detach() - input_lr).pow(2).mean()

            quantize_lr = input_lr + (quantize_lr - input_lr).detach()
            quantize_hr = input_hr + (quantize_hr - input_hr).detach()

            if not rtn_embed_sort:
                return quantize_hr, quantize_lr, diff_hr, diff_lr, embed_ind_hr, embed_ind_lr
            else:
                return quantize_hr, quantize_lr, diff_hr, diff_lr, torch.flip((-dist_hr).sort(1)[1], [1]), torch.flip((-dist_lr).sort(1)[1], [1])
        else:
            raise RuntimeError("no such forward_type")

    def embed_hr_code(self, embed_id):
        return F.embedding(embed_id, self.embed_hr.transpose(0, 1))

    def embed_lr_code(self, embed_id):
        return F.embedding(embed_id, self.embed_lr.transpose(0, 1))

    def gram_loss(self, x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        y = y.reshape(b, h * w, c)

        gmx = x.transpose(1, 2) @ x / (h * w)
        gmy = y.transpose(1, 2) @ y / (h * w)

        return (gmx - gmy).square().mean()


class Dual_Quantize2(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, code_book_opt=False):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed_lr = torch.randn(dim, n_embed).uniform_(-1.0/self.n_embed, -1.0/self.n_embed)
        embed_hr = torch.randn(dim, n_embed).uniform_(-1.0/self.n_embed, -1.0/self.n_embed)
        self.register_buffer("embed_lr", embed_lr)
        self.register_buffer("cluster_size_lr", torch.zeros(n_embed))
        self.register_buffer("embed_avg_lr", embed_lr.clone())
        self.register_buffer("embed_hr", embed_hr)
        self.register_buffer("cluster_size_hr", torch.zeros(n_embed))
        self.register_buffer("embed_avg_hr", embed_hr.clone())
        self.code_book_opt = code_book_opt
        self.hr_weight = 0.5


    def forward(self, input_hr, input_lr, prior=None, prior_weight=0.1, gt_indice=None, rtn_embed_sort=False, forward_type="parallel", normalization=False):
        flatten_hr = input_hr.reshape(-1, self.dim)
        flatten_lr = input_lr.reshape(-1, self.dim)

        if forward_type == "parallel":
            dist_lr = (
                    flatten_lr.pow(2).sum(1, keepdim=True)
                    - 2 * flatten_lr @ self.embed_lr
                    + self.embed_lr.pow(2).sum(0, keepdim=True)
            )
            dist_hr = (
                    flatten_hr.pow(2).sum(1, keepdim=True)
                    - 2 * flatten_hr @ self.embed_hr
                    + self.embed_hr.pow(2).sum(0, keepdim=True)
            )
            # L2 Nearest Quantization
            flatten = torch.cat([flatten_lr, flatten_hr], dim=1)
            embed = torch.cat([self.embed_lr, self.embed_hr], dim=0)

            if not normalization:
                dist = (
                        flatten.pow(2).sum(1, keepdim=True)
                        - 2 * flatten @ embed
                        + embed.pow(2).sum(0, keepdim=True)
                )
            else:
                # print("normalized before distance match")
                dist = (dist_lr - torch.max(dist_lr))/(torch.max(dist_lr)-torch.min(dist_lr)) + (dist_hr - torch.max(dist_hr))/(torch.max(dist_hr)-torch.min(dist_hr))

            _, embed_ind = (-dist).max(1)
            embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
            embed_ind = embed_ind.view(*input_hr.shape[:-1])

            # print(f"dist mean: {torch.mean(dist)}\t dist_hr mean: {torch.mean(dist_hr)}\t dist_lr mean: {torch.mean(dist_lr)}")

            quantize_hr = self.embed_hr_code(embed_ind)
            quantize_lr = self.embed_lr_code(embed_ind)


            if self.training and self.code_book_opt:
                embed_hr_onehot_sum = embed_onehot.sum(0)
                embed_lr_onehot_sum = embed_onehot.sum(0)

                if prior is None:
                    embed_hr_sum = flatten_hr.transpose(0, 1) @ embed_onehot
                    embed_lr_sum = flatten_lr.transpose(0, 1) @ embed_onehot

                self.cluster_size_hr.data.mul_(self.decay).add_(
                    embed_hr_onehot_sum, alpha=1 - self.decay
                )
                self.cluster_size_lr.data.mul_(self.decay).add_(
                    embed_lr_onehot_sum, alpha=1 - self.decay
                )

                self.embed_avg_hr.data.mul_(self.decay).add_(embed_hr_sum, alpha=1 - self.decay)
                self.embed_avg_lr.data.mul_(self.decay).add_(embed_lr_sum, alpha=1 - self.decay)

                n_hr = self.cluster_size_hr.sum()
                cluster_size_hr = (
                        (self.cluster_size_hr + self.eps) / (n_hr + self.n_embed * self.eps) * n_hr
                )
                embed_normalized_hr = self.embed_avg_hr / cluster_size_hr.unsqueeze(0)
                self.embed_hr.data.copy_(embed_normalized_hr)

                n_lr = self.cluster_size_lr.sum()
                cluster_size_lr = (
                        (self.cluster_size_lr + self.eps) / (n_lr + self.n_embed * self.eps) * n_lr
                )
                embed_normalized_lr = self.embed_avg_lr / cluster_size_lr.unsqueeze(0)
                self.embed_lr.data.copy_(embed_normalized_lr)

                quantize_hr = self.embed_hr_code(embed_ind) # update the quantization after codebook update
                quantize_lr = self.embed_lr_code(embed_ind) # update the quantization after codebook update

            diff_hr = (quantize_hr.detach() - input_hr).pow(2).mean()
            diff_lr = (quantize_lr.detach() - input_lr).pow(2).mean()

            quantize_lr = input_lr + (quantize_lr - input_lr).detach()
            quantize_hr = input_hr + (quantize_hr - input_hr).detach()

            if not rtn_embed_sort:
                return quantize_hr, quantize_lr, diff_hr, diff_lr, embed_ind, embed_ind
            else:
                return quantize_hr, quantize_lr, diff_hr, diff_lr, torch.flip((-dist_hr).sort(1)[1], [1]), torch.flip((-dist_lr).sort(1)[1], [1])

        elif forward_type == "lr2hr":
            # L2 Nearest Quantization
            dist_lr = (
                    flatten_lr.pow(2).sum(1, keepdim=True)
                    - 2 * flatten_lr @ self.embed_lr
                    + self.embed_lr.pow(2).sum(0, keepdim=True)
            )
            dist_hr = (
                    flatten_hr.pow(2).sum(1, keepdim=True)
                    - 2 * flatten_hr @ self.embed_hr
                    + self.embed_hr.pow(2).sum(0, keepdim=True)
            )

            _, embed_ind_lr = (-dist_lr).max(1)
            _, embed_ind_hr = (-dist_hr).max(1)
            embed_onehot_lr = F.one_hot(embed_ind_lr, self.n_embed).type(flatten_lr.dtype)
            embed_ind_lr = embed_ind_lr.view(*input_lr.shape[:-1])
            embed_ind_hr = embed_ind_hr.view(*input_hr.shape[:-1])

            quantize_hr = self.embed_hr_code(embed_ind_lr)
            quantize_lr = self.embed_lr_code(embed_ind_lr)

            if self.training and self.code_book_opt:
                embed_hr_onehot_sum = embed_onehot_lr.sum(0)
                embed_lr_onehot_sum = embed_onehot_lr.sum(0)

                if prior is None:
                    embed_hr_sum = flatten_hr.transpose(0, 1) @ embed_onehot_lr
                    embed_lr_sum = flatten_lr.transpose(0, 1) @ embed_onehot_lr


                self.cluster_size_hr.data.mul_(self.decay).add_(
                    embed_hr_onehot_sum, alpha=1 - self.decay
                )
                self.cluster_size_lr.data.mul_(self.decay).add_(
                    embed_lr_onehot_sum, alpha=1 - self.decay
                )
                self.embed_avg_hr.data.mul_(self.decay).add_(embed_hr_sum, alpha=1 - self.decay)
                self.embed_avg_lr.data.mul_(self.decay).add_(embed_lr_sum, alpha=1 - self.decay)

                n_hr = self.cluster_size_hr.sum()
                cluster_size_hr = (
                        (self.cluster_size_hr + self.eps) / (n_hr + self.n_embed * self.eps) * n_hr
                )
                embed_normalized_hr = self.embed_avg_hr / cluster_size_hr.unsqueeze(0)
                self.embed_hr.data.copy_(embed_normalized_hr)

                n_lr = self.cluster_size_lr.sum()
                cluster_size_lr = (
                        (self.cluster_size_lr + self.eps) / (n_lr + self.n_embed * self.eps) * n_lr
                )
                embed_normalized_lr = self.embed_avg_lr / cluster_size_lr.unsqueeze(0)
                self.embed_lr.data.copy_(embed_normalized_lr)

                quantize_hr = self.embed_hr_code(embed_ind_lr) # update the quantization after codebook update
                quantize_lr = self.embed_lr_code(embed_ind_lr) # update the quantization after codebook update

            diff_hr = (input_hr.detach() - input_hr).pow(2).mean() # stop pushing the encoder feature to quantization
            diff_lr = (input_lr.detach() - input_lr).pow(2).mean() # stop pushing the encoder feature to quantization

            quantize_lr = input_lr + (quantize_lr - input_lr).detach()
            quantize_hr = quantize_hr.detach() # stop backward from quantitazion to encoder

            if not rtn_embed_sort:
                return quantize_hr, quantize_lr, diff_hr, diff_lr, embed_ind_hr, embed_ind_lr
            else:
                return quantize_hr, quantize_lr, diff_hr, diff_lr, torch.flip((-dist_hr).sort(1)[1], [1]), torch.flip((-dist_lr).sort(1)[1], [1])

    
        elif forward_type == "hr2lr":
            # L2 Nearest Quantization
            dist_lr = (
                    flatten_lr.pow(2).sum(1, keepdim=True)
                    - 2 * flatten_lr @ self.embed_lr
                    + self.embed_lr.pow(2).sum(0, keepdim=True)
            )
            dist_hr = (
                    flatten_hr.pow(2).sum(1, keepdim=True)
                    - 2 * flatten_hr @ self.embed_hr
                    + self.embed_hr.pow(2).sum(0, keepdim=True)
            )
            _, embed_ind_hr = (-dist_hr).max(1)
            _, embed_ind_lr = (-dist_lr).max(1)
            embed_onehot_hr = F.one_hot(embed_ind_hr, self.n_embed).type(flatten_hr.dtype)
            embed_ind_hr = embed_ind_hr.view(*input_hr.shape[:-1])
            embed_ind_lr = embed_ind_lr.view(*input_lr.shape[:-1])

            quantize_hr = self.embed_hr_code(embed_ind_hr)
            quantize_lr = self.embed_lr_code(embed_ind_hr)

            if self.training and self.code_book_opt:
                embed_hr_onehot_sum = embed_onehot_hr.sum(0)
                embed_lr_onehot_sum = embed_onehot_hr.sum(0)

                if prior is None:
                    embed_hr_sum = flatten_hr.transpose(0, 1) @ embed_onehot_hr
                    embed_lr_sum = flatten_lr.transpose(0, 1) @ embed_onehot_hr


                self.cluster_size_hr.data.mul_(self.decay).add_(
                    embed_hr_onehot_sum, alpha=1 - self.decay
                )
                self.cluster_size_lr.data.mul_(self.decay).add_(
                    embed_lr_onehot_sum, alpha=1 - self.decay
                )
                self.embed_avg_hr.data.mul_(self.decay).add_(embed_hr_sum, alpha=1 - self.decay)
                self.embed_avg_lr.data.mul_(self.decay).add_(embed_lr_sum, alpha=1 - self.decay)

                n_hr = self.cluster_size_hr.sum()
                cluster_size_hr = (
                        (self.cluster_size_hr + self.eps) / (n_hr + self.n_embed * self.eps) * n_hr
                )
                embed_normalized_hr = self.embed_avg_hr / cluster_size_hr.unsqueeze(0)
                self.embed_hr.data.copy_(embed_normalized_hr)

                n_lr = self.cluster_size_lr.sum()
                cluster_size_lr = (
                        (self.cluster_size_lr + self.eps) / (n_lr + self.n_embed * self.eps) * n_lr
                )
                embed_normalized_lr = self.embed_avg_lr / cluster_size_lr.unsqueeze(0)
                self.embed_lr.data.copy_(embed_normalized_lr)

                quantize_hr = self.embed_hr_code(embed_ind_hr) # update the quantization after codebook update
                quantize_lr = self.embed_lr_code(embed_ind_hr) # update the quantization after codebook update
                

            diff_hr = (input_hr.detach() - input_hr).pow(2).mean() # stop pushing the encoder feature to quantization
            diff_lr = (input_lr.detach() - input_lr).pow(2).mean() # stop pushing the encoder feature to quantization

            quantize_lr = quantize_lr.detach() # stop backward from quantitazion to encoder
            quantize_hr = input_hr + (quantize_hr - input_hr).detach()

            if not rtn_embed_sort:
                return quantize_hr, quantize_lr, diff_hr, diff_lr, embed_ind_hr, embed_ind_lr
            else:
                return quantize_hr, quantize_lr, diff_hr, diff_lr, torch.flip((-dist_hr).sort(1)[1], [1]), torch.flip((-dist_lr).sort(1)[1], [1])
        else:
            raise RuntimeError("no such forward_type")

    def embed_hr_code(self, embed_id):
        return F.embedding(embed_id, self.embed_hr.transpose(0, 1))

    def embed_lr_code(self, embed_id):
        return F.embedding(embed_id, self.embed_lr.transpose(0, 1))

    def gram_loss(self, x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        y = y.reshape(b, h * w, c)

        gmx = x.transpose(1, 2) @ x / (h * w)
        gmy = y.transpose(1, 2) @ y / (h * w)

        return (gmx - gmy).square().mean()


# return dist_hr, dist_lr
class Dual_Quantize3(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, code_book_opt=False):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed_lr = torch.randn(dim, n_embed).uniform_(-1.0/self.n_embed, -1.0/self.n_embed)
        embed_hr = torch.randn(dim, n_embed).uniform_(-1.0/self.n_embed, -1.0/self.n_embed)
        self.register_buffer("embed_lr", embed_lr)
        self.register_buffer("cluster_size_lr", torch.zeros(n_embed))
        self.register_buffer("embed_avg_lr", embed_lr.clone())
        self.register_buffer("embed_hr", embed_hr)
        self.register_buffer("cluster_size_hr", torch.zeros(n_embed))
        self.register_buffer("embed_avg_hr", embed_hr.clone())
        self.code_book_opt = code_book_opt
        self.hr_weight = 0.5


    def forward(self, input_hr, input_lr, prior=None, prior_weight=0.1, gt_indice=None, rtn_embed_sort=False, forward_type="parallel", normalization=False):
        flatten_hr = input_hr.reshape(-1, self.dim)
        flatten_lr = input_lr.reshape(-1, self.dim)

        if forward_type == "parallel":
            dist_lr = (
                    flatten_lr.pow(2).sum(1, keepdim=True)
                    - 2 * flatten_lr @ self.embed_lr
                    + self.embed_lr.pow(2).sum(0, keepdim=True)
            )
            dist_hr = (
                    flatten_hr.pow(2).sum(1, keepdim=True)
                    - 2 * flatten_hr @ self.embed_hr
                    + self.embed_hr.pow(2).sum(0, keepdim=True)
            )
            # L2 Nearest Quantization
            flatten = torch.cat([flatten_lr, flatten_hr], dim=1)
            embed = torch.cat([self.embed_lr, self.embed_hr], dim=0)

            if not normalization:
                dist = (
                        flatten.pow(2).sum(1, keepdim=True)
                        - 2 * flatten @ embed
                        + embed.pow(2).sum(0, keepdim=True)
                )
            else:
                # print("normalized before distance match")
                dist = (dist_lr - torch.max(dist_lr))/(torch.max(dist_lr)-torch.min(dist_lr)) + (dist_hr - torch.max(dist_hr))/(torch.max(dist_hr)-torch.min(dist_hr))

            _, embed_ind = (-dist).max(1)
            embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
            embed_ind = embed_ind.view(*input_hr.shape[:-1])

            # print(f"dist mean: {torch.mean(dist)}\t dist_hr mean: {torch.mean(dist_hr)}\t dist_lr mean: {torch.mean(dist_lr)}")

            quantize_hr = self.embed_hr_code(embed_ind)
            quantize_lr = self.embed_lr_code(embed_ind)


            if self.training and self.code_book_opt:
                embed_hr_onehot_sum = embed_onehot.sum(0)
                embed_lr_onehot_sum = embed_onehot.sum(0)

                if prior is None:
                    embed_hr_sum = flatten_hr.transpose(0, 1) @ embed_onehot
                    embed_lr_sum = flatten_lr.transpose(0, 1) @ embed_onehot

                self.cluster_size_hr.data.mul_(self.decay).add_(
                    embed_hr_onehot_sum, alpha=1 - self.decay
                )
                self.cluster_size_lr.data.mul_(self.decay).add_(
                    embed_lr_onehot_sum, alpha=1 - self.decay
                )

                self.embed_avg_hr.data.mul_(self.decay).add_(embed_hr_sum, alpha=1 - self.decay)
                self.embed_avg_lr.data.mul_(self.decay).add_(embed_lr_sum, alpha=1 - self.decay)

                n_hr = self.cluster_size_hr.sum()
                cluster_size_hr = (
                        (self.cluster_size_hr + self.eps) / (n_hr + self.n_embed * self.eps) * n_hr
                )
                embed_normalized_hr = self.embed_avg_hr / cluster_size_hr.unsqueeze(0)
                self.embed_hr.data.copy_(embed_normalized_hr)

                n_lr = self.cluster_size_lr.sum()
                cluster_size_lr = (
                        (self.cluster_size_lr + self.eps) / (n_lr + self.n_embed * self.eps) * n_lr
                )
                embed_normalized_lr = self.embed_avg_lr / cluster_size_lr.unsqueeze(0)
                self.embed_lr.data.copy_(embed_normalized_lr)

                quantize_hr = self.embed_hr_code(embed_ind) # update the quantization after codebook update
                quantize_lr = self.embed_lr_code(embed_ind) # update the quantization after codebook update

            diff_hr = (quantize_hr.detach() - input_hr).pow(2).mean()
            diff_lr = (quantize_lr.detach() - input_lr).pow(2).mean()

            quantize_lr = input_lr + (quantize_lr - input_lr).detach()
            quantize_hr = input_hr + (quantize_hr - input_hr).detach()

            if not rtn_embed_sort:
                return quantize_hr, quantize_lr, diff_hr, diff_lr, embed_ind, embed_ind, dist_hr, dist_lr
            else:
                return quantize_hr, quantize_lr, diff_hr, diff_lr, torch.flip((-dist_hr).sort(1)[1], [1]), torch.flip((-dist_lr).sort(1)[1], [1]), dist_hr, dist_lr

        elif forward_type == "lr2hr":
            # L2 Nearest Quantization
            dist_lr = (
                    flatten_lr.pow(2).sum(1, keepdim=True)
                    - 2 * flatten_lr @ self.embed_lr
                    + self.embed_lr.pow(2).sum(0, keepdim=True)
            )
            dist_hr = (
                    flatten_hr.pow(2).sum(1, keepdim=True)
                    - 2 * flatten_hr @ self.embed_hr
                    + self.embed_hr.pow(2).sum(0, keepdim=True)
            )

            _, embed_ind_lr = (-dist_lr).max(1)
            _, embed_ind_hr = (-dist_hr).max(1)
            embed_onehot_lr = F.one_hot(embed_ind_lr, self.n_embed).type(flatten_lr.dtype)
            embed_ind_lr = embed_ind_lr.view(*input_lr.shape[:-1])
            embed_ind_hr = embed_ind_hr.view(*input_hr.shape[:-1])

            quantize_hr = self.embed_hr_code(embed_ind_lr)
            quantize_lr = self.embed_lr_code(embed_ind_lr)

            if self.training and self.code_book_opt:
                embed_hr_onehot_sum = embed_onehot_lr.sum(0)
                embed_lr_onehot_sum = embed_onehot_lr.sum(0)

                if prior is None:
                    embed_hr_sum = flatten_hr.transpose(0, 1) @ embed_onehot_lr
                    embed_lr_sum = flatten_lr.transpose(0, 1) @ embed_onehot_lr


                self.cluster_size_hr.data.mul_(self.decay).add_(
                    embed_hr_onehot_sum, alpha=1 - self.decay
                )
                self.cluster_size_lr.data.mul_(self.decay).add_(
                    embed_lr_onehot_sum, alpha=1 - self.decay
                )
                self.embed_avg_hr.data.mul_(self.decay).add_(embed_hr_sum, alpha=1 - self.decay)
                self.embed_avg_lr.data.mul_(self.decay).add_(embed_lr_sum, alpha=1 - self.decay)

                n_hr = self.cluster_size_hr.sum()
                cluster_size_hr = (
                        (self.cluster_size_hr + self.eps) / (n_hr + self.n_embed * self.eps) * n_hr
                )
                embed_normalized_hr = self.embed_avg_hr / cluster_size_hr.unsqueeze(0)
                self.embed_hr.data.copy_(embed_normalized_hr)

                n_lr = self.cluster_size_lr.sum()
                cluster_size_lr = (
                        (self.cluster_size_lr + self.eps) / (n_lr + self.n_embed * self.eps) * n_lr
                )
                embed_normalized_lr = self.embed_avg_lr / cluster_size_lr.unsqueeze(0)
                self.embed_lr.data.copy_(embed_normalized_lr)

                quantize_hr = self.embed_hr_code(embed_ind_lr) # update the quantization after codebook update
                quantize_lr = self.embed_lr_code(embed_ind_lr) # update the quantization after codebook update

            diff_hr = (input_hr.detach() - input_hr).pow(2).mean() # stop pushing the encoder feature to quantization
            diff_lr = (input_lr.detach() - input_lr).pow(2).mean() # stop pushing the encoder feature to quantization

            quantize_lr = input_lr + (quantize_lr - input_lr).detach()
            quantize_hr = quantize_hr.detach() # stop backward from quantitazion to encoder

            if not rtn_embed_sort:
                return quantize_hr, quantize_lr, diff_hr, diff_lr, embed_ind_hr, embed_ind_lr, dist_hr, dist_lr
            else:
                return quantize_hr, quantize_lr, diff_hr, diff_lr, torch.flip((-dist_hr).sort(1)[1], [1]), torch.flip((-dist_lr).sort(1)[1], [1]), dist_hr, dist_lr

    
        elif forward_type == "hr2lr":
            # L2 Nearest Quantization
            dist_lr = (
                    flatten_lr.pow(2).sum(1, keepdim=True)
                    - 2 * flatten_lr @ self.embed_lr
                    + self.embed_lr.pow(2).sum(0, keepdim=True)
            )
            dist_hr = (
                    flatten_hr.pow(2).sum(1, keepdim=True)
                    - 2 * flatten_hr @ self.embed_hr
                    + self.embed_hr.pow(2).sum(0, keepdim=True)
            )
            _, embed_ind_hr = (-dist_hr).max(1)
            _, embed_ind_lr = (-dist_lr).max(1)
            embed_onehot_hr = F.one_hot(embed_ind_hr, self.n_embed).type(flatten_hr.dtype)
            embed_ind_hr = embed_ind_hr.view(*input_hr.shape[:-1])
            embed_ind_lr = embed_ind_lr.view(*input_lr.shape[:-1])

            quantize_hr = self.embed_hr_code(embed_ind_hr)
            quantize_lr = self.embed_lr_code(embed_ind_hr)

            if self.training and self.code_book_opt:
                embed_hr_onehot_sum = embed_onehot_hr.sum(0)
                embed_lr_onehot_sum = embed_onehot_hr.sum(0)

                if prior is None:
                    embed_hr_sum = flatten_hr.transpose(0, 1) @ embed_onehot_hr
                    embed_lr_sum = flatten_lr.transpose(0, 1) @ embed_onehot_hr


                self.cluster_size_hr.data.mul_(self.decay).add_(
                    embed_hr_onehot_sum, alpha=1 - self.decay
                )
                self.cluster_size_lr.data.mul_(self.decay).add_(
                    embed_lr_onehot_sum, alpha=1 - self.decay
                )
                self.embed_avg_hr.data.mul_(self.decay).add_(embed_hr_sum, alpha=1 - self.decay)
                self.embed_avg_lr.data.mul_(self.decay).add_(embed_lr_sum, alpha=1 - self.decay)

                n_hr = self.cluster_size_hr.sum()
                cluster_size_hr = (
                        (self.cluster_size_hr + self.eps) / (n_hr + self.n_embed * self.eps) * n_hr
                )
                embed_normalized_hr = self.embed_avg_hr / cluster_size_hr.unsqueeze(0)
                self.embed_hr.data.copy_(embed_normalized_hr)

                n_lr = self.cluster_size_lr.sum()
                cluster_size_lr = (
                        (self.cluster_size_lr + self.eps) / (n_lr + self.n_embed * self.eps) * n_lr
                )
                embed_normalized_lr = self.embed_avg_lr / cluster_size_lr.unsqueeze(0)
                self.embed_lr.data.copy_(embed_normalized_lr)

                quantize_hr = self.embed_hr_code(embed_ind_hr) # update the quantization after codebook update
                quantize_lr = self.embed_lr_code(embed_ind_hr) # update the quantization after codebook update
                

            diff_hr = (input_hr.detach() - input_hr).pow(2).mean() # stop pushing the encoder feature to quantization
            diff_lr = (input_lr.detach() - input_lr).pow(2).mean() # stop pushing the encoder feature to quantization

            quantize_lr = quantize_lr.detach() # stop backward from quantitazion to encoder
            quantize_hr = input_hr + (quantize_hr - input_hr).detach()

            if not rtn_embed_sort:
                return quantize_hr, quantize_lr, diff_hr, diff_lr, embed_ind_hr, embed_ind_lr, dist_hr, dist_lr
            else:
                return quantize_hr, quantize_lr, diff_hr, diff_lr, torch.flip((-dist_hr).sort(1)[1], [1]), torch.flip((-dist_lr).sort(1)[1], [1]), dist_hr, dist_lr
        else:
            raise RuntimeError("no such forward_type")

    def embed_hr_code(self, embed_id):
        return F.embedding(embed_id, self.embed_hr.transpose(0, 1))

    def embed_lr_code(self, embed_id):
        return F.embedding(embed_id, self.embed_lr.transpose(0, 1))

    def gram_loss(self, x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        y = y.reshape(b, h * w, c)

        gmx = x.transpose(1, 2) @ x / (h * w)
        gmy = y.transpose(1, 2) @ y / (h * w)

        return (gmx - gmy).square().mean()



class Dual_Quantize4(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, code_book_opt=False):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed_lr = torch.randn(dim, n_embed).uniform_(-1.0/self.n_embed, -1.0/self.n_embed)
        embed_hr = torch.randn(dim, n_embed).uniform_(-1.0/self.n_embed, -1.0/self.n_embed)
        self.register_buffer("embed_lr", embed_lr)
        self.register_buffer("cluster_size_lr", torch.zeros(n_embed))
        self.register_buffer("embed_avg_lr", embed_lr.clone())
        self.register_buffer("embed_hr", embed_hr)
        self.register_buffer("cluster_size_hr", torch.zeros(n_embed))
        self.register_buffer("embed_avg_hr", embed_hr.clone())
        self.code_book_opt = code_book_opt
        self.hr_weight = 0.5


    def forward(self, input_hr, input_lr, prior=None, prior_weight=0.1, gt_indice=None, rtn_embed_sort=False):
        flatten_hr = input_hr.reshape(-1, self.dim)
        flatten_lr = input_lr.reshape(-1, self.dim)

        
        # upgrade embed_lr
        dist_lc_lr = (
                flatten_lr.pow(2).sum(1, keepdim=True)
                - 2 * flatten_lr @ self.embed_lr
                + self.embed_lr.pow(2).sum(0, keepdim=True)
        )
        dist_lc_hr = (
                flatten_hr.pow(2).sum(1, keepdim=True)
                - 2 * flatten_hr @ self.embed_lr
                + self.embed_lr.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind_lc_lr = (-dist_lc_lr).max(1)
        embed_onehot_lc_lr = F.one_hot(embed_ind_lc_lr, self.n_embed).type(flatten_lr.dtype)
        embed_ind_lc_lr = embed_ind_lc_lr.view(*input_lr.shape[:-1])

        _, embed_ind_lc_hr = (-dist_lc_hr).max(1)
        embed_onehot_lc_hr = F.one_hot(embed_ind_lc_hr, self.n_embed).type(flatten_hr.dtype)
        embed_ind_lc_hr = embed_ind_lc_hr.view(*input_hr.shape[:-1])

        quantize_lc_hr = self.embed_lr_code(embed_ind_lc_hr)
        quantize_lc_lr = self.embed_lr_code(embed_ind_lc_lr)

        if self.training and self.code_book_opt:
            embed_lc_hr_onehot_sum = embed_onehot_lc_hr.sum(0)
            embed_lc_lr_onehot_sum = embed_onehot_lc_lr.sum(0)

            embed_lc_onehot_sum = embed_lc_hr_onehot_sum + embed_lc_lr_onehot_sum

            if prior is None:
                embed_lc_hr_sum = flatten_hr.transpose(0, 1) @ embed_onehot_lc_hr
                embed_lc_lr_sum = flatten_lr.transpose(0, 1) @ embed_onehot_lc_lr
                embed_lc_sum = embed_lc_hr_sum + embed_lc_lr_sum

            self.cluster_size_lr.data.mul_(self.decay).add_(
                embed_lc_onehot_sum, alpha=1 - self.decay
            )

            self.embed_avg_lr.data.mul_(self.decay).add_(embed_lc_sum, alpha=1 - self.decay)

            n_lr = self.cluster_size_lr.sum()
            cluster_size_lr = (
                    (self.cluster_size_lr + self.eps) / (n_lr + self.n_embed * self.eps) * n_lr
            )
            embed_normalized_lr = self.embed_avg_lr / cluster_size_lr.unsqueeze(0)
            self.embed_lr.data.copy_(embed_normalized_lr)

        diff_lc_hr = (quantize_lc_hr.detach() - input_hr).pow(2).mean()
        diff_lc_lr = (quantize_lc_lr.detach() - input_lr).pow(2).mean()

        quantize_lc_lr = input_lr + (quantize_lc_lr - input_lr).detach()
        quantize_lc_hr = input_hr + (quantize_lc_hr - input_hr).detach()

        # upgrade embed_hr
        dist_hc_lr = (
                flatten_lr.pow(2).sum(1, keepdim=True)
                - 2 * flatten_lr @ self.embed_lr
                + self.embed_lr.pow(2).sum(0, keepdim=True)
        )
        dist_hc_hr = (
                flatten_hr.pow(2).sum(1, keepdim=True)
                - 2 * flatten_hr @ self.embed_lr
                + self.embed_lr.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind_hc_lr = (-dist_hc_lr).max(1)
        embed_onehot_hc_lr = F.one_hot(embed_ind_hc_lr, self.n_embed).type(flatten_lr.dtype)
        embed_ind_hc_lr = embed_ind_hc_lr.view(*input_lr.shape[:-1])

        _, embed_ind_hc_hr = (-dist_hc_hr).max(1)
        embed_onehot_hc_hr = F.one_hot(embed_ind_hc_hr, self.n_embed).type(flatten_hr.dtype)
        embed_ind_hc_hr = embed_ind_hc_hr.view(*input_hr.shape[:-1])

        quantize_hc_hr = self.embed_lr_code(embed_ind_hc_hr)
        quantize_hc_lr = self.embed_lr_code(embed_ind_hc_lr)
        
        if self.training and self.code_book_opt:
            embed_hc_hr_onehot_sum = embed_onehot_hc_hr.sum(0)
            embed_hc_lr_onehot_sum = embed_onehot_hc_lr.sum(0)

            embed_hc_onehot_sum = embed_hc_hr_onehot_sum + embed_hc_lr_onehot_sum

            if prior is None:
                embed_hc_hr_sum = flatten_hr.transpose(0, 1) @ embed_onehot_hc_hr
                embed_hc_lr_sum = flatten_lr.transpose(0, 1) @ embed_onehot_hc_lr
                embed_hc_sum = embed_hc_hr_sum + embed_hc_lr_sum

            self.cluster_size_lr.data.mul_(self.decay).add_(
                embed_hc_onehot_sum, alpha=1 - self.decay
            )

            self.embed_avg_lr.data.mul_(self.decay).add_(embed_hc_sum, alpha=1 - self.decay)

            n_lr = self.cluster_size_lr.sum()
            cluster_size_lr = (
                    (self.cluster_size_lr + self.eps) / (n_lr + self.n_embed * self.eps) * n_lr
            )
            embed_normalized_lr = self.embed_avg_lr / cluster_size_lr.unsqueeze(0)
            self.embed_lr.data.copy_(embed_normalized_lr)

        diff_hc_hr = (quantize_hc_hr.detach() - input_hr).pow(2).mean()
        diff_hc_lr = (quantize_hc_lr.detach() - input_lr).pow(2).mean()

        quantize_hc_lr = input_lr + (quantize_hc_lr - input_lr).detach()
        quantize_hc_hr = input_hr + (quantize_hc_hr - input_hr).detach()




        if not rtn_embed_sort:
            return quantize_hc_hr, quantize_hc_lr, quantize_lc_hr, quantize_lc_lr, diff_hc_hr, diff_hc_lr, diff_lc_hr, diff_lc_lr, embed_ind_hc_hr, embed_ind_hc_lr, embed_ind_lc_hr, embed_ind_lc_lr, dist_hc_hr, dist_hc_lr, dist_lc_hr, dist_lc_lr
        else:
            return quantize_hc_hr, quantize_hc_lr, quantize_lc_hr, quantize_lc_lr, diff_hc_hr, diff_hc_lr, diff_lc_hr, diff_lc_lr, torch.flip((-dist_hc_hr).sort(1)[1], [1]), torch.flip((-dist_hc_lr).sort(1)[1], [1]), torch.flip((-dist_lc_hr).sort(1)[1], [1]), torch.flip((-dist_lc_lr).sort(1)[1], [1]), dist_hc_hr, dist_hc_lr, dist_lc_hr, dist_lc_lr


    

    def embed_hr_code(self, embed_id):
        return F.embedding(embed_id, self.embed_hr.transpose(0, 1))

    def embed_lr_code(self, embed_id):
        return F.embedding(embed_id, self.embed_lr.transpose(0, 1))



class Dual_Quantize5(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, code_book_opt=False):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        self.codebook_lr = nn.Embedding(dim, n_embed)
        self.codebook_hr = nn.Embedding(dim, n_embed)
        self.codebook_lr.weight.data.uniform_(-1.0/self.n_embed, -1.0/self.n_embed)
        self.codebook_hr.weight.data.copy_(self.codebook_lr.weight)
        self.embed_lr = self.codebook_lr.weight
        self.embed_hr = self.codebook_hr.weight
        self.code_book_opt = code_book_opt
        self.hr_weight = 0.5


    def forward(self, input_hr, input_lr, gt_indice=None, rtn_embed_sort=False, prior=None, prior_weight=0.1, prior_trans=nn.Identity()):
        flatten_hr = input_hr.reshape(-1, self.dim)
        flatten_lr = input_lr.reshape(-1, self.dim)

        
        # upgrade embed_lr
        dist_lc_lr = (
                flatten_lr.pow(2).sum(1, keepdim=True)
                - 2 * flatten_lr @ self.embed_lr
                + self.embed_lr.pow(2).sum(0, keepdim=True)
        )
        dist_lc_hr = (
                flatten_hr.pow(2).sum(1, keepdim=True)
                - 2 * flatten_hr @ self.embed_lr
                + self.embed_lr.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind_lc_lr = (-dist_lc_lr).max(1)
        embed_onehot_lc_lr = F.one_hot(embed_ind_lc_lr, self.n_embed).type(flatten_lr.dtype)
        embed_ind_lc_lr = embed_ind_lc_lr.view(*input_lr.shape[:-1])

        _, embed_ind_lc_hr = (-dist_lc_hr).max(1)
        embed_onehot_lc_hr = F.one_hot(embed_ind_lc_hr, self.n_embed).type(flatten_hr.dtype)
        embed_ind_lc_hr = embed_ind_lc_hr.view(*input_hr.shape[:-1])

        quantize_lc_hr = self.embed_lr_code(embed_ind_lc_hr)
        quantize_lc_lr = self.embed_lr_code(embed_ind_lc_lr)

        if self.training and self.code_book_opt:
            diff_lc_hr = (quantize_lc_hr.detach() - input_hr).pow(2).mean()*0.25 + (quantize_lc_hr - input_hr.detach()).pow(2).mean()
            diff_lc_lr = (quantize_lc_lr.detach() - input_lr).pow(2).mean()*0.25 + (quantize_lc_lr - input_lr.detach()).pow(2).mean()
            if prior is not None:
                diff_lc_hr += (prior_trans(quantize_lc_hr.permute(0, 3, 1, 2)) - prior).pow(2).mean() * prior_weight
                diff_lc_lr += (prior_trans(quantize_lc_lr.permute(0, 3, 1, 2)) - prior).pow(2).mean() * prior_weight
        else:
            diff_lc_hr = (quantize_lc_hr.detach() - input_hr).pow(2).mean()
            diff_lc_lr = (quantize_lc_lr.detach() - input_lr).pow(2).mean()

        quantize_lc_lr = input_lr + (quantize_lc_lr - input_lr).detach()
        quantize_lc_hr = input_hr + (quantize_lc_hr - input_hr).detach()

        # upgrade embed_hr
        dist_hc_lr = (
                flatten_lr.pow(2).sum(1, keepdim=True)
                - 2 * flatten_lr @ self.embed_lr
                + self.embed_lr.pow(2).sum(0, keepdim=True)
        )
        dist_hc_hr = (
                flatten_hr.pow(2).sum(1, keepdim=True)
                - 2 * flatten_hr @ self.embed_lr
                + self.embed_lr.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind_hc_lr = (-dist_hc_lr).max(1)
        embed_onehot_hc_lr = F.one_hot(embed_ind_hc_lr, self.n_embed).type(flatten_lr.dtype)
        embed_ind_hc_lr = embed_ind_hc_lr.view(*input_lr.shape[:-1])

        _, embed_ind_hc_hr = (-dist_hc_hr).max(1)
        embed_onehot_hc_hr = F.one_hot(embed_ind_hc_hr, self.n_embed).type(flatten_hr.dtype)
        embed_ind_hc_hr = embed_ind_hc_hr.view(*input_hr.shape[:-1])

        quantize_hc_hr = self.embed_lr_code(embed_ind_hc_hr)
        quantize_hc_lr = self.embed_lr_code(embed_ind_hc_lr)
        
        if self.training and self.code_book_opt:
            diff_hc_hr = (quantize_hc_hr.detach() - input_hr).pow(2).mean()* 0.25 + (quantize_hc_hr - input_hr.detach()).pow(2).mean()
            diff_hc_lr = (quantize_hc_lr.detach() - input_lr).pow(2).mean()* 0.25 + (quantize_hc_lr - input_lr.detach()).pow(2).mean()
        else:
            diff_hc_hr = (quantize_hc_hr.detach() - input_hr).pow(2).mean()
            diff_hc_lr = (quantize_hc_lr.detach() - input_lr).pow(2).mean()

        quantize_hc_lr = input_lr + (quantize_hc_lr - input_lr).detach()
        quantize_hc_hr = input_hr + (quantize_hc_hr - input_hr).detach()


        if not rtn_embed_sort:
            return quantize_hc_hr, quantize_hc_lr, quantize_lc_hr, quantize_lc_lr, diff_hc_hr, diff_hc_lr, diff_lc_hr, diff_lc_lr, embed_ind_hc_hr, embed_ind_hc_lr, embed_ind_lc_hr, embed_ind_lc_lr, dist_hc_hr, dist_hc_lr, dist_lc_hr, dist_lc_lr
        else:
            return quantize_hc_hr, quantize_hc_lr, quantize_lc_hr, quantize_lc_lr, diff_hc_hr, diff_hc_lr, diff_lc_hr, diff_lc_lr, torch.flip((-dist_hc_hr).sort(1)[1], [1]), torch.flip((-dist_hc_lr).sort(1)[1], [1]), torch.flip((-dist_lc_hr).sort(1)[1], [1]), torch.flip((-dist_lc_lr).sort(1)[1], [1]), dist_hc_hr, dist_hc_lr, dist_lc_hr, dist_lc_lr


    

    def embed_hr_code(self, embed_id):
        return F.embedding(embed_id, self.embed_hr.transpose(0, 1))

    def embed_lr_code(self, embed_id):
        return F.embedding(embed_id, self.embed_lr.transpose(0, 1))



class Dual_Quantize6(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, code_book_opt=False):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        self.codebook_lr = nn.Embedding(dim, n_embed)
        self.codebook_hr = nn.Embedding(dim, n_embed)
        self.codebook_lr.weight.data.uniform_(-1.0/self.n_embed, -1.0/self.n_embed)
        self.codebook_hr.weight.data.copy_(self.codebook_lr.weight)
        self.embed_lr = self.codebook_lr.weight
        self.embed_hr = self.codebook_hr.weight
        self.code_book_opt = code_book_opt
        self.hr_weight = 0.5


    def forward(self, input_hr, input_lr, prior=None, prior_weight=0.1, gt_indice=None, rtn_embed_sort=False):
        flatten_hr = input_hr.reshape(-1, self.dim)
        flatten_lr = input_lr.reshape(-1, self.dim)

        
        # upgrade embed_lr
        dist_lc_lr = (
                flatten_lr.pow(2).sum(1, keepdim=True)
                - 2 * flatten_lr @ self.embed_lr
                + self.embed_lr.pow(2).sum(0, keepdim=True)
        )
        dist_lc_hr = (
                flatten_hr.pow(2).sum(1, keepdim=True)
                - 2 * flatten_hr @ self.embed_lr
                + self.embed_lr.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind_lc_lr = (-dist_lc_lr).max(1)
        embed_onehot_lc_lr = F.one_hot(embed_ind_lc_lr, self.n_embed).type(flatten_lr.dtype)
        embed_ind_lc_lr = embed_ind_lc_lr.view(*input_lr.shape[:-1])

        _, embed_ind_lc_hr = (-dist_lc_hr).max(1)
        embed_onehot_lc_hr = F.one_hot(embed_ind_lc_hr, self.n_embed).type(flatten_hr.dtype)
        embed_ind_lc_hr = embed_ind_lc_hr.view(*input_hr.shape[:-1])

        quantize_lc_hr = self.embed_lr_code(embed_ind_lc_hr)
        quantize_lc_lr = self.embed_lr_code(embed_ind_lc_lr)

        if self.training and self.code_book_opt:
            diff_lc_hr = (quantize_lc_hr.detach() - input_hr).pow(2).mean()*0.25 + (quantize_lc_hr - input_hr.detach()).pow(2).mean()
            diff_lc_lr = (quantize_lc_lr.detach() - input_lr).pow(2).mean()*0.25 + (quantize_lc_lr - input_lr.detach()).pow(2).mean()
        else:
            diff_lc_hr = (quantize_lc_hr.detach() - input_hr).pow(2).mean()
            diff_lc_lr = (quantize_lc_lr.detach() - input_lr).pow(2).mean()

        quantize_lc_lr = input_lr + (quantize_lc_lr - input_lr).detach()
        quantize_lc_hr = input_hr + (quantize_lc_hr - input_hr).detach()

        # upgrade embed_hr
        dist_hc_lr = (
                flatten_lr.pow(2).sum(1, keepdim=True)
                - 2 * flatten_lr @ self.embed_hr
                + self.embed_hr.pow(2).sum(0, keepdim=True)
        )
        dist_hc_hr = (
                flatten_hr.pow(2).sum(1, keepdim=True)
                - 2 * flatten_hr @ self.embed_hr
                + self.embed_hr.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind_hc_lr = (-dist_hc_lr).max(1)
        embed_onehot_hc_lr = F.one_hot(embed_ind_hc_lr, self.n_embed).type(flatten_lr.dtype)
        embed_ind_hc_lr = embed_ind_hc_lr.view(*input_lr.shape[:-1])

        _, embed_ind_hc_hr = (-dist_hc_hr).max(1)
        embed_onehot_hc_hr = F.one_hot(embed_ind_hc_hr, self.n_embed).type(flatten_hr.dtype)
        embed_ind_hc_hr = embed_ind_hc_hr.view(*input_hr.shape[:-1])

        quantize_hc_hr = self.embed_hr_code(embed_ind_hc_hr)
        quantize_hc_lr = self.embed_hr_code(embed_ind_hc_lr) if self.training else self.embed_hr_code(embed_ind_lc_lr)
        
        if self.training and self.code_book_opt:
            diff_hc_hr = (quantize_hc_hr.detach() - input_hr).pow(2).mean()* 0.25 + (quantize_hc_hr - input_hr.detach()).pow(2).mean()
            diff_hc_lr = (quantize_hc_lr.detach() - input_lr).pow(2).mean()* 0.25 + (quantize_hc_lr - input_lr.detach()).pow(2).mean()
        else:
            diff_hc_hr = (quantize_hc_hr.detach() - input_hr).pow(2).mean()
            diff_hc_lr = (quantize_hc_lr.detach() - input_lr).pow(2).mean()

        quantize_hc_lr = input_lr + (quantize_hc_lr - input_lr).detach()
        quantize_hc_hr = input_hr + (quantize_hc_hr - input_hr).detach()




        if not rtn_embed_sort:
            return quantize_hc_hr, quantize_hc_lr, quantize_lc_hr, quantize_lc_lr, diff_hc_hr, diff_hc_lr, diff_lc_hr, diff_lc_lr, embed_ind_hc_hr, embed_ind_hc_lr, embed_ind_lc_hr, embed_ind_lc_lr, dist_hc_hr, dist_hc_lr, dist_lc_hr, dist_lc_lr
        else:
            return quantize_hc_hr, quantize_hc_lr, quantize_lc_hr, quantize_lc_lr, diff_hc_hr, diff_hc_lr, diff_lc_hr, diff_lc_lr, torch.flip((-dist_hc_hr).sort(1)[1], [1]), torch.flip((-dist_hc_lr).sort(1)[1], [1]), torch.flip((-dist_lc_hr).sort(1)[1], [1]), torch.flip((-dist_lc_lr).sort(1)[1], [1]), dist_hc_hr, dist_hc_lr, dist_lc_hr, dist_lc_lr


    

    def embed_hr_code(self, embed_id):
        return F.embedding(embed_id, self.embed_hr.transpose(0, 1))

    def embed_lr_code(self, embed_id):
        return F.embedding(embed_id, self.embed_lr.transpose(0, 1))



class VQEmbedding(nn.Embedding):
    """VQ embedding module with ema update."""

    def __init__(self, n_embed, embed_dim, ema=True, decay=0.99, restart_unused_codes=True, eps=1e-5):
        super().__init__(n_embed + 1, embed_dim, padding_idx=n_embed)

        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed

        if self.ema:
            _ = [p.requires_grad_(False) for p in self.parameters()]

            # padding index is not updated by EMA
            self.register_buffer('cluster_size_ema', torch.zeros(n_embed))
            self.register_buffer('embed_ema', self.weight[:-1, :].detach().clone())

    @torch.no_grad()
    def compute_distances(self, inputs):
        codebook_t = self.weight[:-1, :].t()

        (embed_dim, _) = codebook_t.shape
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == embed_dim

        inputs_flat = inputs.reshape(-1, embed_dim)

        inputs_norm_sq = inputs_flat.pow(2.).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = codebook_t.pow(2.).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            inputs_flat,
            codebook_t,
            alpha=-2.0,
        )
        distances = distances.reshape(*inputs_shape[:-1], -1)  # [B, h, w, n_embed or n_embed+1]
        return distances

    @torch.no_grad()
    def find_nearest_embedding(self, inputs):
        distances = self.compute_distances(inputs)  # [B, h, w, n_embed or n_embed+1]
        embed_idxs = distances.argmin(dim=-1)  # use padding index or not

        return embed_idxs

    @torch.no_grad()
    def _tile_with_noise(self, x, target_n):
        B, embed_dim = x.shape
        n_repeats = (target_n + B -1) // B
        std = x.new_ones(embed_dim) * 0.01 / np.sqrt(embed_dim)
        x = x.repeat(n_repeats, 1)
        x = x + torch.rand_like(x) * std
        return x    
    
    @torch.no_grad()
    def _update_buffers(self, vectors, idxs):

        n_embed, embed_dim = self.weight.shape[0]-1, self.weight.shape[-1]
        
        vectors = vectors.reshape(-1, embed_dim)
        idxs = idxs.reshape(-1)
        
        n_vectors = vectors.shape[0]
        n_total_embed = n_embed

        one_hot_idxs = vectors.new_zeros(n_total_embed, n_vectors)
        one_hot_idxs.scatter_(dim=0,
                              index=idxs.unsqueeze(0),
                              src=vectors.new_ones(1, n_vectors)
                              )

        cluster_size = one_hot_idxs.sum(dim=1)
        vectors_sum_per_cluster = one_hot_idxs @ vectors

        self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_ema.mul_(self.decay).add_(vectors_sum_per_cluster, alpha=1 - self.decay)
        
        if self.restart_unused_codes:
            if n_vectors < n_embed:
                vectors = self._tile_with_noise(vectors, n_embed)
            n_vectors = vectors.shape[0]
            _vectors_random = vectors[torch.randperm(n_vectors, device=vectors.device)][:n_embed]
            
        
            usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()
            self.embed_ema.mul_(usage).add_(_vectors_random * (1-usage))
            self.cluster_size_ema.mul_(usage.view(-1))
            self.cluster_size_ema.add_(torch.ones_like(self.cluster_size_ema) * (1-usage).view(-1))

    @torch.no_grad()
    def _update_embedding(self):

        n_embed = self.weight.shape[0] - 1
        n = self.cluster_size_ema.sum()
        normalized_cluster_size = (
            n * (self.cluster_size_ema + self.eps) / (n + n_embed * self.eps)
        )
        self.weight[:-1, :] = self.embed_ema / normalized_cluster_size.reshape(-1, 1)

    def forward(self, inputs):
        embed_idxs = self.find_nearest_embedding(inputs)
        if self.training:
            if self.ema:
                self._update_buffers(inputs, embed_idxs)
        
        embeds = self.embed(embed_idxs)

        if self.ema and self.training:
            self._update_embedding()

        return embeds, embed_idxs

    def embed(self, idxs):
        embeds = super().forward(idxs)
        return embeds


class RQBottleneck(nn.Module):
    """
    Quantization bottleneck via Residual Quantization.
    Arguments:
        latent_shape (Tuple[int, int, int]): the shape of latents, denoted (H, W, D)
        code_shape (Tuple[int, int, int]): the shape of codes, denoted (h, w, d)
        n_embed (int, List, or Tuple): the number of embeddings (i.e., the size of codebook)
            If isinstance(n_embed, int), the sizes of all codebooks are same.
        shared_codebook (bool): If True, codebooks are shared in all location. If False,
            uses separate codebooks along the ``depth'' dimension. (default: False)
        restart_unused_codes (bool): If True, it randomly assigns a feature vector in the curruent batch
            as the new embedding of unused codes in training. (default: True)
    """

    def __init__(self,
                 latent_shape,
                 code_shape,
                 n_embed,
                 decay=0.99,
                 shared_codebook=False,
                 restart_unused_codes=True,
                 commitment_loss='cumsum'
                 ):
        super().__init__()

        if not len(code_shape) == len(latent_shape) == 3:
            raise ValueError("incompatible code shape or latent shape")
        if any([y % x != 0 for x, y in zip(code_shape[:2], latent_shape[:2])]):
            raise ValueError("incompatible code shape or latent shape")

        #residual quantization does not divide feature dims for quantization.
        embed_dim = np.prod(latent_shape[:2]) // np.prod(code_shape[:2]) * latent_shape[2]

        self.latent_shape = torch.Size(latent_shape)
        self.code_shape = torch.Size(code_shape)
        self.shape_divisor = torch.Size([latent_shape[i] // code_shape[i] for i in range(len(latent_shape))])
        
        self.shared_codebook = shared_codebook
        if self.shared_codebook:
            if isinstance(n_embed, Iterable) or isinstance(decay, Iterable):
                raise ValueError("Shared codebooks are incompatible \
                                    with list types of momentums or sizes: Change it into int")

        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed if isinstance(n_embed, Iterable) else [n_embed for _ in range(self.code_shape[-1])]
        self.decay = decay if isinstance(decay, Iterable) else [decay for _ in range(self.code_shape[-1])]
        assert len(self.n_embed) == self.code_shape[-1]
        assert len(self.decay) == self.code_shape[-1]

        if self.shared_codebook:
            codebook0 = VQEmbedding(self.n_embed[0], 
                                    embed_dim, 
                                    decay=self.decay[0], 
                                    restart_unused_codes=restart_unused_codes,
                                    )
            self.codebooks = nn.ModuleList([codebook0 for _ in range(self.code_shape[-1])])
        else:
            codebooks = [VQEmbedding(self.n_embed[idx], 
                                     embed_dim, 
                                     decay=self.decay[idx], 
                                     restart_unused_codes=restart_unused_codes,
                                     ) for idx in range(self.code_shape[-1])]
            self.codebooks = nn.ModuleList(codebooks)

        self.commitment_loss = commitment_loss

    def to_code_shape(self, x):
        (B, H, W, D) = x.shape
        (rH, rW, _) = self.shape_divisor

        x = x.reshape(B, H//rH, rH, W//rW, rW, D)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, H//rH, W//rW, -1)

        return x

    def to_latent_shape(self, x):
        (B, h, w, _) = x.shape
        (_, _, D) = self.latent_shape
        (rH, rW, _) = self.shape_divisor

        x = x.reshape(B, h, w, rH, rW, D)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, h*rH, w*rW, D)

        return x

    def quantize(self, x):
        r"""
        Return list of quantized features and the selected codewords by the residual quantization.
        The code is selected by the residuals between x and quantized features by the previous codebooks.
        Arguments:
            x (Tensor): bottleneck feature maps to quantize.
        Returns:
            quant_list (list): list of sequentially aggregated and quantized feature maps by codebooks.
            codes (LongTensor): codewords index, corresponding to quants.
        Shape:
            - x: (B, h, w, embed_dim)
            - quant_list[i]: (B, h, w, embed_dim)
            - codes: (B, h, w, d)
        """
        B, h, w, embed_dim = x.shape

        residual_feature = x.detach().clone()

        quant_list = []
        code_list = []
        aggregated_quants = torch.zeros_like(x)
        for i in range(self.code_shape[-1]):
            quant, code = self.codebooks[i](residual_feature)

            residual_feature.sub_(quant)
            aggregated_quants.add_(quant)

            quant_list.append(aggregated_quants.clone())
            code_list.append(code.unsqueeze(-1))
        
        codes = torch.cat(code_list, dim=-1)
        return quant_list, codes

    def forward(self, x):
        x_reshaped = self.to_code_shape(x)
        quant_list, codes = self.quantize(x_reshaped)

        commitment_loss = self.compute_commitment_loss(x_reshaped, quant_list)
        quants_trunc = self.to_latent_shape(quant_list[-1])
        quants_trunc = x + (quants_trunc - x).detach()

        return quants_trunc, commitment_loss, codes
    
    def compute_commitment_loss(self, x, quant_list):
        r"""
        Compute the commitment loss for the residual quantization.
        The loss is iteratively computed by aggregating quantized features.
        """
        loss_list = []
        
        for idx, quant in enumerate(quant_list):
            partial_loss = (x-quant.detach()).pow(2.0).mean()
            loss_list.append(partial_loss)
        
        commitment_loss = torch.mean(torch.stack(loss_list))
        return commitment_loss
    
    @torch.no_grad()
    def embed_code(self, code):
        assert code.shape[1:] == self.code_shape
        
        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)

        if self.shared_codebook:
            embeds = [self.codebooks[0].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        else:
            embeds = [self.codebooks[i].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        
        embeds = torch.cat(embeds, dim=-2).sum(-2)
        embeds = self.to_latent_shape(embeds)

        return embeds
    
    @torch.no_grad()
    def embed_code_with_depth(self, code, to_latent_shape=False):
        '''
        do not reduce the code embedding over the axis of code-depth.
        
        Caution: RQ-VAE does not use scale of codebook, thus assume all scales are ones.
        '''
        # spatial resolution can be different in the sampling process
        assert code.shape[-1] == self.code_shape[-1]
        
        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)

        if self.shared_codebook:
            embeds = [self.codebooks[0].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        else:
            embeds = [self.codebooks[i].embed(code_slice) for i, code_slice in enumerate(code_slices)]

        if to_latent_shape:
            embeds = [self.to_latent_shape(embed.squeeze(-2)).unsqueeze(-2) for embed in embeds]
        embeds = torch.cat(embeds, dim=-2)
        
        return embeds, None

    @torch.no_grad()
    def embed_partial_code(self, code, code_idx, decode_type='select'):
        r"""
        Decode the input codes, using [0, 1, ..., code_idx] codebooks.
        Arguments:
            code (Tensor): codes of input image
            code_idx (int): the index of the last selected codebook for decoding
        Returns:
            embeds (Tensor): quantized feature map
        """

        assert code.shape[1:] == self.code_shape
        assert code_idx < code.shape[-1]
        
        B, h, w, _ = code.shape
        
        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)
        if self.shared_codebook:
            embeds = [self.codebooks[0].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        else:
            embeds = [self.codebooks[i].embed(code_slice) for i, code_slice in enumerate(code_slices)]
            
        if decode_type == 'select':
            embeds = embeds[code_idx].view(B, h, w, -1)
        elif decode_type == 'add':
            embeds = torch.cat(embeds[:code_idx+1], dim=-2).sum(-2)
        else:
            raise NotImplementedError(f"{decode_type} is not implemented in partial decoding")

        embeds = self.to_latent_shape(embeds)

        return embeds

    @torch.no_grad()
    def get_soft_codes(self, x, temp=1.0, stochastic=False):

        x = self.to_code_shape(x)

        residual_feature = x.detach().clone()
        soft_code_list = []
        code_list = []

        n_codebooks = self.code_shape[-1]
        for i in range(n_codebooks):
            codebook = self.codebooks[i]
            distances = codebook.compute_distances(residual_feature)
            soft_code = F.softmax(-distances / temp, dim=-1)

            if stochastic:
                soft_code_flat = soft_code.reshape(-1, soft_code.shape[-1])
                code = torch.multinomial(soft_code_flat, 1)
                code = code.reshape(*soft_code.shape[:-1])
            else:
                code = distances.argmin(dim=-1)
            quants = codebook.embed(code)
            residual_feature -= quants

            code_list.append(code.unsqueeze(-1))
            soft_code_list.append(soft_code.unsqueeze(-2))

        code = torch.cat(code_list, dim=-1)
        soft_code = torch.cat(soft_code_list, dim=-2)
        return soft_code, 



def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu", position_emb_depth=128):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim + position_emb_depth, nhead, dropout=dropout)
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor], type="sum"):
        if type == "sum":
            return tensor if pos is None else tensor + pos
        elif type == "concat":
            return tensor if pos is None else torch.cat([tensor,pos], dim=-1)

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        # self attention
        tgt2 = self.norm1(tgt)
        q = k = v = self.with_pos_embed(tgt2, query_pos,'concat')
        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0][:,:,:tgt.shape[-1]]
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt


class mish(nn.Module):
    def __init__(self, ):
        super(mish, self).__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * (torch.tanh(F.softplus(x)))
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransformerEncoderBlock, self).__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        self.transformerEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(out_channels, 8, out_channels),
                                                        num_layers=3)


    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        b = x.size()
        # print(x.shape)
        x = x.view(b[0] * b[1], b[2], b[3])
        # print(x.shape)
        x = x.transpose(1, 0)
        # print(x.shape)
        x = self.transformerEncoder(x)
        # print(x.shape)
        x = x.transpose(1, 0)
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2)
        return x


class RecurrentResidualBlock(nn.Module):
    def __init__(self, channels):
        super(RecurrentResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.gru1 = TransformerEncoderBlock(channels, channels)
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gru2 = TransformerEncoderBlock(channels, channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.gru1(residual.transpose(-1, -2)).transpose(-1, -2)

        return self.gru2(x + residual)


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        ks = 3
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = 512

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)

        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta

        return out


class Dual_Quantize7(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, code_book_opt=False):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        self.codebook_lr = nn.Embedding(dim, n_embed)
        self.codebook_hr = nn.Embedding(dim, n_embed)
        self.codebook_lr.weight.data.uniform_(-1.0/self.n_embed, -1.0/self.n_embed)
        self.codebook_hr.weight.data.copy_(self.codebook_lr.weight)
        self.embed_lr = self.codebook_lr.weight
        self.embed_hr = self.codebook_hr.weight
        self.code_book_opt = code_book_opt
        self.hr_weight = 0.5


    def forward(self, input_hr, input_lr, gt_indice=None, rtn_embed_sort=False, prior=None, prior_weight=0.1, prior_trans=nn.Identity()):
        flatten_hr = input_hr.reshape(-1, self.dim)
        flatten_lr = input_lr.reshape(-1, self.dim)

        
        # upgrade embed_lr
        dist_lc_lr = (
                flatten_lr.pow(2).sum(1, keepdim=True)
                - 2 * flatten_lr @ self.embed_lr
                + self.embed_lr.pow(2).sum(0, keepdim=True)
        )
        dist_lc_hr = (
                flatten_hr.pow(2).sum(1, keepdim=True)
                - 2 * flatten_hr @ self.embed_lr
                + self.embed_lr.pow(2).sum(0, keepdim=True)
        )
        # _, embed_ind_lc_lr = (-dist_lc_lr).max(1)
        p = (1/dist_lc_lr)
        selected_indices = torch.argsort(p,dim=-1,descending=True)[:,:64]
        p = torch.gather(p,1,selected_indices)
        p = p / torch.sum(p,dim=-1, keepdim=True)
        embed_ind_lc_lr= torch.gather(selected_indices, 1, torch.multinomial(p, 1)).squeeze()
        
        embed_onehot_lc_lr = F.one_hot(embed_ind_lc_lr, self.n_embed).type(flatten_lr.dtype)
        embed_ind_lc_lr = embed_ind_lc_lr.view(*input_lr.shape[:-1])

        # _, embed_ind_lc_hr = (-dist_lc_hr).max(1)
        p = (1/dist_lc_hr)
        selected_indices = torch.argsort(p,dim=-1,descending=True)[:,:64]
        p = torch.gather(p,1,selected_indices)
        p = p / torch.sum(p,dim=-1, keepdim=True)
        embed_ind_lc_hr= torch.gather(selected_indices, 1, torch.multinomial(p, 1)).squeeze()
        embed_onehot_lc_hr = F.one_hot(embed_ind_lc_hr, self.n_embed).type(flatten_hr.dtype)
        embed_ind_lc_hr = embed_ind_lc_hr.view(*input_hr.shape[:-1])

        quantize_lc_hr = self.embed_lr_code(embed_ind_lc_hr)
        quantize_lc_lr = self.embed_lr_code(embed_ind_lc_lr)

        if self.training and self.code_book_opt:
            diff_lc_hr = (quantize_lc_hr.detach() - input_hr).pow(2).mean()*0.25 + (quantize_lc_hr - input_hr.detach()).pow(2).mean()
            diff_lc_lr = (quantize_lc_lr.detach() - input_lr).pow(2).mean()*0.25 + (quantize_lc_lr - input_lr.detach()).pow(2).mean()
            if prior is not None:
                diff_lc_hr += (prior_trans(quantize_lc_hr.permute(0, 3, 1, 2)) - prior).pow(2).mean() * prior_weight
                diff_lc_lr += (prior_trans(quantize_lc_lr.permute(0, 3, 1, 2)) - prior).pow(2).mean() * prior_weight
        else:
            diff_lc_hr = (quantize_lc_hr.detach() - input_hr).pow(2).mean()
            diff_lc_lr = (quantize_lc_lr.detach() - input_lr).pow(2).mean()

        quantize_lc_lr = input_lr + (quantize_lc_lr - input_lr).detach()
        quantize_lc_hr = input_hr + (quantize_lc_hr - input_hr).detach()

        # upgrade embed_hr
        dist_hc_lr = (
                flatten_lr.pow(2).sum(1, keepdim=True)
                - 2 * flatten_lr @ self.embed_lr
                + self.embed_lr.pow(2).sum(0, keepdim=True)
        )
        dist_hc_hr = (
                flatten_hr.pow(2).sum(1, keepdim=True)
                - 2 * flatten_hr @ self.embed_lr
                + self.embed_lr.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind_hc_lr = (-dist_hc_lr).max(1)
        embed_onehot_hc_lr = F.one_hot(embed_ind_hc_lr, self.n_embed).type(flatten_lr.dtype)
        embed_ind_hc_lr = embed_ind_hc_lr.view(*input_lr.shape[:-1])

        _, embed_ind_hc_hr = (-dist_hc_hr).max(1)
        embed_onehot_hc_hr = F.one_hot(embed_ind_hc_hr, self.n_embed).type(flatten_hr.dtype)
        embed_ind_hc_hr = embed_ind_hc_hr.view(*input_hr.shape[:-1])

        quantize_hc_hr = self.embed_lr_code(embed_ind_hc_hr)
        quantize_hc_lr = self.embed_lr_code(embed_ind_hc_lr)
        
        if self.training and self.code_book_opt:
            diff_hc_hr = (quantize_hc_hr.detach() - input_hr).pow(2).mean()* 0.25 + (quantize_hc_hr - input_hr.detach()).pow(2).mean()
            diff_hc_lr = (quantize_hc_lr.detach() - input_lr).pow(2).mean()* 0.25 + (quantize_hc_lr - input_lr.detach()).pow(2).mean()
        else:
            diff_hc_hr = (quantize_hc_hr.detach() - input_hr).pow(2).mean()
            diff_hc_lr = (quantize_hc_lr.detach() - input_lr).pow(2).mean()

        quantize_hc_lr = input_lr + (quantize_hc_lr - input_lr).detach()
        quantize_hc_hr = input_hr + (quantize_hc_hr - input_hr).detach()


        if not rtn_embed_sort:
            return quantize_hc_hr, quantize_hc_lr, quantize_lc_hr, quantize_lc_lr, diff_hc_hr, diff_hc_lr, diff_lc_hr, diff_lc_lr, embed_ind_hc_hr, embed_ind_hc_lr, embed_ind_lc_hr, embed_ind_lc_lr, dist_hc_hr, dist_hc_lr, dist_lc_hr, dist_lc_lr
        else:
            return quantize_hc_hr, quantize_hc_lr, quantize_lc_hr, quantize_lc_lr, diff_hc_hr, diff_hc_lr, diff_lc_hr, diff_lc_lr, torch.flip((-dist_hc_hr).sort(1)[1], [1]), torch.flip((-dist_hc_lr).sort(1)[1], [1]), torch.flip((-dist_lc_hr).sort(1)[1], [1]), torch.flip((-dist_lc_lr).sort(1)[1], [1]), dist_hc_hr, dist_hc_lr, dist_lc_hr, dist_lc_lr


    

    def embed_hr_code(self, embed_id):
        return F.embedding(embed_id, self.embed_hr.transpose(0, 1))

    def embed_lr_code(self, embed_id):
        return F.embedding(embed_id, self.embed_lr.transpose(0, 1))