from urllib.parse import non_hierarchical
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.archs.arch_util import default_init_weights
from basicsr.utils.registry import ARCH_REGISTRY
from archs.util_arch import *
from archs.DualVQVAE1a_arch import DualVQVAE1a_ARCH
from archs.vgg_arch import *

channel_query_dict = {
            64: 256,
            32: 256,
            16: 256,
            8: 256,
            4: 128,
            2: 64,
            1: 32,
}


class MultiScaleEncoder(nn.Module):
    def __init__(self,
                 in_channel,
                 max_depth=3,
                 need_swinlayer=False,
                 output_scale=[4,8]
                 ):
        super().__init__()
        ksz = 3
        self.output_scales = output_scale
        self.in_conv = nn.Conv2d(in_channel, 32, ksz, padding=1)
        self.blocks = nn.ModuleList()

        
        for i in range(max_depth):
            in_ch, out_ch = channel_query_dict[2 ** i], channel_query_dict[2 ** (i + 1)]
            tmp_down_block = [
                nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1),
                ResBlock(out_ch),
                ResBlock(out_ch),
                SwinLayers(input_resolution=(32, 32), embed_dim=out_ch, blk_depth=6, num_heads=8, window_size=8,
                               num_rstb=2) if 2**(i+1) in output_scale and need_swinlayer else nn.Identity()
            ]
            self.blocks.append(nn.Sequential(*tmp_down_block))
    
    def forward(self, input):
        outputs = []
        x = self.in_conv(input)
        for idx, m in enumerate(self.blocks):
            x = m(x)
            if 2 ** (idx + 1) in self.output_scales:
                outputs.append(x)

        return outputs

class DecoderBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.block = []
        self.block += [
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
            ResBlock(out_channel),
            ResBlock(out_channel),
        ]

        self.block = nn.Sequential(*self.block)

    def forward(self, input):
        return self.block(input)



@ARCH_REGISTRY.register()
class DualVQVAE2_ARCH(nn.Module):
    """Example architecture.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        upscale (int): Upsampling factor. Default: 4.
    """

    def __init__(self, in_channel=3, scale_factor=4,quant_params=[[4,512],[8,256]], code_book_opt=True, single_scale_pretrain_weight_path=None, semantic_prior_weight=None):
        super().__init__()
        self.max_depth=3
        self.scale_factor = scale_factor

        self.single_scale_model = DualVQVAE1a_ARCH(in_channel=in_channel, scale_factor=scale_factor,quant_params=[quant_params[-1]])
        if single_scale_pretrain_weight_path is not None:
            self.single_scale_model.load_state_dict(torch.load(single_scale_pretrain_weight_path)["params"])
        for k, v in self.single_scale_model.named_parameters():
            v.requires_grad = False
        
        self.enc_out_scales=[p[0] for p in quant_params]
        quant_params = quant_params[:-1]

        self.lr_low_encoder = []
        self.hr_low_encoder = []
        self.dual_quantizes = []
        self.quantize_scales = []
        for p in quant_params:
            self.quantize_scales.append(p[0])
            self.dual_quantizes.append(Dual_Quantize5(channel_query_dict[p[0]], p[1], code_book_opt=code_book_opt))
            self.lr_low_encoder.append(nn.Sequential(
                SwinLayers(input_resolution=(32, 32), embed_dim=channel_query_dict[p[0]], blk_depth=6, num_heads=8, window_size=8,
                               num_rstb=2))
            )
            self.hr_low_encoder.append(nn.Sequential(
                torch.nn.Identity())
            )
        self.dual_quantizes = nn.ModuleList(self.dual_quantizes)
        self.lr_low_encoder = nn.ModuleList(self.lr_low_encoder)
        self.hr_low_encoder = nn.ModuleList(self.hr_low_encoder)

        self.conv_semantic = nn.Sequential(
                nn.Conv2d(128, 128, 1, 1, 0),
                nn.ReLU(),
                )
        self.vgg_feat_layer = 'pool2'
        if semantic_prior_weight is None:
            print("No semantic prior weight loaded, use the default imagenet pretrained weight.")
            self.vgg_feat_extractor = VGGFeatureExtractor([self.vgg_feat_layer])
        elif os.path.exists(semantic_prior_weight):
            print("Load semantic prior weight from: ", semantic_prior_weight)
            # exit(0)
            self.vgg_feat_extractor = VGGFeatureExtractor([self.vgg_feat_layer], weight_paths=semantic_prior_weight)
        else:
            raise ValueError("Semantic prior weight path not exists: ", semantic_prior_weight)

        self.lr_quantize_convs = []
        self.hr_quantize_convs = []
        for i in range(1, self.max_depth):
            if 2 ** i in self.quantize_scales:
                self.hr_quantize_convs.append(nn.Conv2d(channel_query_dict[2 ** i], channel_query_dict[2 ** i], 1))                    
                self.lr_quantize_convs.append(nn.Conv2d(channel_query_dict[2 ** i], channel_query_dict[2 ** i], 1))

        self.lr_quantize_convs = nn.ModuleList(self.lr_quantize_convs)
        self.hr_quantize_convs = nn.ModuleList(self.hr_quantize_convs)



        self.lr_decoder=[]
        self.hr_decoder=[]
        self.lr_decoder_low_branch=[]
        self.hr_decoder_low_branch=[]
        # concat_c = channel_query_dict[2**(self.max_depth)]
        for i in range(0, self.max_depth-1)[::-1]:
            if 2 ** (i + 1) not in self.quantize_scales:
                decoder_lr = DecoderBlock(channel_query_dict[2 ** (i + 1)], channel_query_dict[2 ** i])
                decoder_hr = DecoderBlock(channel_query_dict[2 ** (i + 1)], channel_query_dict[2 ** i])
                decoder_lr_low_branch = DecoderBlock(channel_query_dict[2 ** (i + 1)], channel_query_dict[2 ** i])
                decoder_hr_low_branch = DecoderBlock(channel_query_dict[2 ** (i + 1)], channel_query_dict[2 ** i])
            else:
                decoder_lr = DecoderBlock(2 * channel_query_dict[2 ** (i + 1)], channel_query_dict[2 ** i])
                decoder_hr = DecoderBlock(2 * channel_query_dict[2 ** (i + 1)], channel_query_dict[2 ** i])
                decoder_lr_low_branch = DecoderBlock(channel_query_dict[2 ** (i + 1)], channel_query_dict[2 ** i])
                decoder_hr_low_branch = DecoderBlock(channel_query_dict[2 ** (i + 1)], channel_query_dict[2 ** i])
                # concat_c = channel_query_dict[2 ** i]
            self.lr_decoder.append(decoder_lr)
            self.hr_decoder.append(decoder_hr)
            self.lr_decoder_low_branch.append(decoder_lr_low_branch)
            self.hr_decoder_low_branch.append(decoder_hr_low_branch)

        self.lr_decoder = nn.ModuleList(self.lr_decoder[::-1])
        self.hr_decoder = nn.ModuleList(self.hr_decoder[::-1])
        self.lr_decoder_low_branch = nn.ModuleList(self.lr_decoder_low_branch[::-1])
        self.hr_decoder_low_branch = nn.ModuleList(self.hr_decoder_low_branch[::-1])

        self.decoder_conv_lr = nn.Conv2d(channel_query_dict[1], in_channel, 3, padding=1)
        self.decoder_conv_hr = nn.Conv2d(channel_query_dict[1], in_channel, 3, padding=1)

    def forward(self, lr, hr=None, rtn_embed_sort=False):
        decs_hc_hr, decs_hc_lr, decs_lc_hr, decs_lc_lr, quant_losses_hc_hr, quant_losses_hc_lr, quant_losses_lc_hr, quant_losses_lc_lr, quant_hc_ids_hr, quant_hc_ids_lr, quant_lc_ids_hr, quant_lc_ids_lr, dists_hc_hr, dists_hc_lr, dists_lc_hr, dists_lc_lr, enc_constraints, encs_hr, encs_lr = self.single_scale_model(lr, hr, multi_scale=True, output_scales=self.enc_out_scales)

        quants_hc_hr = []
        quants_hc_lr = []
        quants_lc_hr = []
        quants_lc_lr = []
        quant_hc_ids_hr = quant_hc_ids_hr
        quant_hc_ids_lr = quant_hc_ids_lr
        quant_lc_ids_hr = quant_lc_ids_hr
        quant_lc_ids_lr = quant_lc_ids_lr
        quant_losses_hc_hr = 0 #quant_losses_hc_hr
        quant_losses_hc_lr = 0 #quant_losses_hc_lr
        quant_losses_lc_hr = 0 #quant_losses_lc_hr
        quant_losses_lc_lr = 0 #quant_losses_lc_lr
        decs_hc_hr = [decs_hc_hr]
        decs_hc_lr = [decs_hc_lr]
        decs_lc_hr = [decs_lc_hr]
        decs_lc_lr = [decs_lc_lr]
        dists_hc_hr = [dists_hc_hr]
        dists_hc_lr = [dists_hc_lr]
        dists_lc_hr = [dists_lc_hr]
        dists_lc_lr = [dists_lc_lr]

        decs_hc_hr_low_branch = []
        decs_hc_lr_low_branch = []
        decs_lc_hr_low_branch = []
        decs_lc_lr_low_branch = []
        enc_constraints=[]




        for i in range(self.max_depth-1)[::-1]:
            if 2 ** (i + 1) in self.quantize_scales:
                quantize_scale_index = self.quantize_scales.index(2 ** (i + 1))
                enc_hr = self.hr_low_encoder[quantize_scale_index](encs_hr[quantize_scale_index])
                enc_lr = self.lr_low_encoder[quantize_scale_index](encs_lr[quantize_scale_index])

                enc_constraints.append(torch.nn.functional.mse_loss(enc_lr, enc_hr.detach()))

                quant_hc_hr, quant_hc_lr, quant_lc_hr, quant_lc_lr, quant_loss_hc_hr, quant_loss_hc_lr, quant_loss_lc_hr, quant_loss_lc_lr, embed_ind_hc_hr, embed_ind_hc_lr, embed_ind_lc_hr, embed_ind_lc_lr, dist_hc_hr, dist_hc_lr, dist_lc_hr, dist_lc_lr = self.dual_quantizes[quantize_scale_index](
                    input_lr=self.lr_quantize_convs[quantize_scale_index](enc_lr).permute(0, 2, 3, 1),
                    input_hr=self.hr_quantize_convs[quantize_scale_index](enc_hr).permute(0, 2, 3, 1),
                    rtn_embed_sort=rtn_embed_sort)
                quant_hc_hr = quant_hc_hr.permute(0, 3, 1, 2)
                quant_hc_lr = quant_hc_lr.permute(0, 3, 1, 2)
                quant_lc_hr = quant_lc_hr.permute(0, 3, 1, 2)
                quant_lc_lr = quant_lc_lr.permute(0, 3, 1, 2)


                quants_hc_hr.append(quant_hc_hr)
                quants_hc_lr.append(quant_hc_lr)
                quants_lc_hr.append(quant_lc_hr)
                quants_lc_lr.append(quant_lc_lr)

                dec_lc_lr = self.lr_decoder[i](torch.cat([decs_lc_lr[-1], quant_lc_lr], dim=1))
                dec_lc_hr = self.lr_decoder[i](torch.cat([decs_lc_hr[-1], quant_lc_hr], dim=1))
                dec_hc_lr = self.hr_decoder[i](torch.cat([decs_hc_lr[-1], quant_hc_lr], dim=1))
                dec_hc_hr = self.hr_decoder[i](torch.cat([decs_hc_hr[-1], quant_hc_hr], dim=1))
                decs_lc_lr.append(dec_lc_lr)
                decs_lc_hr.append(dec_lc_hr)
                decs_hc_lr.append(dec_hc_lr)
                decs_hc_hr.append(dec_hc_hr)

                quant_hc_ids_hr.append(embed_ind_hc_hr)
                quant_hc_ids_lr.append(embed_ind_hc_lr)
                quant_lc_ids_hr.append(embed_ind_lc_hr)
                quant_lc_ids_lr.append(embed_ind_lc_lr)

                # with torch.no_grad():
                vgg_feat = self.vgg_feat_extractor(hr)[self.vgg_feat_layer]
                semantic_quant_hc_hr = self.conv_semantic(quant_hc_hr)
                semantic_quant_hc_lr = self.conv_semantic(quant_hc_lr)
                semantic_loss_hc_hr = F.mse_loss(semantic_quant_hc_hr, vgg_feat)
                semantic_loss_hc_lr = F.mse_loss(semantic_quant_hc_lr, vgg_feat)

                quant_losses_hc_hr += quant_loss_hc_hr + semantic_loss_hc_hr * 0.1
                quant_losses_hc_lr += quant_loss_hc_lr + semantic_loss_hc_lr * 0.1
                quant_losses_lc_hr += quant_loss_lc_hr
                quant_losses_lc_lr += quant_loss_lc_lr
                dists_hc_hr.append(dist_hc_hr)
                dists_hc_lr.append(dist_hc_lr)
                dists_lc_hr.append(dist_lc_hr)
                dists_lc_lr.append(dist_lc_lr)

                dec_lc_lr_low_branch = self.lr_decoder_low_branch[i](quant_lc_lr)
                dec_lc_hr_low_branch = self.lr_decoder_low_branch[i](quant_lc_hr)
                dec_hc_lr_low_branch = self.hr_decoder_low_branch[i](quant_hc_lr)
                dec_hc_hr_low_branch = self.hr_decoder_low_branch[i](quant_hc_hr)
                decs_lc_lr_low_branch.append(dec_lc_lr_low_branch)
                decs_lc_hr_low_branch.append(dec_lc_hr_low_branch)
                decs_hc_lr_low_branch.append(dec_hc_lr_low_branch)
                decs_hc_hr_low_branch.append(dec_hc_hr_low_branch)

            else:
                dec_lc_lr = self.lr_decoder[i](decs_lc_lr[-1])
                dec_lc_hr = self.lr_decoder[i](decs_lc_hr[-1])
                dec_hc_lr = self.hr_decoder[i](decs_hc_lr[-1])
                dec_hc_hr = self.hr_decoder[i](decs_hc_hr[-1])
                decs_lc_lr.append(dec_lc_lr)
                decs_lc_hr.append(dec_lc_hr)
                decs_hc_lr.append(dec_hc_lr)
                decs_hc_hr.append(dec_hc_hr)

                dec_lc_lr_low_branch = self.lr_decoder_low_branch[i](decs_lc_lr_low_branch[-1])
                dec_lc_hr_low_branch = self.lr_decoder_low_branch[i](decs_lc_hr_low_branch[-1])
                dec_hc_lr_low_branch = self.hr_decoder_low_branch[i](decs_hc_lr_low_branch[-1])
                dec_hc_hr_low_branch = self.hr_decoder_low_branch[i](decs_hc_hr_low_branch[-1])
                decs_lc_lr_low_branch.append(dec_lc_lr_low_branch)
                decs_lc_hr_low_branch.append(dec_lc_hr_low_branch)
                decs_hc_lr_low_branch.append(dec_hc_lr_low_branch)
                decs_hc_hr_low_branch.append(dec_hc_hr_low_branch)
         

        if not self.training:
            return self.decoder_conv_hr(decs_hc_hr[-1]), self.decoder_conv_hr(decs_hc_lr[-1]), self.decoder_conv_lr(decs_lc_hr[-1]), self.decoder_conv_lr(decs_lc_lr[-1]), \
                quant_losses_hc_hr, quant_losses_hc_lr, quant_losses_lc_hr, quant_losses_lc_lr,\
                quant_hc_ids_hr[::-1], quant_hc_ids_lr[::-1], quant_lc_ids_hr[::-1], quant_lc_ids_lr[::-1], \
                dist_hc_hr, dist_hc_lr, dist_lc_hr, dist_lc_lr, \
                enc_constraints
        else:
            return self.decoder_conv_hr(decs_hc_hr[-1]), self.decoder_conv_hr(decs_hc_hr_low_branch[-1]), \
                self.decoder_conv_hr(decs_hc_lr[-1]), self.decoder_conv_hr(decs_hc_lr_low_branch[-1]),\
                self.decoder_conv_lr(decs_lc_hr[-1]), self.decoder_conv_lr(decs_lc_hr_low_branch[-1]),\
                self.decoder_conv_lr(decs_lc_lr[-1]), self.decoder_conv_lr(decs_lc_lr_low_branch[-1]),\
                quant_losses_hc_hr, quant_losses_hc_lr, quant_losses_lc_hr, quant_losses_lc_lr,\
                quant_hc_ids_hr[::-1], quant_hc_ids_lr[::-1], quant_lc_ids_hr[::-1], quant_lc_ids_lr[::-1], \
                dist_hc_hr, dist_hc_lr, dist_lc_hr, dist_lc_lr, \
                enc_constraints

