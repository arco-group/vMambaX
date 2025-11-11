# Modified by emulero 10/2025
# Copyright (c) 2025 Jie Mei

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from functools import partial
from src.models.encoders.local_vmamba.region_mamba import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import time
from src.utils.logger import get_logger
from src.models.encoders.vmamba import Backbone_VSSM
from src.models.layers import ContextGatingModule
logger = get_logger()


class RGBXTransformer(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 norm_layer=nn.LayerNorm,
                 depths=[2, 2, 27, 2],  # [2,2,27,2] for vmamba small
                 dims=96,
                 pretrained=None,
                 mlp_ratio=4.0,
                 downsample_version='v1',
                 ape=False,
                 img_size=[512, 512],
                 patch_size=4,
                 drop_path_rate=0.2,
                 cfg=None,
                 **kwargs):
        super().__init__()

        self.ape = ape
        self.cfg = cfg

        self.vssm = Backbone_VSSM(
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
        )

        # Context-Gated Cross-Modal Perception Module
        gate_cfg = cfg.fusion_type 
        gate_reduction = cfg.context_gate.reduction if gate_cfg is not None else 4
        gate_min_channel = cfg.context_gate.min_channel if gate_cfg is not None else 32
        gate_residual = cfg.context_gate.residual if gate_cfg is not None else True
        
        self.modal_fusion = nn.ModuleList(
            ContextGatingModule(
                dim=dims * (2 ** i),
                reduction=gate_reduction,
                min_channel=gate_min_channel,
                residual=gate_residual,
            )
            for i in range(4)
        )

        # DCIM
        self.channel_attn_mamba = nn.ModuleList(
            Region_global_Block(
                outer_dim=dims * (2 ** i), inner_dim=dims * (2 ** i),
                # num_words:4*4 small region
                num_words=16, drop_path=0)for i in range(4)
        )

        self.region_patch = nn.ModuleList(
            Stem(inner_dim=dims * (2 ** i), outer_dim=dims * (2 ** i))
            for i in range(4)
        )

        if self.ape:
            self.patches_resolution = [img_size[0] //
                                       patch_size, img_size[1] // patch_size]
            self.absolute_pos_embed = []
            self.absolute_pos_embed_x = []
            for i_layer in range(len(depths)):
                input_resolution = (self.patches_resolution[0] // (2 ** i_layer),
                                    self.patches_resolution[1] // (2 ** i_layer))
                dim = int(dims * (2 ** i_layer))
                absolute_pos_embed = nn.Parameter(torch.zeros(
                    1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed, std=.02)
                absolute_pos_embed_x = nn.Parameter(torch.zeros(
                    1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed_x, std=.02)

                self.absolute_pos_embed.append(absolute_pos_embed)
                self.absolute_pos_embed_x.append(absolute_pos_embed_x)

    def forward_features(self, x_rgb, x_e):
        """
        x_rgb: B x C x H x W  #ct
        x_e  #pet
        """
        B = x_rgb.shape[0]
        outs_fused = []

        outs_rgb = self.vssm(x_rgb)  # B x C x H x W
        outs_x = self.vssm(x_e)  # B x C x H x W

        for i in range(4):
            if self.ape:
                # this has been discarded
                out_rgb = self.absolute_pos_embed[i].to(
                    outs_rgb[i].device) + outs_rgb[i]
                out_x = self.absolute_pos_embed_x[i].to(
                    outs_x[i].device) + outs_x[i]
            else:
                out_rgb = outs_rgb[i]
                out_x = outs_x[i]

            if self.cfg.use_cgm and self.cfg.use_dcim:
                fused_rgb, fused_x = self.modal_fusion[i](out_rgb, out_x)
                cross_rgb, cross_x, (H_out, W_out), (H_in, W_in) = self.region_patch[i](
                    fused_rgb, fused_x)
                base_rgb = fused_rgb
                base_x = fused_x
                x_fuse = base_rgb+base_x+self.channel_attn_mamba[i](cross_rgb.contiguous(
                ), cross_x.contiguous(), H_out, W_out, H_in, W_in).permute(0, 3, 1, 2).contiguous()

            elif not self.cfg.use_dcim and self.cfg.use_cgm:
                out_rgb,  out_x = self.modal_fusion[i](out_rgb, out_x)
                x_fuse = out_rgb + out_x

            elif self.cfg.use_dcim and not self.cfg.use_cgm:
                cross_rgb = out_rgb
                cross_x = out_x
                cross_rgb, cross_x, (H_out, W_out), (H_in, W_in) = self.region_patch[i](
                    cross_rgb, cross_x)
                x_fuse = out_rgb+out_x+self.channel_attn_mamba[i](cross_rgb.contiguous(), cross_x.contiguous(),
                                                                  H_out, W_out, H_in, W_in).permute(0, 3, 1, 2).contiguous()

            elif not self.cfg.use_dcim and not self.cfg.use_cgm:
                x_fuse = (out_rgb + out_x)

            outs_fused.append(x_fuse)
        return outs_fused

    def forward(self, x_rgb, x_e):
        out = self.forward_features(x_rgb, x_e)
        return out


class vssm_tiny(RGBXTransformer):
    def __init__(self, cfg=None, **kwargs):
        super(vssm_tiny, self).__init__(
            depths=[2, 2, 9, 2],
            dims=96,
            pretrained='pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0,
            cfg=cfg,
        )


class vssm_small(RGBXTransformer):
    def __init__(self, cfg=None, **kwargs):
        super(vssm_small, self).__init__(
            depths=[2, 2, 27, 2],
            dims=96,
            pretrained='pretrained/vmamba/vssmsmall_dp03_ckpt_epoch_238.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.3,
            fuseContextGatingModule_cfg=cfg,
        )


class vssm_base(RGBXTransformer):
    def __init__(self, cfg=None, **kwargs):
        super(vssm_base, self).__init__(
            depths=[2, 2, 27, 2],
            dims=128,
            pretrained='pretrained/vmamba/vssmbase_dp06_ckpt_epoch_241.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            # VMamba-B with droppath 0.5 + no ema. VMamba-B* represents for VMamba-B with droppath 0.6 + ema
            drop_path_rate=0.6,
            cfg=cfg,
        )
