# Copyright (c) OpenMMLab. All rights reserved.
import math
from turtle import forward
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
from ..utils import CSPLayerWithTwoConv

from mmcv.cnn import ACTIVATION_LAYERS

# @ACTIVATION_LAYERS.register_module()
# class SiLU(nn.Module):

#     def __init__(self, inplace=False):
#         super(SiLU, self).__init__()
#         self.act = nn.SiLU(inplace)

#     def forward(self, x):
#         return self.act(x)
ACTIVATION_LAYERS.register_module(module=nn.SiLU, name='SiLU')


class SPPFBottleneck(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='SiLU'),
                 init_cfg=None):
        super().__init__(init_cfg)
        mid_channels = in_channels // 2
        self.conv1 = ConvModule(in_channels,
                                mid_channels,
                                1,
                                stride=1,
                                padding=0,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg)
        conv2_in = mid_channels * 4
        conv2_out = out_channels
        self.conv2 = ConvModule(conv2_in, conv2_out, 1, stride=1,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        pool_pad = kernel_size // 2
        self.pooling = nn.MaxPool2d(kernel_size, stride=1, padding=pool_pad)

    def forward(self, x):
        x = self.conv1(x)
        # with warnings.catch_warning():
        # warning.simplefilter('ignore')
        y1 = self.pooling(x)
        y2 = self.pooling(y1)
        y3 = torch.cat([x, y1, y2, self.pooling(y2)], dim=1)
        out = self.conv2(y3)
        return out

@BACKBONES.register_module()
class CSPDarknetV8(BaseModule):
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, None, 3, True, True]],
    }
    def __init__(self,
                 arch='P5',
                 deepen_factor=1.0,
                 widen_factor=1.0,
                 last_stage_out_channels = 1024,
                 out_indices=(2, 3, 4),
                 frozen_stages=-1,
                 use_depthwise=False,
                 arch_ovewrite=None,
                 sppf_kernal_size=5,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='SiLU',inplace=True),
                 norm_eval=False,
                 init_cfg=dict(type='Kaiming',
                               layer='Conv2d',
                               a=0,
                               distribution='normal',
                               mode='fan_out',
                               nonlinearity='relu')):
        super().__init__(init_cfg)
        self.arch_settings[arch][-1][1] = last_stage_out_channels
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('frozen_stages must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        self.stem_YOLOV5 = ConvModule(  # 替代原本的Focus
            3,  # 替代focus卷积层的in_channel
            int(64 * widen_factor),  # 替代focus卷积层的out_channel
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.layers = ['stem_YOLOV8']

        for i, (in_channels, out_channels, num_blocks, add_identity, use_sppf) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = []
            conv_layer = conv(in_channels,
                              out_channels,
                              3,
                              stride=2,
                              padding=1,
                              conv_cfg=conv_cfg,
                              norm_cfg=norm_cfg,
                              act_cfg=act_cfg)
            stage.append(conv_layer)
            csp_layer = CSPLayerWithTwoConv(out_channels,
                                 out_channels,
                                 num_blocks=num_blocks,
                                 add_identity=add_identity,
                                 conv_cfg=conv_cfg,
                                 norm_cfg=norm_cfg,
                                 act_cfg=act_cfg)
            stage.append(csp_layer)
            if use_sppf:
                sppf = SPPFBottleneck(out_channels,
                                      out_channels,
                                      kernel_size=sppf_kernal_size,
                                      conv_cfg=conv_cfg,
                                      norm_cfg=norm_cfg,
                                      act_cfg=act_cfg)
                stage.append(sppf)
            self.add_module(f'stage{i + 1}_YOLOV8', nn.Sequential(*stage))
            self.layers.append(f'stage{i + 1}_YOLOV8')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(CSPDarknetV8, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    