"""  
paper title: ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation
reference code: https://github.com/Eromera/erfnet_pytorch
"""
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)
from mmcv.utils.parrots_wrapper import _BatchNorm

from ..builder import BACKBONES
from .base_backbone import BaseBackbone

class DownsamplerBlock(nn.Module):
    """Downsampler Block
    """
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(DownsamplerBlock, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.conv = build_conv_layer(
            self.conv_cfg,
            in_channels,
            out_channels-in_channels,
            kernel_size=3,
            stride=2, 
            padding=1,
            bias=True
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = torch.cat([self.conv(x), self.pool(x)], 1)
        output = self.bn(output)
        return self.relu(output)


class Non_bottleneck_1d(nn.Module):
    """Non-bt-1D
    """
    def __init__(self, 
                 channels,
                 dropout_ratio=0.3, 
                 dilation=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):        
        super(Non_bottleneck_1d, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.dilation = dilation
        self.conv3x1_1 = build_conv_layer(
            self.conv_cfg,
            channels,
            channels,
            kernel_size=(3, 1),
            stride=1, 
            padding=(1, 0),
            bias=True)
        self.conv1x3_1 = build_conv_layer(
            self.conv_cfg,
            channels,
            channels,
            kernel_size=(1, 3),
            stride=1, 
            padding=(0, 1),
            bias=True)
        self.bn1 = build_norm_layer(self.norm_cfg, channels)[1]
        self.conv3x1_2 = build_conv_layer(
            self.conv_cfg,
            channels,
            channels,
            kernel_size=(3, 1),
            stride=1, 
            padding=(self.dilation, 0),
            bias=True,
            dilation=(self.dilation, 1))
        self.conv1x3_2 = build_conv_layer(
            self.conv_cfg,
            channels,
            channels,
            kernel_size=(1, 3),
            stride=1, 
            padding=(0, self.dilation),
            bias=True,
            dilation=(1, self.dilation))
        self.bn2 = build_norm_layer(self.norm_cfg, channels)[1]
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        output = self.conv3x1_1(x)
        output = self.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv3x1_2(output)
        output = self.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return self.relu(output+x)    #+input = identity (residual connection)


@BACKBONES.register_module()
class ERFNet(BaseBackbone):
    """The Encoder part of ERFNet
    """
    def __init__(self,
                 in_channels=3,
                 mid_channels=64,
                 encoder_out_channels=128,
                 out_indices=(0, ),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(ERFNet, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.encoder_out_channels = encoder_out_channels
        self.out_indices = out_indices
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.initial_block = DownsamplerBlock(self.in_channels, 16)
        self.layers = nn.ModuleList()
        # layer 2
        self.layers.append(
            DownsamplerBlock(
                16, self.mid_channels))
        # layer 3-7
        for i in range(3, 7+1):
            self.layers.append(
                Non_bottleneck_1d(
                    self.mid_channels, 0.03))
        # layer 8
        self.layers.append(
            DownsamplerBlock(
                self.mid_channels, 
                self.encoder_out_channels))
        # layer 9-16
        for i in range(0, 2): # 2 times
            for j in range(0, 4): # dilation = 2^(j+1)
                self.layers.append(
                    Non_bottleneck_1d(
                        self.encoder_out_channels, 0.3, dilation=2**(j+1)))

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        
    def forward(self, x):
        x = self.initial_block(x)
        outs = []
        for layer in self.layers:
            x = layer(x)
        outs.append(x)
        
        return outs[0]
        