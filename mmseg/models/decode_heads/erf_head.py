import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..backbones.erfnet import Non_bottleneck_1d
from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class ERFHead(BaseDecodeHead):
    """ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation
        The only differece from original paper is the upsampling procedure: 
            replace the Deconvolution to Bilinear Interpolation.
    """
    def __init__(self,
                 mid_channels,
                 align_corners=False,
                 **kwargs):
        super(ERFHead, self).__init__(**kwargs)
        # self.in_channels = in_channels
        self.mid_channels = mid_channels
        # self.channels = channels
        # self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners
        # layer 17, change to the 1x1_conv
        self.conv1x1_17 = ConvModule(
            self.in_channels,
            self.mid_channels,
            1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # layer 18-19
        self.layer_18_19 = nn.Sequential(
            Non_bottleneck_1d(
                channels=self.mid_channels,
                dropout_ratio=0,
                dilation=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            Non_bottleneck_1d(
                channels=self.mid_channels,
                dropout_ratio=0,
                dilation=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        )
        # layer 20
        self.conv1x1_20 = ConvModule(
            self.mid_channels,
            self.channels,
            1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # layer 21-22
        self.layer_21_22 = nn.Sequential(
            Non_bottleneck_1d(
                channels=self.channels,
                dropout_ratio=0,
                dilation=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            Non_bottleneck_1d(
                channels=self.channels,
                dropout_ratio=0,
                dilation=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        )

    def forward(self, x):
        """Forward function."""
        h_8, w_8 = x[0].shape[-2:]
        output = self.conv1x1_17(x)
        output = resize(  # the spatial size: 1/8 -> 1/4
            input=output,
            size=(h_8, w_8),
            mode='bilinear',
            align_corners=self.align_corners)
        output = self.layer_18_19(output)
        output = self.conv1x1_20(output)
        output = resize(  # the spatial size: 1/4 -> 1/2
            input=output,
            size=(h_8 * 2, w_8 * 2),
            mode='bilinear',
            align_corners=self.align_corners)
        output = self.layer_21_22(output)

        output = self.cls_seg(output)
        return output