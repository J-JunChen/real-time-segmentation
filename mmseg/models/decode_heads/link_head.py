"""
paper title: LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation.
paper link: https://arxiv.org/pdf/1707.03718
reference code: https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks
"""
import torch
import torch.nn as nn
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)
from mmseg.ops import resize

from ..builder import HEADS
from .decode_head import BaseDecodeHead

class DecoderBlock(nn.Module):
    """ DecoderBlock is "Convolutional modules in decoder-block" in Fig. 3.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 up_ratio=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False):
        super(DecoderBlock, self).__init__()
        self.up_ratio=up_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        # self.conv1 is modified "conv[(1x1), (m, m/4)]" in original paper.
        self.conv1 = ConvModule(
            in_channels,
            in_channels//4,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        # self.conv2 is modified "full-conv[(3x3), (m/4, m/4), *2]" in original paper.
        # Notes: I replace the ConvTranspose2d to Bilinear Interpolation,
        #         however, it need add non-linear function before upsample.
        self.conv2 = ConvModule(
            in_channels//4,
            in_channels//4,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        # self.conv3 is modified "conv[(1x1), (m/4, n)]" in original paper.
        self.conv3 = ConvModule(
            in_channels//4,
            out_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

    def forward(self, x):
        h, w = x[0].shape[-2:]
        output = self.conv1(x)
        output = self.conv2(output)
        if self.up_ratio != 1:
            # [Notes] Decoder Block 1 doesn't upsample
            output = resize(
                input=output,
                size=(h * self.up_ratio, w * self.up_ratio),
                mode='bilinear',
                align_corners=self.align_corners
            )
        output = self.conv3(output)
        return output


@HEADS.register_module()
class LINKHead(BaseDecodeHead):
    """LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation.
        The two difference from original paper is 
            (1) the full-conv in Decoder-Block:
                replace the ConvTranspose2d to Bilinear Interpolation.
            (2) Discard the Classifier layers, including:
                full-conv, conv, full-conv in the last three layers.
    """
    def __init__(self,
                 **kwargs):
        super(LINKHead, self).__init__(**kwargs)

        self.decoder_block_4 = DecoderBlock(512, 256, 2, self.conv_cfg, self.norm_cfg, self.act_cfg, self.align_corners)
        self.decoder_block_3 = DecoderBlock(256, 128, 2, self.conv_cfg, self.norm_cfg, self.act_cfg, self.align_corners)
        self.decoder_block_2 = DecoderBlock(128, 64, 2, self.conv_cfg, self.norm_cfg, self.act_cfg, self.align_corners)
        self.decoder_block_1 = DecoderBlock(64, 64, 1, self.conv_cfg, self.norm_cfg, self.act_cfg, self.align_corners)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        feat_32 = x[3]
        feat_16 = x[2]
        feat_8 = x[1]
        feat_4 = x[0]

        mix_16 = feat_16 + self.decoder_block_4(feat_32)
        mix_8 = feat_8 + self.decoder_block_3(mix_16)
        mix_4 = feat_4 + self.decoder_block_2(mix_8)
        output = self.decoder_block_1(mix_4)

        output = self.cls_seg(output)
        return output
