import torch
import torch.nn as nn
from mmcv.cnn import (ConvModule, constant_init, build_conv_layer,
                      kaiming_init)
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.runner import load_checkpoint

from ..builder import BACKBONES
from .resnet import  ResNet
from mmseg.ops import resize
from mmseg.utils import get_root_logger


class AttentionRefinementModule(nn.Module):
    """Attention Refinement Module
    """
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(AttentionRefinementModule, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.conv3x3_bn_relu = ConvModule(
            in_channels,
            out_channels,
            3,
            stride=1, 
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                out_channels,
                out_channels,
                1,
                stride=1, 
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=dict(type='Sigmoid')
            )
        )
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, x):
        feature = self.conv3x3_bn_relu(x)
        attention = self.channel_attention(feature)
        out = torch.mul(feature, attention)
        return out


class FeatureFusionModule(nn.Module):
    """Feature Fusion Module.
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 reduction=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(FeatureFusionModule, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.conv1x1_bn_relu = ConvModule(
            in_channels, 
            out_channels, 
            1, 
            stride=1, 
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                out_channels, 
                out_channels // reduction, 
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=None,
                act_cfg=self.act_cfg),
            ConvModule(
                out_channels // reduction, 
                out_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=None,
                act_cfg=dict(type='Sigmoid'))
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        feature = self.conv1x1_bn_relu(x)
        attention = self.channel_attention(feature)
        y = torch.mul(feature, attention)
        out = feature + y
        return out


class SpatialPath(nn.Module):
    """Spatial Path
    """
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(SpatialPath, self).__init__()
        inner_channels = 64  # This figure from the original code
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        # # follow the resnet's the first three convolution
        self.downsample = nn.Sequential(
            ConvModule(
                in_channels, 
                inner_channels, 
                7, 
                stride=2, 
                padding=3,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                inner_channels, 
                inner_channels,
                3, 
                stride=2, 
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                inner_channels, 
                inner_channels,
                3, 
                stride=2, 
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        )
        # this conv 1x1 didn't appear in the paper
        self.conv1x1 = ConvModule(
            inner_channels, 
            out_channels, 
            1, 
            stride=1, 
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, x):
        x = self.downsample(x)
        out = self.conv1x1(x)
        return out


class ContextPath(ResNet):
    """Context Path
    """
    def __init__(self,
                 depth,
                 out_channels,
                 act_cfg=dict(type='ReLU'),
                 align_corners=False):
        super(ContextPath, self).__init__(depth)
        inner_channels = 128
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        # follow the original paper of ResNet, when deep_base is False.
        # self.backbone = backbone(deep_stem=deep_stem)  
        self.arm16 = AttentionRefinementModule(256, inner_channels)
        self.arm32 = AttentionRefinementModule(512, inner_channels)
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.conv1x1_gap = ConvModule(
            512,
            inner_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)
        self.conv_head32 = ConvModule(
            inner_channels, 
            out_channels, 
            3, 
            stride=1, 
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)  
        self.conv_head16 = ConvModule(
            inner_channels, 
            out_channels, 
            3, 
            stride=1, 
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
    
    def forward(self, x):
        # x = self.init_layer(x)
        # x = self.maxpool(x)
        # x = self.layer1(x)
        # feat_8 = self.layer2(x)  # 1/8 of spatial size
        # feat_16 = self.layer3(feat_8)  # 1/16 of spatial size
        # feat_32 = self.layer4(feat_16)  # 1/32 of spatial size
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        feat_8 = outs[1]
        feat_16 = outs[2]  # 1/16 of spatial size
        feat_32 = outs[3] # 1/32 of spatial size
        H8, W8 = feat_8.size()[2:]
        H16, W16 = feat_16.size()[2:]
        H32, W32 = feat_32.size()[2:]

        gap_feat = self.gap(feat_32)
        conv_gap_feat = self.conv1x1_gap(gap_feat)
        up_gap_feat = resize(
            conv_gap_feat, 
            size=(H32, W32), 
            mode='bilinear', 
            align_corners=self.align_corners)

        feat_32_arm = self.arm32(feat_32)
        feat_32_sum = feat_32_arm + up_gap_feat
        feat32_up = resize(
            feat_32_sum, 
            size=(H16, W16), 
            mode='bilinear', 
            align_corners=self.align_corners)
        feat32_up = self.conv_head32(feat32_up)

        feat_16_arm = self.arm16(feat_16)
        feat_16_sum = feat_16_arm + feat32_up
        feat16_up = resize(
            feat_16_sum, 
            size=(H8, W8), 
            mode='bilinear', 
            align_corners=self.align_corners)
        feat16_up = self.conv_head16(feat16_up)

        return feat16_up, feat32_up  # x8, x16


@BACKBONES.register_module()
class BiseNetV1(nn.Module):
    """BiseNet: Bilateral Segmentation Network for Real-time Semantic Segmentation.
    
    Args:
        in_channels (int): Number of input image channels. Default: 3.
        out_indices (tuple): Tuple of indices of list
            [higher_res_features, lower_res_features, fusion_output].
            Often set to (0,1,2) to enable aux. heads.
            Default: (0, 1, 2).
    """
    def __init__(self,
                 depth=18,
                 in_channels=3,
                 out_indices=(0, 1, 2),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False):
        super(BiseNetV1, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.sp = SpatialPath(
            in_channels=self.in_channels, 
            out_channels=128,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.cp = ContextPath(
            depth=self.depth,
            out_channels=128,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.ffm = FeatureFusionModule(  # concat: 128+128
            in_channels=256, 
            out_channels=256,
            reduction=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)  
    
    def init_weights(self, pretrained=None):
        # if isinstance(pretrained, str):
        #     logger = get_root_logger()
        #     load_checkpoint(self.cp, pretrained, strict=False, logger=logger)
        # elif pretrained is None:
        #     for m in self.modules():
        #         if isinstance(m, nn.Conv2d):
        #             kaiming_init(m)
        #         elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
        #             constant_init(m, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self.cp, pretrained, strict=False, logger=logger)

    def forward(self, x):
        feat_sp = self.sp(x)
        feat_cp8, feat_cp16 = self.cp(x)
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        outs = [feat_cp8, feat_cp16, feat_fuse]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)



