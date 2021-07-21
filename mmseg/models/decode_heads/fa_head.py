"""
paper title: Real-time Semantic Segmentation with Fast Attention.
paper link: https://arxiv.org/pdf/2007.03815
reference code: https://github.com/feinanshan/FANet/blob/master/Testing/models/fanet/fanet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, build_upsample_layer)
from mmseg.ops import resize

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .link_head import LINKHead


class FastAttentionBlock(nn.Module):
    """ following .utils.SelfAttentionBlock

    Args:
        in_channels (int): Input channels of key, query, value feature.
        channels (int): Output channels of key/query transform.
        smf_channels (int): Output channels.
        key_query_num_convs (int): Number of convs for key/query projection.
        value_num_convs (int): Number of convs for value projection.
        with_out (bool): Whether use out projection.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """
    def __init__(self, in_channels, channels, up_channels,
                 smf_channels, up_flag, smf_flag,
                 conv_cfg, norm_cfg, act_cfg, align_corners):
        super(FastAttentionBlock, self).__init__()
        self.up_flag = up_flag
        self.smf_flag = smf_flag
        self.align_corners = align_corners
        self.key_project = ConvModule(
            in_channels,
            channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.query_project = ConvModule(
            in_channels,
            channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.value_project = ConvModule(
            in_channels,
            in_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.out_project = ConvModule(
            in_channels,
            in_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if self.up_flag:
            self.up = ConvModule(
                in_channels,
                up_channels,
                1,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        if self.smf_flag:
            self.smooth = ConvModule(
                in_channels,
                smf_channels,
                3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
    
    def upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        '''
        _, _, H, W = y.size()
        x = resize(
            x,
            size=(H, W),
            mode='bilinear',
            align_corners=self.align_corners)
        return x + y

    def forward(self, x, up_feat_in):
        """Forward function."""
        batch_size, channels, height, width = x.size()

        query = self.query_project(x)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()
        query = F.normalize(query, p=2, dim=2, eps=1e-12) # l2 norm for query along the channel dimension

        key = self.key_project(x)
        key = key.reshape(*key.shape[:2], -1)
        key = F.normalize(key, p=2, dim=1, eps=1e-12) # l2 norm for key along the channel dimension

        value = self.value_project(x)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()

        sim_map = torch.matmul(key, value) 
        # sim_map /= (height * width) # cosine similarity
        context = torch.matmul(query, sim_map)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *x.shape[2:])

        context = self.out_project(context)
        fuse_feature = context + x

        if self.up_flag and self.smf_flag:
            if up_feat_in is not None:
                fuse_feature = self.upsample_add(up_feat_in, fuse_feature)
            up_feature = self.up(fuse_feature)
            smooth_feature = self.smooth(fuse_feature)
            return up_feature, smooth_feature
        
        if self.up_flag and not self.smf_flag:
            if up_feat_in is not None:
                fuse_feature = self.upsample_add(up_feat_in, fuse_feature)
            up_feature = self.up(fuse_feature)
            return up_feature
        
        if not self.up_flag and self.smf_flag:
            if up_feat_in is not None:
                fuse_feature = self.upsample_add(up_feat_in, fuse_feature)
            smooth_feature = self.smooth(fuse_feature)
            return smooth_feature


@HEADS.register_module()
class FastAttentionHead(BaseDecodeHead):
    def __init__(self,
                 **kwargs):
        super(FastAttentionHead, self).__init__(**kwargs)
        self.fa = nn.ModuleList()
        for i in range(6,10): # the number of in_channels is 2^i
            self.fa.append(
                FastAttentionBlock(
                    in_channels=2 ** i,
                    channels=32,  # channels can be changed if you want.
                    up_channels=2 ** i if i is 6 else 2 ** (i-1),
                    # smf_channels=2 ** i if i is 6 else 2 ** (i-1), 
                    smf_channels=128, 
                    up_flag=False if i is 6 else True ,
                    smf_flag=True if i%2 == 0 else False ,
                    conv_cfg=None,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    align_corners=self.align_corners
                )
            )
    
    def upsample_cat(self, x, y):
        '''Upsample and concatenate feature maps.
        '''
        _, _, H, W = y.size()
        x = resize(
            x,
            size=(H, W),
            mode='bilinear',
            align_corners=self.align_corners
        )
        return torch.cat([x, y], dim=1)
            
    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        feat_32 = x[3]
        feat_16 = x[2]
        feat_8 = x[1]
        feat_4 = x[0]

        # up_feat_32, smf_feat_32 = self.fa[3](feat_32, None, True, True)
        # up_feat_32 = self.fa[3](feat_32, None, True, False)
        # up_feat_16, smf_feat_16 = self.fa[2](feat_16, up_feat_32 , True, True)
        # up_feat_8 = self.fa[1](feat_8, up_feat_16, True, False)
        # smf_feat_4 = self.fa[0](feat_4, up_feat_8, False, True)
        up_feat_32 = self.fa[3](feat_32, None)
        up_feat_16, smf_feat_16 = self.fa[2](feat_16, up_feat_32)
        up_feat_8 = self.fa[1](feat_8, up_feat_16)
        smf_feat_4 = self.fa[0](feat_4, up_feat_8)

        output = self.upsample_cat(smf_feat_16, smf_feat_4)

        output = self.cls_seg(output)
        return output