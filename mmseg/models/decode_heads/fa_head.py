"""
paper title: Real-time Semantic Segmentation with Fast Attention.
paper link: https://arxiv.org/pdf/2007.03815
"""
import torch
import torch.nn as nn
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init, build_activation_layer,
                      build_upsample_layer)
from mmseg.ops import resize

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .link_head import LINKHead


class FastAttentionBlock(nn.Module):
    """ following .utils.SelfAttentionBlock

    Args:
        in_channels (int): Input channels of key, query, value feature.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
        key_query_num_convs (int): Number of convs for key/query projection.
        value_num_convs (int): Number of convs for value projection.
        with_out (bool): Whether use out projection.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """
    def __init__(self, in_channels, channels,
                 out_channels, key_query_num_convs, value_out_num_convs,
                 with_out, conv_cfg, norm_cfg, act_cfg):
        super(FastAttentionBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.key_project = self.build_project(
            in_channels,
            channels,
            num_convs=key_query_num_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.query_project = self.build_project(
            in_channels,
            channels,
            num_convs=key_query_num_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.value_project = self.build_project(
            in_channels,
            channels if with_out else out_channels,
            num_convs=value_out_num_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if with_out:
            self.out_project = self.build_project(
                channels,
                out_channels,
                num_convs=value_out_num_convs,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.out_project = None

        self.init_weights()

    def init_weights(self):
        """Initialize weight of later layer."""
        if self.out_project is not None:
            if not isinstance(self.out_project, ConvModule):
                constant_init(self.out_project, 0)

    def build_project(self, in_channels, channels, num_convs,
                      conv_cfg, norm_cfg, act_cfg):
        """Build projection layer for key/query/value/out."""    
        convs = [nn.Conv2d(in_channels, channels, 1)]
        for _ in range(num_convs - 1):
            convs.append(nn.Conv2d(channels, channels, 1))
        if len(convs) > 1:
            convs = nn.Sequential(*convs)
        else:
            convs = convs[0]
        return convs

    def forward(self, x):
        """Forward function."""
        batch_size, channels, height, width = x.size()

        query = self.query_project(x)
        query = query.reshape(*query.shape[:2], -1)
        query_l2 = torch.norm(query, dim=1).reshape(query.shape[0], -1, query.shape[2]) # l2 norm for query along the channel dimension
        query = query / query_l2
        query = query.permute(0, 2, 1).contiguous()

        key = self.key_project(x)
        key = key.reshape(*key.shape[:2], -1)
        key_l2 = key / torch.norm(key, dim=1).reshape(key.shape[0], -1, key.shape[2]) # l2 norm for key along the channel dimension
        key = key / key_l2

        _value = self.value_project(x)
        value = _value.reshape(*_value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()

        # sim_map = torch.matmul(query, key)
        # if self.matmul_norm:
        #     sim_map = (self.channels**-.5) * sim_map
        # sim_map = F.softmax(sim_map, dim=-1)
        # context = torch.matmul(sim_map, value)

        sim_map = torch.bmm(key, value) 
        # sim_map /= (height * width) # cosine similarity
        
        context = torch.bmm(query, sim_map)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *x.shape[2:])
        if self.out_project is not None:
            context = self.out_project(context)
        # context += _value
        # context += x
        return context


@HEADS.register_module()
class FastAttentionHead(LINKHead):
    def __init__(self,
                 upsample_cfg=dict(type='InterpConv'),
                 **kwargs):
        super(FastAttentionHead, self).__init__(**kwargs)
        self.fa = nn.ModuleList()
        for i in range(6,10): # the number of in_channels is 2^i
            self.fa.append(
                FastAttentionBlock(
                    in_channels=2 ** i,
                    channels=32,  # channels can be changed if you want.
                    # out_channels=2 ** i if i is 6 else 2 ** (i-1), 
                    out_channels=2 ** i, 
                    key_query_num_convs=1,
                    value_out_num_convs=1,
                    with_out=True,
                    conv_cfg=None,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
            
    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        feat_32 = x[3]
        feat_16 = x[2]
        feat_8 = x[1]
        feat_4 = x[0]

        mix_16 = self.decoder_block_4(self.fa[3](feat_32) + feat_32)
        mix_8 = self.decoder_block_3(self.fa[2](feat_16) + mix_16)
        mix_4 = self.decoder_block_2(self.fa[1](feat_8) + mix_8)
        output = self.decoder_block_1(self.fa[0](feat_4) + mix_4)
        # mix_16 = self.fa[2](feat_16) + self.decoder_block_4(feat_32)
        # mix_8 = self.fa[1](feat_8) + self.decoder_block_3(mix_16)
        # mix_4 = self.fa[0](feat_4) + self.decoder_block_2(mix_8)
        # output =  self.decoder_block_1(mix_4)

        output = self.cls_seg(output)
        return output