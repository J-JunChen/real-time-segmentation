import torch
import torch.nn as nn
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule, constant_init,
                      kaiming_init)
from torch.nn.modules.batchnorm import _BatchNorm

from mmseg.models.decode_heads.psp_head import PPM
from mmseg.ops import resize
from ..builder import BACKBONES
from ..utils.inverted_residual import InvertedResidual

