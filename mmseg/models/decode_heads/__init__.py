from .ann_head import ANNHead
from .aspp_head import ASPPHead
from .cc_head import CCHead
from .da_head import DAHead
from .fcn_head import FCNHead
from .gc_head import GCHead
from .lraspp_head import LRASPPHead
from .nl_head import NLHead
from .psp_head import PSPHead
from .erf_head import ERFHead

__all__ = [
    'FCNHead', 'PSPHead', 'ASPPHead', 'NLHead', 'GCHead', 'CCHead',
    'ANNHead', 'DAHead', 'LRASPPHead', 'ERFHead'
]
