import torch.nn as nn
from .simam import SimAM
from .cbam import CBAM
from .gam import GAM

def get_att_module(attention_type, channels):
    if attention_type == 'cbam':
        return CBAM(channels)
    elif attention_type == 'gam':
        return GAM(channels)
    elif attention_type == 'simam':
        return SimAM()
    else:
        return nn.Identity()