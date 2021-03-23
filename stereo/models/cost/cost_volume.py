# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-21

# Top level imports.
from stereo.models.globals import GLOBAL

# System packages.
# import numpy as np

# PyTorch packages.
import torch
import torch.nn as nn
# import torch.nn.functional as F

# Local packages.
from stereo.models.common.base_module import BaseModule
from stereo.models.register import ( COST_VOL, register )

class CostVolume3D(BaseModule):
    def __init__(self, refIsRight=False, freeze=False):
        super(CostVolume3D, self).__init__(freeze=freeze)

        self.refIsRight = refIsRight

    def forward(self, featL, featR):
        raise NotImplementedError()

@register(COST_VOL)
class CVConcat(CostVolume3D):
    @classmethod
    def get_default_init_args(cls):
        return dict(
            type=cls.__name__,
            refIsRight=False, 
            freeze=False)

    def __init__(self, refIsRight=False, freeze=False):
        super(CVConcat, self).__init__(refIsRight, freeze)

    def initialize(self):
        # Do nothing.
        self.mark_initialized()

    def forward(self, featL, featR):
        raise NotImplementedError()

@register(COST_VOL)
class CVDiff(CostVolume3D):
    @classmethod
    def get_default_init_args(cls):
        return dict(
            type=cls.__name__,
            refIsRight=False, 
            freeze=False)

    def __init__(self, refIsRight=False, freeze=False):
        super(CVDiff, self).__init__(refIsRight, freeze)

    def initialize(self):
        # Do nothing.
        self.mark_initialized()

    def forward(self, featL, featR, maxDisp):
        B, C, H, W = featL.shape

        cost = torch.zeros( ( B, C, maxDisp, H, W ), device=featL.device )

        for i in range(maxDisp):
            featA = featL[ :, :, :, i:W   ]
            featB = featR[ :, :, :,  :W-i ]
            # concat
            if ( self.refIsRight ):
                cost[:, :, i, :,  :W-i] = torch.abs(featB-featA)
            else:
                cost[:, :, i, :, i:   ] = torch.abs(featA-featB)

        return cost.contiguous()
