# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-04-04

# Top level imports.
from stereo.models.globals import GLOBAL

# System modules.
import numpy as np

# PyTorch modules.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local module.
from .base_module import BaseModule
from . import common_modules as cm

class Conv3D(BaseModule):
    def __init__(self, inCh, outCh, k=3, s=1, p=1, d=1, 
        normLayer=None, activation=None, bias=False):
        super(Conv3D, self).__init__()

        moduleList = [ nn.Conv3d( inCh, outCh, 
            kernel_size=k, stride=s, padding=p, dilation=d, bias=bias) ]

        if ( normLayer is not None ):
            moduleList.append( normLayer )

        if ( activation is not None):
            moduleList.append( activation )

        self.model = nn.Sequential( *moduleList )

    # Override.
    def initialize(self):
        # Do nothing.
        self.mark_initialized()

    def forward(self, x):
        return self.model(x)

class Conv3D_W(BaseModule):
    def __init__(self, inCh, outCh, 
        k=3, d=1, 
        normLayer=None, activation=None, bias=False):
        super(Conv3D_W, self).__init__()

        p = int( k // 2 * d )

        self.model = Conv3D( inCh, outCh, 
            k=k, s=1, p=p, d=d, 
            normLayer=normLayer, 
            activation=activation, 
            bias=bias)

    # Override.
    def initialize(self):
        # Do nothing.
        self.mark_initialized()

    def forward(self, x):
        return self.model(x)

class FeatureNorm3D(BaseModule):
    def __init__(self, inCh, trs=None):
        super(FeatureNorm3D, self).__init__()
        self.model = nn.BatchNorm3d( inCh, track_running_stats=GLOBAL.torch_batch_normal_track_stat() ) \
            if trs is None \
            else nn.BatchNorm3d( inCh, track_running_stats=trs )

    def initialize(self):
        self.model.weight.data.fill_(1.0)
        self.model.bias.data.zero_()
        self.mark_initialized()

    def forward(self, x):
        return self.model(x)

class Interpolate3D_FixedScale(BaseModule):
    def __init__(self, s, flagNearest=False):
        super(Interpolate3D_FixedScale, self).__init__()

        self.s = s
        self.flagAlignCorners = GLOBAL.torch_align_corners()
        self.mode = 'nearest' if flagNearest else 'trilinear'

    # Override.
    def initialize(self):
        # Do nothing.
        self.mark_initialized()

    def forward(self, x):
        # Get the size of x.
        D, H, W = x.shape[2:5]

        newD = int( D * self.s )
        newH = int( H * self.s )
        newW = int( W * self.s )

        return F.interpolate( x, (newD, newH, newW), \
            mode=self.mode, align_corners=self.flagAlignCorners )

