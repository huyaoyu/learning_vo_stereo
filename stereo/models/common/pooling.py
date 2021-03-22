# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-21

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

class SpatialPyramidPooling(BaseModule):
    def __init__(self, inCh, levels, maxKFactor=0.5, lastActivation=None, flagNearest=False):
        super(SpatialPyramidPooling, self).__init__()
        
        self.flagAlighCorners = GLOBAL.torch_align_corners()
        self.flagReLUInplace  = GLOBAL.torch_relu_inplace()

        self.levels = int(levels)
        assert ( self.levels > 0 )

        assert(0 < maxKFactor <= 1)
        self.maxKFactor = maxKFactor

        self.poolingConvs = nn.ModuleList()

        for i in range(self.levels):
            self.poolingConvs.append( 
                cm.Conv_W(inCh, inCh, k=1, 
                normLayer=cm.FeatureNormalization(inCh),
                activation=nn.ReLU(inplace=self.flagReLUInplace) ) )
        
        self.lastActivation = lastActivation

        self.interMode = 'nearest' if flagNearest else 'bilinear'

    def initialize(self):
        # Do nothing.
        self.mark_initialized()

    def forward(self, x):
        # Get the size.
        h, w = x.shape[2:4]
        d = min(h, w)

        assert( d > 4 )

        # The kernel size.
        kernelFloatH = np.linspace( 2, h * self.maxKFactor, self.levels )
        kernelFloatW = np.linspace( 2, w * self.maxKFactor, self.levels )

        spp = x
        f = 1.0/self.levels

        for i in range( self.levels ):
            # The kernel size.
            kH = int( np.floor( kernelFloatH[i] ) )
            kW = int( np.floor( kernelFloatW[i] ) )

            pooled = F.avg_pool2d( x, (kH, kW), ( kH, kW ), padding=0 )
            pooled = self.poolingConvs[i](pooled)

            # Up-sample.
            upSampled = F.interpolate( 
                pooled, (h, w), mode=self.interMode, align_corners=self.flagAlighCorners )

            spp = spp + f * upSampled
        
        spp = spp * 0.5

        if ( self.lastActivation is not None ):
            spp = self.lastActivation(spp)

        return spp