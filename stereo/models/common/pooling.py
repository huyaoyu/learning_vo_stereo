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
from . import common_modules_3d as cm3d

class SpatialPyramidPooling(BaseModule):
    def __init__(self, inCh, levels, 
        maxKFactor=0.5, lastActivation=None, flagNearest=False):
        super(SpatialPyramidPooling, self).__init__()
        
        # Global settings.
        self.flagAlighCorners = GLOBAL.torch_align_corners()
        self.flagReLUInplace  = GLOBAL.torch_relu_inplace()

        self.inCh = inCh

        # Pooling levels.
        self.levels = int(levels)
        assert ( self.levels > 0 )

        # Kernel size factor.
        assert(0 < maxKFactor <= 1)
        self.maxKFactor = maxKFactor

        # Convolusion layers for the pooling levels.
        self.poolingConvs = self.make_pooling_convs()
        
        # Last activation.
        self.lastActivation = lastActivation

        # The interpolation mode.
        self.interMode = 'nearest' if flagNearest else self.get_interpolate_mode()

    # Must be overrided by the inherited class.
    def make_pooling_convs(self):
        raise NotImplementedError()

    # Must be overrided by the inherited class.
    def get_interpolate_mode(self):
        raise NotImplementedError()

    # Must be overrided by the inherited class.
    def get_shape(self, x):
        raise NotImplementedError()

    def check_input_dimensions(self, x):
        shape = self.get_shape(x)
        m = min( *shape )
        return ( m > self.levels )

    @staticmethod
    def get_integer_linspace(start, end, n):
        return np.floor( np.linspace( start, end, n ) ).astype(np.int32)

    def get_kernel_size_list(self, shape):
        kernelSizes = []
        for s in shape:
            kernelSizes.append(
                SpatialPyramidPooling.get_integer_linspace( 2, s * self.maxKFactor, self.levels ) )
        return np.stack( kernelSizes, axis=1 ).tolist()

    # Must be overriede by the inherited class.
    def pool(self, x, kernelSize):
        raise NotImplementedError()

    # Override.
    def initialize(self):
        # Do nothing.
        self.mark_initialized()

    def forward(self, x):
        assert( self.check_input_dimensions(x) )

        # The shape of the input other than the batch size and channel size.
        shape = self.get_shape(x)

        # The kernel size list.
        kernelSizes = self.get_kernel_size_list(shape)

        # Scale factor of each level.
        f = 1.0/self.levels

        spp = x
        for i in range( self.levels ):
            # The kernel size.
            kernelSize = kernelSizes[i]

            # Pooling.
            pooled = self.pool(x, kernelSize)
            pooled = self.poolingConvs[i](pooled)

            # Up-sample.
            upSampled = F.interpolate( 
                pooled, shape, mode=self.interMode, 
                align_corners=self.flagAlighCorners )

            spp = spp + f * upSampled
        
        spp = spp * 0.5

        if ( self.lastActivation is not None ):
            spp = self.lastActivation(spp)

        return spp

class SPP2D(SpatialPyramidPooling):
    def __init__(self, inCh, levels, 
        maxKFactor=0.5, lastActivation=None, flagNearest=False):
        super(SpatialPyramidPooling, self).__init__( 
            inCh, levels, maxKFactor, lastActivation, flagNearest )

    # Override.
    def make_pooling_convs(self):
        poolingConvs = nn.ModuleList()
        for i in range(self.levels):
            poolingConvs.append( 
                cm.Conv_W(self.inCh, self.inCh, k=1, 
                normLayer=cm.FeatureNormalization(inCh),
                activation=nn.ReLU(inplace=self.flagReLUInplace) ) )
        return poolingConvs

    # Override.
    def get_interpolate_mode(self):
        return 'bilinear'

    # Override.
    def get_shape(self, x):
        return x.shape[2:4]
    
    # Override.
    def pool(self, x, kernelSize):
        return F.avg_pool2d( x, kernelSize, kernelSize, padding=0 )
        
class SPP3D(SpatialPyramidPooling):
    def __init__(self, inCh, levels, 
        maxKFactor=0.5, lastActivation=None, flagNearest=False):
        super(SpatialPyramidPooling, self).__init__( 
            inCh, levels, maxKFactor, lastActivation, flagNearest )

    # Override.
    def make_pooling_convs(self):
        poolingConvs = nn.ModuleList()
        for i in range(self.levels):
            poolingConvs.append( 
                cm3d.Conv3D_W(self.inCh, self.inCh, k=1, 
                normLayer=cm3d.FeatureNorm3D(inCh),
                activation=nn.ReLU(inplace=self.flagReLUInplace) ) )
        return poolingConvs

    # Override.
    def get_interpolate_mode(self):
        return 'trilinear'

    # Override.
    def get_shape(self, x):
        return x.shape[2:5]
    
    # Override.
    def pool(self, x, kernelSize):
        return F.avg_pool3d( x, kernelSize, kernelSize, padding=0 )