# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-21

# Top level imports.
from stereo.models.globals import GLOBAL

# System packages.
import numpy as np

# PyTorch packages.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local packages.
from stereo.models.common.base_module import BaseModule
from stereo.models.common import common_modules as cm
from stereo.models.common import common_modules_3d as cm3d
from stereo.models.common.pooling import SPP3D

class ProjFeat3D(BaseModule):
    '''
    Turn 3d projection into 2d projection
    '''
    def __init__(self, inCh, outCh, stride):
        super(ProjFeat3D, self).__init__()

        self.stride = stride
        self.conv = cm.Conv( inCh, outCh, 
            k=1, s=stride[:2], p=0, 
            normLayer=cm.FeatureNormalization(outCh), 
            activation=None )

    # Override
    def initialize(self):
        super(ProjFeat3D, self).initialize()

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.conv( x.view(B, C, D, H*W) )
        return x.view( B, -1, D//self.stride[0], H, W)

class SepConv3DBlock(BaseModule):
    '''
    ResNet like 3D convolusion block.
    '''
    def __init__(self, inCh, outCh, stride=(1,1,1)):
        super(SepConv3DBlock, self).__init__()

        self.flagReLUInplace = GLOBAL.torch_relu_inplace()

        if inCh == outCh and stride==(1,1,1):
            self.downsample = None
        else:
            self.downsample = ProjFeat3D(inCh, outCh, stride)

        self.conv0 = cm3d.Conv3D( inCh, outCh, s=stride, 
            normLayer=cm3d.FeatureNorm3D(outCh), 
            activation=nn.ReLU(inplace=self.flagReLUInplace) )
        self.conv1 = cm3d.Conv3D_W( outCh, outCh, 
            normLayer=cm3d.FeatureNorm3D(outCh), 
            activation=nn.ReLU(inplace=self.flagReLUInplace) )
    
    # Override.
    def initialize(self):
        super(SepConv3DBlock, self).initialize()

    def forward(self,x):
        # Convolusion branch.
        out = self.conv0(x)

        # Short-cut branch.
        if self.downsample:
            x = self.downsample(x)
        
        return x + self.conv1(out)

class DecoderBlock(BaseModule):
    def __init__(self, 
        nConvs, inCh, interCh, outCh,
        baseStride=(1,1,1), nStrides=1, 
        outputUpSampledFeat=False, pooling=False ):
        super(DecoderBlock, self).__init__()

        # Get the global settings.
        self.flagAlignCorners = GLOBAL.torch_align_corners()
        self.flagReLUInplace  = GLOBAL.torch_relu_inplace()

        # Prepare the list of strides.
        assert( nConvs >= nStrides )
        strideList = [baseStride] * nStrides + [(1,1,1)] * (nConvs - nStrides)

        # Create the the convolusion layers.
        convs = [ SepConv3DBlock( inCh, interCh, stride=strideList[0] ) ]
        for i in range(1, nConvs):
            convs.append( SepConv3DBlock( interCh, interCh, stride=strideList[i] ) )
        self.entryConvs = nn.Sequential(*convs)
        self.append_init_here( self.entryConvs )

        # Classification layer.
        self.classify = nn.Sequential(
            cm3d.Conv3D_W( interCh, interCh, 
                normLayer=cm3d.FeatureNorm3D(interCh), 
                activation=nn.ReLU(inplace=self.flagReLUInplace) ), 
            cm3d.Conv3D_W(interCh, outCh, bias=True) )
        self.append_init_here(self.classify)

        # Feature up-sample setting.
        self.featUpSampler = None
        if outputUpSampledFeat:
            self.featUpSampler = nn.Sequential(
                cm3d.Interpolate3D_FixedScale(2),
                cm3d.Conv3D_W( interCh, interCh//2, 
                    normLayer=cm3d.FeatureNorm3D(interCh//2), 
                    activation=nn.ReLU(inplace=self.flagReLUInplace) ) )
            self.append_init_here(self.featUpSampler)

        # Pooling.
        if pooling:
            self.spp = pooling.SPP3D( interCh, levels=4 )
            self.append_init_here(self.spp)
        else:
            self.spp = None

    # Override.
    def initialize(self):
        if ( self.is_initialized() ):
            print('Warning: Already initialized!')
            return

        for m in self.toBeInitializedHere:
            if ( not m.is_initialized() ):
                for mm in m.modules():
                    if isinstance(mm, nn.Conv3d):
                        n = np.prod(mm.kernel_size) * mm.out_channels
                        nn.init.normal_(mm.weight, 0, np.sqrt( 2. / n ))
                        if hasattr(mm.bias, 'data'):
                            nn.init.zeros_(mm.bias)
                
                m.mark_initialized()

        self.mark_initialized()

    def forward(self, x):
        # Entry.
        x = self.entryConvs(x)

        # Pooling.
        if ( self.spp is not None ):
            x = self.spp(x)

        # Classification.
        cost = self.classify(x)

        # Up-sample the feature.
        if ( self.featUpSampler is not None ):
            x = self.featUpSampler(x)

        return cost, x