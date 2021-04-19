# -*- coding: future_fstrings -*-

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
from stereo.models.common import pooling

from stereo.models.register import ( FEAT_EXT, register )

from .fe_base import FEBase

class HalfSizeExtractor(nn.Module):
    def __init__(self, nLayers=3, intermediateChannels=8, ):
        super(HalfSizeExtractor, self).__init__()

        self.flagReLUInplace = GLOBAL.torch_relu_inplace()

        modelList = [ 
            cm.Conv_Half( 3, intermediateChannels, 
                normLayer=cm.FeatureNormalization(intermediateChannels),
                activation=nn.ReLU(inplace=self.flagReLUInplace) ) ]

        for i in range(nLayers):
            if ( i == nLayers - 1):
                modelList.append(
                    cm.Conv_W( intermediateChannels, intermediateChannels, 
                    normLayer=cm.FeatureNormalization(intermediateChannels),
                    activation=nn.ReLU(inplace=self.flagReLUInplace) ) )
            else:
                modelList.append(
                    cm.Conv_W( intermediateChannels, intermediateChannels, 
                    normLayer=None,
                    activation=nn.ReLU(inplace=self.flagReLUInplace) ) )

        self.model = nn.Sequential( *modelList )

    def forward(self, x):
        return self.model(x)

# Encoder-decoder with skip connections.
# e0------------------------------> d0-> f0 -> out0
#   \                             /
#    \                           u0
#     \                         /
#     e1 --------------------> d1 -----> f1 -> out1
#       \                    /
#        \                  u1
#         \                /
#         e2 -> middle -> d2 ----------> f2 -> out2
@register(FEAT_EXT)
class UNetOneHalf(FEBase):

    CH_IDX_E_IN  = 0
    CH_IDX_E_OUT = 1
    CH_IDX_D_OUT = 2
    CH_IDX_U_OUT = 3
    CH_IDX_F_OUT = 4

    @classmethod
    def get_default_init_args(cls):
        return dict(
            type=cls.__name__,
            edChannels = [
                [  8, 16, 16, 16, 16 ],
                [ 16, 16, 16, 16, 16 ],
                [ 16, 16, 16, 16, 16 ],
                [ 16, 16, 16, 16, 16 ]
            ],
            freeze=False )

    def __init__(self, 
        edChannels = [
            [  3, 16, 16, 16, 16 ],
            [ 16, 16, 16, 16, 16 ],
            [ 16, 16, 16, 16, 16 ],
            [ 16, 16, 16, 16, 16 ]
        ],
        freeze=False):
        '''
        edChannels (list of lists): Channel specification for every layer. 
            [ eIn, eOut, dOut, up, out ]
        '''
        super(UNetOneHalf, self).__init__(
            levels=[2, 4, 8, 16],
            freeze=freeze)

        N = len(levels)
        assert( N == len(edChannels) ), \
            f'Wrong level and channel specification. levels = {levels}, edChannels = {edChannels}'

        self.flagTS = GLOBAL.torch_batch_normal_track_stat()
        self.flagReLUInplace = GLOBAL.torch_relu_inplace()

        # Encoders.
        self.encoders = nn.ModuleList()
        for i in range(N):
            inCh  = edChannels[i][CH_IDX_E_IN]
            outCh = edChannels[i][CH_IDX_E_OUT]
            self.encoders.append(
                cm.ResidualBlock( inCh, outCh, 
                    stride=2, downsample=
                        cm.Conv( inCh, outCh, k=1, s=2, p=0, 
                            normLayer=cm.FeatureNormalization( outCh ) )
                )
            )

        # Decoders.
        self.decoders = nn.ModuleList()
        for i in range(N-1, -1, -1):
            if ( i == N-1 ):
                inCh = edChannels[i][CH_IDX_E_OUT]
            else:
                inCh = edChannels[i][CH_IDX_E_OUT] \
                     + edChannels[i-1][CH_IDX_U_OUT]

            outCh = edChannels[i][CH_IDX_D_OUT]

            self.decoders.append(
                cm.Conv_W( inCh, outCh, 
                    normLayer=cm.FeatureNormalization( outCh ),
                    activation=nn.ReLU(inplace=self.flagReLUInplace) ) )

        self.decoders = self.decoders[::-1]

        # Up-feature layers.
        self.ups = nn.ModuleList()
        for i in range( N-1, 0, -1 ):
            inCh  = edChannels[i][CH_IDX_D_OUT]
            outCh = edChannels[i][CH_IDX_U_OUT]
            self.ups.append(
                cm.Interpolate2D_FixedScale(2),
                cm.Conv_W( inCh, outCh, 
                    normLayer=cm.FeatureNormalization( outCh ),
                    activation=nn.ReLU(inplace=self.flagReLUInplace) ) )

        self.ups = self.ups[::-1]

        # Finale layers.
        self.finals = nn.ModuleList()
        for i in range( N-1, -1, -1 ):
            inCh  = edChannels[i][CH_IDX_U_OUT]
            outCh = edChannels[i][CH_IDX_F_OUT]
            self.finals.append( 
                cm.Conv_W( inCh, outCh, 
                    normLayer=cm.FeatureNormalization( outCh ),
                    activation=nn.ReLU(inplace=self.flagReLUInplace) ) )

        self.finals = self.finals[::-1]

        # Middle.
        self.middle = cm.ResidualBlock(
            edChannels[-1][CH_IDX_E_OUT], edChannels[-1][CH_IDX_E_OUT], 
                lastActivation=nn.ReLU(inplace=self.flagReLUInplace) )
    
    def initialize(self):
        # Do nothing.
        self.mark_initialized()

    def forward(self, x):
        N = len( self.levels )
        
        # Encode.
        encodes = []
        e = x
        for m in self.encoders:
            e = m(e)
            encodes.append( e )

        # Middle.
        encodes[-1] = self.middle( encodes[-1] )

        # Decode.
        dLast = self.decoders[-1]( encodes[-1] )
        outputs = [ self.finals[-1](dLast) ]
        for i in range(N-2, -1, -1):
            u = self.ups[i](dLast)
            t = torch.cat( ( encodes[i], u ), dim=1 )
            dLast = self.decoders[i](t)
            outputs.append( self.finals[i](dLast) )

        return outputs[::-1]

