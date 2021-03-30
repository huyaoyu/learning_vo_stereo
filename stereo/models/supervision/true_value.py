# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-20

import numpy as np

import torch
import torch.nn.functional as F

from stereo.models.globals import GLOBAL as modelGLOBAL
from .classification_label import float_2_cls_label

from stereo.models.register import ( TRUE_GEN, register )

# TT stands for true value terms
class TT(object):
    DISP      = 'dispL'
    MASK      = 'maskL'
    OCC       = 'occL'
    DISP_LIST = 'dispLs'
    MAKS_LIST = 'maskLs'
    OCC_LIST  = 'occLs'
    UNCT_LIST = 'uncertaintyLs'

class TrueValueGenerator(object):
    def __init__(self, trueDispMax, flagIntNearest=False):
        super(TrueValueGenerator, self).__init__()

        self.trueDispMax      = trueDispMax
        self.flagAlignCorners = modelGLOBAL.torch_align_corners()
        self.flagIntNearest   = flagIntNearest

    def interpolate(self, t, shape):
        if ( self.flagIntNearest ):
            return F.interpolate( t, shape, 
                mode="nearest" )
        else:
            return F.interpolate( t, shape, 
                mode="bilinear", align_corners=self.flagAlignCorners )

    def interpolate_bool(self, t, shape):
        t = t.to(torch.float32)
        t = self.interpolate( t, shape )
        return t > 0.5

    def resize_mask(self, m, newShape):
        '''
        Arguments: 
        m (tensor): [ B, C, H, W ] in torch.bool.
        newShape (2-element): (H, W)

        Returns:
        A resized mask.
        '''

        return self.interpolate_bool(m, newShape)

@register(TRUE_GEN)
class MultiScaleTrueValues(TrueValueGenerator):
    def __init__(self, scales, trueDispMax, flagIntNearest=False):
        super(MultiScaleTrueValues, self).__init__(
            trueDispMax, flagIntNearest)

        self.scales = scales

    def make_true_values(self, dispL, occL=None, validL=None):
        B, C, H, W = dispL.size()

        with torch.no_grad():
            dispLs = [ dispL ]

            for i in range(1, self.scales):
                dispLs.append( 
                    self.interpolate( dispL,  (int(H / 2**i), int(W / 2**i)) ) * 0.5**i )
            
            maskLs = [ dispLs[i] <= self.trueDispMax * 0.5**i \
                    for i in range(self.scales) ]

            if ( occL is not None ):
                occLs = [ occL > 0.5 ]
                for i in range(1, self.scales):
                    scaledOcc = self.interpolate( occL, (int(H/2**i),  int(W/2**i) ) )
                    occLs.append( scaledOcc > 0.5 )

                trueValues = {
                    TT.DISP_LIST: dispLs,
                    TT.OCC_LIST: occLs }
            else:
                trueValues = {
                    TT.DISP_LIST: dispLs }

            if ( validL is not None ):
                maskLs[0] = torch.logical_and( 
                    maskLs[0], validL )

                for i in range(1, self.scales):
                    maskLs[i] = torch.logical_and( 
                        maskLs[i], 
                        self.interpolate_bool( validL, (int(H/2**i), int(W/2**i)) ) )

            trueValues[TT.MASK_LIST] = maskLs

        return trueValues

@register(TRUE_GEN)
class OriScaleTrueValues(TrueValueGenerator):
    def __init__(self, trueDispMax, flagIntNearest=False):
        super(OriScaleTrueValues, self).__init__(
            trueDispMax, flagIntNearest)

    def make_true_values(self, dispL, occL=None, validL=None):
        B, C, H, W = dispL.size()

        with torch.no_grad():
            maskL = dispL <= self.trueDispMax

            if ( occL is not None ):
                trueValues = {
                    TT.DISP: dispL, 
                    TT.OCC: occL > 0.5 }
            else:
                trueValues = {
                    TT.DISP: dispL }

            if ( validL is not None ):
                maskL = torch.logical_and( maskL, validL )

            trueValues[TT.MASK] = maskL

        return trueValues
