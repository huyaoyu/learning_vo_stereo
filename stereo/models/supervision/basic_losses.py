# coding=utf-8

import numpy as np

import torch
import torch.nn.functional as F

from stereo.models.globals import GLOBAL as modelGLOBAL
from stereo.models.supervision.true_value import TT

from stereo.models.register import ( LOSS_CMP, register )

# LT stands for loss term
class LT(object):
    LOSS      = 'loss'
    LOSS_LIST = 'lossPerScale'

def single_channel_mask_from_four_channel(a, m):
    '''It is assumed that m has dimension [B, H, W] or [B, 1, H, W]'''

    if ( m.dim() == 4 ):
        assert(m.shape[1] == 1), f'm.shape[1] must be 1 if m is 4 dim. m.shape = {m.shape}'
        m = m.squeeze(1)

    b = a.permute((1, 0, 2, 3))
    return b[:, m].unsqueeze(0)

class LossComputer(object):
    def __init__(self, 
        flagMaskedLoss=True, 
        flagIntNearest=False):
        super(LossComputer, self).__init__()

        self.flagAlignCorners = modelGLOBAL.torch_align_corners()
        self.flagIntNearest   = flagIntNearest
        self.flagMaskedLoss   = flagMaskedLoss # Set True to mask the true disparity larger than self.trueDispMask.

    def interpolate(self, t, shape):
        if ( self.flagIntNearest ):
            return F.interpolate( t, shape, 
                mode="nearest" )
        else:
            return F.interpolate( t, shape, 
                mode="bilinear", align_corners=self.flagAlignCorners )

class ListLosses(LossComputer):
    def __init__(self,
        weights=None,
        flagMaskedLoss=True, 
        flagIntNearest=False):
        super(ListLosses, self).__init__(
            flagMaskedLoss=flagMaskedLoss, 
            flagIntNearest=flagIntNearest )

        self.weights  = weights
        self.nWeights = len(self.weights) \
            if self.weights is not None \
            else 0
    
    def weight_losses(self, losses):
        loss = 0
        if ( self.weights is not None ):
            for lv, w in zip(losses, self.weights):
                loss += lv * w
        else:
            for lv in zip(losses):
                loss += lv

        return loss

@register(LOSS_CMP)
class MultiScaleLoss(ListLosses):
    def __init__(self, 
        weights=None,
        flagMaskedLoss=True, 
        flagIntNearest=False):
        super(MultiScaleLoss, self).__init__(
            weights=weights,
            flagMaskedLoss=flagMaskedLoss, 
            flagIntNearest=flagIntNearest )

    def compute_masked_disp_loss(self, dispTLs, maskLs, dispPLs):
        losses = []

        for dispT, mask, dispP in zip( dispTLs, maskLs, dispPLs ):
            if ( dispP is None ):
                losses.append(0)
            else:
                losses.append(
                    F.smooth_l1_loss( dispP[mask], dispT[mask], reduction='mean' ) )
        
        return losses

    def compute_disp_loss(self, dispTLs, dispPLs):
        losses = []
        for dispT, dispP in zip( dispTLs, dispPLs ):
            if ( dispP is None ):
                losses.append(0)
            else:
                losses.append(
                    F.smooth_l1_loss( dispP, dispT, reduction='mean' ) )

        return losses

    def compute_loss(self, trueValueDict, predValueDict):    
        dispTLs = trueValueDict[TT.DISP_LIST]
        dispPLs = predValueDict[TT.DISP_LIST]
        maskLs  = trueValueDict[TT.MAKS_LIST]

        assert( len(dispTLs) == len(dispPLs) == len(maskLs) ), \
            f'The number of the true and predicted values should be equal. \
len(dispTLs) = {len(dispTLs)}, len(dispPLs) = {len(dispPLs)}, len(maskLs) = {len(maskLs)}'

        assert( self.nWeights == len(dispTLs) ), \
            f'The number of the weights must be equal to the number of true values. \
self.nWeights = {self.nWeights}, len(dispTLs) = {len(dispTLs)}'

        if ( self.flagMaskedLoss ):
            losses = self.compute_masked_disp_loss( dispTLs, maskLs, dispPLs )
        else:
            losses = self.compute_disp_loss( dispTLs, dispPLs )
        
        loss = self.weight_losses(losses)

        return { 
            LT.LOSS: loss,
            LT.LOSS_LIST: losses }

@register(LOSS_CMP)
class OriScaleLoss(ListLosses):
    def __init__(self, 
        weights=None,
        flagMaskedLoss=True, 
        flagIntNearest=False):
        super(OriScaleLoss, self).__init__(
            weights=weights,
            flagMaskedLoss=flagMaskedLoss, 
            flagIntNearest=flagIntNearest )

    def scale_tensor_if_not_same(self, ref, x, flagScaleValue=False):
        '''Resize x if the shape of x is not the same as ref. 
        Set flagSavleValue to scale the resized x as well.'''

        rH, rW = ref.shape[2:4]
        xH, xW = x.shape[2:4]

        if ( rH == xH and rW == xW ):
            return x

        x = self.interpolate( x, ( rH, rW ) )

        if ( flagScaleValue ):
            x = x * rW / xW

        return x

    def compute_masked_disp_loss(self, dispT, mask, dispPLs):
        losses = []

        for dispP in dispPLs:
            if ( dispP is None ):
                losses.append(0)
            else:
                dp = self.scale_tensor_if_not_same( 
                    dispT, dispP, flagScaleValue=True )
                losses.append(
                    F.smooth_l1_loss( dp[mask], dispT[mask], reduction='mean' ) )
        
        return losses

    def compute_disp_loss(self, dispT, dispPLs):
        losses = []

        for dispP in dispPLs:
            if ( dispP is None ):
                losses.append(0)
            else:
                dp = self.scale_tensor_if_not_same( 
                    dispT, dispP, flagScaleValue=True )
                losses.append(
                    F.smooth_l1_loss( dispP, dispT, reduction='mean' ) )
        
        return losses

    def compute_loss(self, trueValueDict, predValueDict):    
        dispTL  = trueValueDict[TT.DISP]
        dispPLs = predValueDict[TT.DISP_LIST]
        maskL   = trueValueDict[TT.MASK]

        if ( self.flagMaskedLoss ):
            losses = self.compute_masked_disp_loss( dispTL, maskL, dispPLs )
        else:
            losses = self.compute_disp_loss( dispTL, dispPLs )
        
        loss = self.weight_losses(losses)

        return { 
            LT.LOSS: loss,
            LT.LOSS_LIST: losses }
