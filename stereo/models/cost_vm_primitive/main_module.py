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
from stereo.models.cost.cost_volume import CVDiff
from stereo.models.disparity_regression.cls_linear_combination import ClsLinearCombination
from stereo.models.feature_extractor.unet import UNet
from stereo.models.supervision.true_value import TT
from stereo.models.uncertainty.classified_cost_volume_epistemic import ClassifiedCostVolumeEpistemic
from . import sub_modules as sm

from stereo.models.register import (
    FEAT_EXT, COST_VOL, DISP_REG, MODELS, register, make_object )

@register(MODELS)
class CostVolPrimitive(BaseModule):
    def __init__(self, 
        maxDisp=192,
        featExtConfig=None, 
        costVolConfig=None,
        dispRegConfigs=None,
        uncertainty=False,
        freeze=False):

        super(CostVolPrimitive, self).__init__(freeze=freeze)

        # Global setttings.
        self.flagAlignCorners = GLOBAL.torch_align_corners()

        # ========== Module definition. ==========
        self.maxDisp = maxDisp

        # Uncertainty setting.
        self.uncertainty = uncertainty

        # Feature extractor.
        if ( featExtConfig is None ):
            featExtConfig = UNet.get_default_init_args()

        self.featureExtractor = make_object(FEAT_EXT, featExtConfig)
        self.append_init_impl(self.featureExtractor)
        
        # Cost volume.
        if ( costVolConfig is None ):
            costVolConfig = CVDiff.get_default_init_args()
        
        self.costVol = make_object(COST_VOL, costVolConfig)
        self.append_init_impl(self.costVol)

        # Cost volume regularization layers.
        nLevels = self.featureExtractor.n_levels()
        costRegulatorOutCh = 2 if self.uncertainty else 1
        self.CostRegulatorList = nn.ModuleList()
        for i in range( nLevels ):
            flagUpsample = True if i != 0 else False
            flagPool     = True if i == nLevels - 1 else False
            regulator = sm.DecoderBlock(
                6, 32, 32, 
                outCh=costRegulatorOutCh, 
                outputUpSampledFeat=flagUpsample, 
                pooling=flagPool)
            self.CostRegulatorList.append( regulator )
            self.append_init_impl( regulator )

        # Disparity regressions.
        if ( dispRegConfigs is None ):
            dispRegConfigs = [ ClsLinearCombination.get_default_init_args() for _ in range(nLevels)]
        else:
            assert( isinstance( dispRegConfigs, (tuple, list) ) ), \
                f'dispRegConfigs must be a tuple or list. It is {type(dispRegConfigs)}'
            assert( len(dispRegConfigs) == nLevels ), \
                f'len(dispRegConfigs) = {len(dispRegConfigs)}, nLevels = {nLevels}'

        self.dispRegList = nn.ModuleList()
        for config in dispRegConfigs:
            dispReg = make_object(DISP_REG, config)
            self.dispRegList.append( dispReg )
            self.append_init_impl( dispReg )

        # Uncertainty computer.
        self.uncertaintyComputer = ClassifiedCostVolumeEpistemic() \
            if self.uncertainty else None

        # Must be called at the end of __init__().
        self.update_freeze()

    # Override.
    def initialize(self):
        super(CostVolPrimitive, self).initialize()

    def predict_disp(self, imgShape, cost, dispReg):
        cost = F.interpolate( cost, 
            [ dispReg.disp.shape[1], imgShape[0], imgShape[1] ], 
            mode='trilinear', align_corners=self.flagAlignCorners )

        if ( self.uncertainty ):
            cost, uChannels = torch.split( cost, 1, dim=1 )
            logSigmaSqured  = self.uncertaintyComputer( uChannels.squeeze(1), imgShape )
        else:
            logSigmaSqured = None

        pred = dispReg( F.softmax( cost.squeeze(1), 1 ) )

        return pred, logSigmaSqured

    def forward(self, inputs, flagDeploy=False):
        left  = inputs['img0']
        right = inputs['img1']

        # Feature extraction.
        featList = self.featureExtractor( torch.cat( (left, right), 0 ) )

        # Build the cost volumes.
        nSamples = left.shape[0]
        levels = self.featureExtractor.levels
        costVolList = [ 
            self.costVol( feat[:nSamples], feat[nSamples:], self.maxDisp//level ) 
            for feat, level in zip(featList, levels) ]

        # Cost volume regularization.
        costList = []
        start    = len(levels) - 1
        for i in range( start, -1, -1 ):
            if ( i == start ):
                feat = costVolList[i]
            else:
                feat = torch.cat( ( feat2x, costVolList[i] ), dim=1 )
            cost, feat2x = self.CostRegulatorList[i]( feat )
            costList.append(cost)
        # Make the order consistent.
        costList = costList[::-1]

        # Prediction.
        pred0, logSigmaSqured0 = self.predict_disp( 
            left.shape[2:4], costList[0], self.dispRegList[0] )
        preds = [ pred0 ]
        uncertainties = [ logSigmaSqured0 ]

        # Only used for training/non-deployment.
        if ( not flagDeploy ):
            for i in range( 1, len(levels) ):
                pred, logSigmaSqured = self.predict_disp(
                    left.shape[2:4], costList[i], self.dispRegList[i] )

                preds.append( pred )
                uncertainties.append( logSigmaSqured )

        # Make the dimension right. A disparity has B, 1, H, W.
        stacked = [ pred.unsqueeze(1) for pred in preds ]
        
        # Prepare outputs.
        res = { TT.DISP_LIST: stacked }
        if ( self.uncertainty ):
            res[TT.UNCT_LIST] = uncertainties

        return res