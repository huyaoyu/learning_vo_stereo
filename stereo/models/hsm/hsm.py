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
from stereo.models.cost.cost_volume import CVDiff
from stereo.models.disparity_regression.cls_linear_combination import ClsLinearCombination
from stereo.models.feature_extractor.unet import UNet
from stereo.models.supervision.true_value import TT
from stereo.models.uncertainty.classified_cost_volume_epistemic import ClassifiedCostVolumeEpistemic
from .submodules import *

from stereo.models.register import (
    FEAT_EXT, COST_VOL, DISP_REG, MODELS, register, make_object )

@register(MODELS)
class HSMNet(BaseModule):
    def __init__(self, 
        maxDisp=192,
        featExtConfig=None, 
        costVolConfig=None,
        dispRegConfigs=None,
        uncertainty=False,
        freeze=False):

        super(HSMNet, self).__init__(freeze=freeze)

        # Global setttings.
        self.flagAlignCorners = GLOBAL.torch_align_corners()

        # ========== Module definition. ==========
        self.maxDisp = maxDisp

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

        # block 4
        self.decoder6 = decoderBlock(6,32,32,up=True, pool=True, uncertainty=uncertainty)
        self.append_init_impl(self.decoder6)

        self.decoder5 = decoderBlock(6,32,32,up=True, pool=True, uncertainty=uncertainty)
        self.append_init_impl(self.decoder5)

        self.decoder4 = decoderBlock(6,32,32, up=True, uncertainty=uncertainty)
        self.append_init_impl(self.decoder4)

        self.decoder3 = decoderBlock(5,32,32, stride=(2,1,1),up=False, nstride=1, uncertainty=uncertainty)
        self.append_init_impl(self.decoder3)
        
        self.uncertainty = uncertainty

        # Disparity regressions.

        # self.disp_reg3 = disparityregression(self.maxDisp,16)
        # self.disp_reg4 = disparityregression(self.maxDisp,16)
        # self.disp_reg3 = disparityregression(self.maxDisp,32)
        # self.disp_reg6 = disparityregression(self.maxDisp,64)

        if ( dispRegConfigs is None ):
            dispRegConfigs = [ ClsLinearCombination.get_default_init_args() for _ in range(4)]
        else:
            assert( isinstance( dispRegConfigs, (tuple, list) ) ), \
                f'dispRegConfigs must be a tuple or list. It is {type(dispRegConfigs)}'

        self.disp_reg3 = make_object(DISP_REG, dispRegConfigs[0])
        self.disp_reg4 = make_object(DISP_REG, dispRegConfigs[1])
        self.disp_reg5 = make_object(DISP_REG, dispRegConfigs[2])
        self.disp_reg6 = make_object(DISP_REG, dispRegConfigs[3])

        self.append_init_impl(self.disp_reg3)
        self.append_init_impl(self.disp_reg4)
        self.append_init_impl(self.disp_reg5)
        self.append_init_impl(self.disp_reg6)

        self.uncertaintyComputer = ClassifiedCostVolumeEpistemic() \
            if self.uncertainty else None

        # Must be called at the end of __init__().
        self.update_freeze()

    # Override.
    def initialize(self):
        super(HSMNet, self).initialize()

    def feature_vol(self, refimg_fea, targetimg_fea, maxDisp, leftview=True):
        '''
        diff feature volume
        '''
        width = refimg_fea.shape[-1]
        # cost = Variable(torch.cuda.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1], maxDisp,  refimg_fea.size()[2],  refimg_fea.size()[3]).fill_(0.))
        cost = torch.Tensor( 
            ( refimg_fea.size()[0], refimg_fea.size()[1], maxDisp, refimg_fea.size()[2],  refimg_fea.size()[3] ),
            device=refimg_fea.device )
        for i in range(maxDisp):
            feata = refimg_fea[:,:,:,i:width]
            featb = targetimg_fea[:,:,:,:width-i]
            # concat
            if leftview:
                cost[:, :refimg_fea.size()[1], i, :,i:]   = torch.abs(feata-featb)
            else:
                cost[:, :refimg_fea.size()[1], i, :,:width-i]   = torch.abs(featb-feata)
        cost = cost.contiguous()
        return cost

    def forward(self, inputs, flagDeploy=False):
        left  = inputs['img0']
        right = inputs['img1']

        nsample = left.shape[0]
        conv1, conv2, conv3, conv4   = self.featureExtractor(torch.cat([left,right],0))
        conv40,conv30,conv20,conv10  = conv4[:nsample], conv3[:nsample], conv2[:nsample], conv1[:nsample]
        conv41,conv31,conv21,conv11  = conv4[nsample:], conv3[nsample:], conv2[nsample:], conv1[nsample:]

        feat6 = self.costVol( conv40, conv41, self.maxDisp//64 )
        feat5 = self.costVol( conv30, conv31, self.maxDisp//32 )
        feat4 = self.costVol( conv20, conv21, self.maxDisp//16 )
        feat3 = self.costVol( conv10, conv11, self.maxDisp//8  )

        feat6_2x, cost6 = self.decoder6(feat6)
        feat5 = torch.cat((feat6_2x, feat5), dim=1)

        feat5_2x, cost5 = self.decoder5(feat5)

        feat4 = torch.cat((feat5_2x, feat4), dim=1)
        feat4_2x, cost4 = self.decoder4(feat4) # 32

        feat3 = torch.cat((feat4_2x, feat3), dim=1)
        feat3_2x, cost3 = self.decoder3(feat3) # 32

        cost3 = F.interpolate(cost3, 
            [ self.disp_reg3.disp.shape[1], left.shape[2], left.shape[3] ], 
            mode='trilinear', align_corners=self.flagAlignCorners)

        if ( self.uncertainty ):
            cost3, uChannels3 = torch.split( cost3, 1, dim=1 )
            logSigmaSqured3 = self.uncertaintyComputer( uChannels3.squeeze(1), left.shape[2:4] )

        pred3 = self.disp_reg3( F.softmax(cost3.squeeze(1), 1) )

        if ( not flagDeploy ):
            cost6 = F.interpolate(cost6, 
                [ self.disp_reg6.disp.shape[1], left.shape[2], left.shape[3] ], 
                mode='trilinear', align_corners=self.flagAlignCorners)
            cost5 = F.interpolate(cost5, 
                [ self.disp_reg5.disp.shape[1], left.shape[2], left.shape[3] ], 
                mode='trilinear', align_corners=self.flagAlignCorners)
            cost4 = F.interpolate(cost4, 
                [ self.disp_reg4.disp.shape[1], left.shape[2], left.shape[3] ], 
                mode='trilinear', align_corners=self.flagAlignCorners)

            if ( self.uncertainty ):
                cost6, uChannels6 = torch.split( cost6, 1, dim=1 )
                logSigmaSqured6 = self.uncertaintyComputer( uChannels6.squeeze(1), left.shape[2:4] )
                cost5, uChannels5 = torch.split( cost5, 1, dim=1 )
                logSigmaSqured5 = self.uncertaintyComputer( uChannels5.squeeze(1), left.shape[2:4] )
                cost4, uChannels4 = torch.split( cost4, 1, dim=1 )
                logSigmaSqured4 = self.uncertaintyComputer( uChannels4.squeeze(1), left.shape[2:4] )

                uncertainties = [ 
                    logSigmaSqured3, logSigmaSqured4, logSigmaSqured5, logSigmaSqured6 ]

            pred6 = self.disp_reg6( F.softmax(cost6.squeeze(1), 1) )
            pred5 = self.disp_reg5( F.softmax(cost5.squeeze(1), 1) )
            pred4 = self.disp_reg4( F.softmax(cost4.squeeze(1), 1) )

            stacked = [ 
                pred3.unsqueeze(1), 
                pred4.unsqueeze(1), 
                pred5.unsqueeze(1), 
                pred6.unsqueeze(1)]
        else:
            if ( self.uncertainty ):
                uncertainties = [ logSigmaSqured3 ]
            stacked = [ pred3.unsqueeze(1) ]
        
        res = { TT.DISP_LIST: stacked }

        if ( self.uncertainty ):
            res[TT.UNCT_LIST] = uncertainties

        return res