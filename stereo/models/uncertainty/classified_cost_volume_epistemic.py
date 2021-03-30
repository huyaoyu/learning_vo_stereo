# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-30

import torch
import torch.nn.functional as F

from stereo.models.globals import GLOBAL as modelGLOBAL

class ClassifiedCostVolumeEpistemic(torch.nn.Module):
    def __init__(self):
        super(ClassifiedCostVolumeEpistemic, self).__init__()

        self.flagAlignCorners = modelGLOBAL.torch_align_corners()

    def forward(self, x, targetShape):
        '''
        x (tensor): BxCxHxW. Normally, the C dimension is the D dimension of the cost volume.
        targetShape (2-element): The target H and W.
        '''

        if ( 0 not in targetShape ):
            x = F.interpolate( x, targetShape, 
                mode='bilinear', align_corners=self.flagAlignCorners)

        return torch.mean( x, dim=1, keepdim=True )