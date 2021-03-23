# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-21

# Top level imports.
from stereo.models.globals import GLOBAL

# System packages.
import numpy as np

# PyTorch packages.
import torch
# import torch.nn as nn
import torch.nn.functional as F

# Local packages.
from stereo.models.common.base_module import BaseModule
from stereo.models.register import ( DISP_REG, register )

@register(DISP_REG)
class ClsLinearCombination(BaseModule):
    @classmethod
    def get_default_init_args(cls):
        return dict( 
            type=cls.__name__,
            maxDisp=192,
            divisor=1, 
            freeze=False)

    def __init__(self, maxDisp, divisor, freeze=False):
        super(ClsLinearCombination, self).__init__(freeze=freeze)
        maxDisp = int(maxDisp/divisor)
        #self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxDisp)),[1,maxDisp,1,1])).cuda(), requires_grad=False)
        self.register_buffer('disp',torch.Tensor(np.reshape(np.array(range(maxDisp)),[1,maxDisp,1,1])))
        self.divisor = divisor

        # Must be called at the end of __init__().
        self.update_freeze()

    def initialize(self):
        # Do nothing.
        self.mark_initialized()

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0],1,x.size()[2],x.size()[3])
        out  = torch.sum( x * disp, 1) * self.divisor
        return out