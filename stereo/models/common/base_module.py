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

class BaseModule(nn.Module):
    def __init__(self, freeze=False):
        super(BaseModule, self).__init__()

        self.freeze = freeze
        
        self.toBeInitializedHere = [] # Modules that need to be initialzed here.
        self.toBeInitializedImpl = [] # Modules that have their own initialization sequeences.

        self.initialized = False

    def update_freeze(self):
        if ( self.freeze ):
            # Freeze this instance and all its components.
            for p in self.parameters():
                p.requires_grad = False

    def append_init_here(self, m):
        self.toBeInitializedHere.append(m)

    def append_init_impl(self, m):
        self.toBeInitializedImpl.append(m)

    def is_initialized(self):
        return self.initialized

    def mark_initialized(self):
        assert( not self.initialized )
        self.initialized = True

    def initialize(self):
        if ( self.is_initialized() ):
            print('Warning: Already initialized!')
            return

        for m in self.toBeInitializedHere:
            if ( not m.is_initialized() ):
                for mm in m.modules():
                    if ( isinstance( mm, nn.BatchNorm2d ) ):
                        mm.weight.data.fill_(1.0)
                        mm.bias.data.zero_()
                    elif ( isinstance( mm, nn.InstanceNorm2d ) ):
                        mm.weight.data.fill_(1.0)
                        mm.bias.data.zero_()
                    elif ( isinstance( mm, nn.BatchNorm3d ) ):
                        mm.weight.data.fill_(1.0)
                        mm.bias.data.zero_()
                    elif ( isinstance( mm, nn.InstanceNorm3d ) ):
                        mm.weight.data.fill_(1.0)
                        mm.bias.data.zero_()
                
                m.mark_initialized()
        
        for m in self.toBeInitializedImpl:
            m.initialize()

        self.mark_initialized()

class WrappedModule(BaseModule):
    def __init__(self, m, freeze=False):
        super(WrappedModule, self).__init__(freeze=freeze)

        self.model = m

    def forward(self, x):
        return self.model(x)
