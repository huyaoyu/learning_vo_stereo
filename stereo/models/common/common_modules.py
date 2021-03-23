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

def selected_relu(x):
    # return F.selu(x, inplace=False)
    return F.leaky_relu(x, 0.1, inplace=GLOBAL.torch_relu_inplace())

class SelectedReLU(BaseModule):
    def __init__(self):
        super(SelectedReLU, self).__init__()

    def initialize(self):
        # Do nothing.
        self.mark_initialized()

    def forward(self, x):
        return selected_relu(x)

class Conv(BaseModule):
    def __init__(self, inCh, outCh, k=3, s=1, p=1, d=1, 
        normLayer=None, activation=None, bias=False):
        super(Conv, self).__init__()

        moduleList = [
            nn.ReflectionPad2d(padding=p),
            nn.Conv2d(inCh, outCh, 
                kernel_size=k, stride=s, padding=0, dilation=d, bias=bias) ]
        
        if ( normLayer is not None ):
            moduleList.append( normLayer )

        if ( activation is not None):
            moduleList.append( activation )

        self.model = nn.Sequential( *moduleList )

    def initialize(self):
        # Do nothing.
        self.mark_initialized()

    def forward(self, x):
        return self.model(x)

class Conv_W(BaseModule):
    def __init__(self, inCh, outCh, 
        k=3, d=1,
        normLayer=None, activation=None, bias=False):
        """
        k should be an odd number most of the time.
        """
        super(Conv_W, self).__init__()

        p = int(k // 2 * d) # Padding.

        moduleList = [
            nn.ReflectionPad2d(padding=p), 
            nn.Conv2d(inCh, outCh, kernel_size=k, stride=1, padding=0, dilation=d, bias=bias)
        ]

        if ( normLayer is not None):
            moduleList.append( normLayer )

        if ( activation is not None ):
            moduleList.append( activation )

        self.model = nn.Sequential( *moduleList )

    def initialize(self):
        # Do nothing.
        self.mark_initialized()

    def forward(self, x):
        return self.model(x)

class Conv_Half(BaseModule):
    def __init__(self, inCh, outCh, 
        k=3, d=1,
        normLayer=None, activation=None, bias=False):
        """
        k should be an odd number most of the time.
        """
        super(Conv_Half, self).__init__()

        p = int(k // 2 * d) # Padding.

        moduleList = [
            nn.ReflectionPad2d(padding=p), 
            nn.Conv2d(inCh, outCh, kernel_size=k, stride=2, padding=0, dilation=d, bias=bias)
        ]

        if ( normLayer is not None ):
            moduleList.append( normLayer )

        if ( activation is not None ):
            moduleList.append( activation )

        self.model = nn.Sequential( *moduleList )

    def initialize(self):
        # Do nothing.
        self.mark_initialized()

    def forward(self, x):
        return self.model(x)

class Deconv_Double(BaseModule):
    def __init__(self, inCh, outCh, k=3, p=1, op=0, activation=None, bias=False):
        super(Deconv_Double, self).__init__()

        moduleList = [ nn.ConvTranspose2d(inCh, outCh, kernel_size=k, stride=2, padding=p, dilation=1, output_padding=op, bias=bias) ]

        if ( activation is not None ):
            moduleList.append( activation )

        self.model = nn.Sequential( *moduleList )

    def initialize(self):
        # Do nothing.
        self.mark_initialized()

    def forward(self, x):
        return self.model(x)

class Interpolate2D_FixedScale(BaseModule):
    def __init__(self, s, flagNearest=False):
        super(Interpolate2D_FixedScale, self).__init__()

        self.s = s
        self.flagAlignCorners = GLOBAL.torch_align_corners()

        self.mode = 'nearest' if flagNearest else 'bilinear'

    def initialize(self):
        # Do nothing.
        self.mark_initialized()

    def forward(self, x):
        # Get the size of x.
        h = x.size()[-2]
        w = x.size()[-1]

        newH = int( h * self.s )
        newW = int( w * self.s )

        return F.interpolate( x, (newH, newW), \
            mode=self.mode, align_corners=self.flagAlignCorners )

class FeatureNormalization(BaseModule):
    def __init__(self, inCh, trs=None):
        """
        trs stands for track_running_stats.
        """
        super(FeatureNormalization, self).__init__()

        assert inCh > 0

        if ( trs is None ):
            self.model = nn.BatchNorm2d(inCh, 
                track_running_stats=GLOBAL.torch_batch_normal_track_stat())
        else:
            self.model = nn.BatchNorm2d(inCh, 
                track_running_stats=trs)

    def initialize(self):
        self.model.weight.data.fill_(1.0)
        self.model.bias.data.zero_()
        self.mark_initialized()

    def forward(self, x):
        return self.model(x)

class InstanceNormalization(nn.Module):
    def __init__(self, inCh, trs=False):
        """
        trs stands for track_running_stats.
        """
        super(InstanceNormalization, self).__init__()

        if (trs is None):
            self.model = nn.InstanceNorm2d(inCh, 
                track_running_stats=GLOBAL.torch_inst_normal_track_stat())
        else:
            self.model = nn.InstanceNorm2d(inCh, track_running_stats=trs)

    def initialize(self):
        self.model.weight.data.fill_(1.0)
        self.model.bias.data.zero_()
        self.mark_initialized()

    def forward(self, x):
        return self.model(x)

NORM_LAYER_TYPE = { 'batch': FeatureNormalization, 'instance': InstanceNormalization }

class ResidualBlock(BaseModule):
    def __init__(self, inChs, interChs, 
        stride=1, downsample=None, dilation=1, 
        lastActivation=None):
        super(ResidualBlock, self).__init__()

        self.flagReLUInplace = GLOBAL.torch_relu_inplace()

        if dilation > 1:
            padding = dilation
        else:
            padding = 1
        
        self.conv0 = Conv( inChs, interChs, 
            k=3, s=stride, p=padding, d=dilation,
            normLayer=FeatureNormalization(interChs),
            activation=nn.ReLU(inplace=self.flagReLUInplace) )
        self.conv1 = Conv_W( interChs, interChs, 
            normLayer=FeatureNormalization(interChs) )

        self.downsample = downsample
        self.stride = stride

        self.lastActivation = lastActivation \
            if lastActivation is not None \
            else None

    def initialize(self):
        # Do nothing.
        self.mark_initialized()

    def forward(self, x):
        residual = x

        out = self.conv0(x)
        out = self.conv1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if ( self.lastActivation is not None ):
            out = self.lastActivation(out)
        return out
