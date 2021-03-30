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

class sepConv3dBlock(BaseModule):
    '''
    Separable 3d convolution block as 2 separable convolutions and a projection
    layer
    '''
    def __init__(self, in_planes, out_planes, stride=(1,1,1)):
        super(sepConv3dBlock, self).__init__()

        self.flagReLUInplace = GLOBAL.torch_relu_inplace()

        if in_planes == out_planes and stride==(1,1,1):
            self.downsample = None
        else:
            self.downsample = projfeat3d(in_planes, out_planes,stride)
        self.conv1 = sepConv3d(in_planes, out_planes, 3, stride, 1)
        self.conv2 = sepConv3d(out_planes, out_planes, 3, (1,1,1), 1)
    
    def initialize(self):
        super(sepConv3dBlock, self).initialize()

    def forward(self,x):
        out = F.relu(self.conv1(x),inplace=self.flagReLUInplace)
        if self.downsample:
            x = self.downsample(x)
        out = F.relu(x + self.conv2(out),inplace=self.flagReLUInplace)
        return out

class projfeat3d(BaseModule):
    '''
    Turn 3d projection into 2d projection
    '''
    def __init__(self, in_planes, out_planes, stride):
        super(projfeat3d, self).__init__()
        self.flagTS = GLOBAL.torch_batch_normal_track_stat()

        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, out_planes, (1,1), padding=(0,0), stride=stride[:2],bias=False)
        self.bn = nn.BatchNorm2d(out_planes, track_running_stats=self.flagTS)

    def initialize(self):
        super(projfeat3d, self).initialize()

    def forward(self,x):
        b,c,d,h,w = x.size()
        x = self.conv1(x.view(b,c,d,h*w))
        x = self.bn(x)
        x = x.view(b,-1,d//self.stride[0],h,w)
        return x

# original conv3d block
def sepConv3d(in_planes, out_planes, kernel_size, stride, pad,bias=False):
    if bias:
        return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=bias))
    else:
        return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=bias),
                         nn.BatchNorm3d(out_planes))

class decoderBlock(BaseModule):
    def __init__(self, nconvs, inchannelF, channelF, 
        stride=(1,1,1), up=False, nstride=1, pool=False, 
        uncertainty=False):
        super(decoderBlock, self).__init__()

        self.flagAlignCorners = GLOBAL.torch_align_corners()
        self.flagReLUInplace  = GLOBAL.torch_relu_inplace()

        self.pool=pool
        stride = [stride]*nstride + [(1,1,1)] * (nconvs-nstride)
        self.convs = [sepConv3dBlock(inchannelF,channelF,stride=stride[0])]
        for i in range(1,nconvs):
            self.convs.append(sepConv3dBlock(channelF,channelF, stride=stride[i]))
        self.convs = nn.Sequential(*self.convs)

        classifyModules = [ sepConv3d(channelF, channelF, 3, (1,1,1), 1),
                            nn.ReLU(inplace=self.flagReLUInplace) ]
        
        if ( uncertainty ):
            classifyModules.append( sepConv3d(channelF, 2, 3, (1,1,1), 1, bias=True) )
        else:
            classifyModules.append( sepConv3d(channelF, 1, 3, (1,1,1), 1, bias=True) )
        
        self.classify = nn.Sequential(*classifyModules)

        self.up = False
        if up:
            self.up = True
            self.up = nn.Sequential(nn.Upsample(scale_factor=(2,2,2),mode='trilinear', align_corners=self.flagAlignCorners),
                                 sepConv3d(channelF, channelF//2, 3, (1,1,1),1,bias=False),
                                 nn.ReLU(inplace=self.flagReLUInplace))

        if pool:
            self.pool_convs = torch.nn.ModuleList([sepConv3d(channelF, channelF, 1, (1,1,1), 0),
                               sepConv3d(channelF, channelF, 1, (1,1,1), 0),
                               sepConv3d(channelF, channelF, 1, (1,1,1), 0),
                               sepConv3d(channelF, channelF, 1, (1,1,1), 0)])

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                #torch.nn.init.xavier_uniform(m.weight)
                #torch.nn.init.constant(m.weight,0.001)
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()
            #elif isinstance(m, nn.BatchNorm3d):
            #    m.weight.data.fill_(1)
            #    m.bias.data.zero_()
            #    m.running_mean.data.fill_(0)
            #    m.running_var.data.fill_(1)

        self.mark_initialized()

    def forward(self,fvl):
        # left
        fvl = self.convs(fvl)
        # pooling
        if self.pool:
            fvl_out = fvl
            _,_,d,h,w=fvl.shape
            for i,pool_size in enumerate(np.linspace(1,min(d,h,w)//2,4,dtype=int)):
                kernel_size = (int(d/pool_size), int(h/pool_size), int(w/pool_size))
                out = F.avg_pool3d(fvl, kernel_size, stride=kernel_size)       
                out = self.pool_convs[i](out)
                out = F.interpolate(out, size=(d,h,w), 
                    mode='trilinear', align_corners=self.flagAlignCorners)
                fvl_out = fvl_out + 0.25*out
            fvl = F.relu(fvl_out/2.,inplace=self.flagReLUInplace)

        # classification
        costl = self.classify(fvl)
        if self.up:
            fvl = self.up(fvl)

        return fvl, costl