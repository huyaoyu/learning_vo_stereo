# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-20

# Top level imports.
from stereo.models.globals import GLOBAL

# System packages.
import math
import numpy as np

# PyTorch packages.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local packages.
from .base_module import BaseModule

# ==============================================

class conv2DBatchNorm(BaseModule):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1, with_bn=True):
        super(conv2DBatchNorm, self).__init__()

        self.flagTS = GLOBAL.torch_batch_normal_track_stat()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)


        if with_bn:
            self.cb_unit = nn.Sequential(conv_mod,
                                         nn.BatchNorm2d(int(n_filters), track_running_stats=self.flagTS),)
        else:
            self.cb_unit = nn.Sequential(conv_mod,)

    def initialize(self):
        # Do nothing.
        self.mark_initialized()

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs

class conv2DBatchNormRelu(BaseModule):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1, with_bn=True):
        super(conv2DBatchNormRelu, self).__init__()

        self.flagTS = GLOBAL.torch_batch_normal_track_stat()
        self.flagReLUInplace = GLOBAL.torch_relu_inplace()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters), track_running_stats=self.flagTS),
                                          nn.LeakyReLU(0.1, inplace=self.flagReLUInplace),)
        else:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.LeakyReLU(0.1, inplace=self.flagReLUInplace),)

    def initialize(self):
        # Do nothing.
        self.mark_initialized()

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class residualBlock(BaseModule):
    expansion = 1

    def __init__(self, in_channels, n_filters, stride=1, downsample=None,dilation=1):
        super(residualBlock, self).__init__()

        self.flagReLUInplace = GLOBAL.torch_relu_inplace()

        if dilation > 1:
            padding = dilation
        else:
            padding = 1
        self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3,  stride, padding, bias=False,dilation=dilation)
        self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1, bias=False)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=self.flagReLUInplace)

    def initialize(self):
        # Do nothing.
        self.mark_initialized()

    def forward(self, x):
        residual = x

        out = self.convbnrelu1(x)
        out = self.convbn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        #out = self.relu(out, inplace=self.flagReLUInplace)
        return out

class pyramidPooling(BaseModule):
    def __init__(self, in_channels, pool_sizes, model_name='pspnet', fusion_mode='cat', with_bn=True):
        super(pyramidPooling, self).__init__()

        self.flagAlignCorners = GLOBAL.torch_align_corners()
        self.flagReLUInplace  = GLOBAL.torch_relu_inplace()

        bias = not with_bn

        self.paths = []
        if pool_sizes is None:
            for i in range(4):
                self.paths.append(conv2DBatchNormRelu(in_channels, in_channels, 1, 1, 0, bias=bias, with_bn=with_bn))
        else:
            for i in range(len(pool_sizes)):
                self.paths.append(conv2DBatchNormRelu(in_channels, int(in_channels / len(pool_sizes)), 1, 1, 0, bias=bias, with_bn=with_bn))

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes
        self.model_name = model_name
        self.fusion_mode = fusion_mode
    
    def initialize(self):
        # Do nothing.
        self.mark_initialized()

    #@profile
    def forward(self, x):
        h, w = x.shape[2:]

        k_sizes = []
        strides = []
        if self.pool_sizes is None: 
            for pool_size in np.linspace(1,min(h,w)//2,4,dtype=int):
                k_sizes.append((int(h/pool_size), int(w/pool_size)))
                strides.append((int(h/pool_size), int(w/pool_size)))
            k_sizes = k_sizes[::-1]
            strides = strides[::-1]
        else:
            k_sizes = [(self.pool_sizes[0],self.pool_sizes[0]),(self.pool_sizes[1],self.pool_sizes[1]) ,(self.pool_sizes[2],self.pool_sizes[2]) ,(self.pool_sizes[3],self.pool_sizes[3])]
            strides = k_sizes

        if self.fusion_mode == 'cat': # pspnet: concat (including x)
            output_slices = [x]

            for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                #out = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
                if self.model_name != 'icnet':
                    out = module(out)
                out = F.interpolate(out, size=(h,w), 
                    mode='bilinear', align_corners=self.flagAlignCorners)
                output_slices.append(out)

            return torch.cat(output_slices, dim=1)
        else: # icnet: element-wise sum (including x)
            pp_sum = x

            for i, module in enumerate(self.path_module_list):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                out = module(out)
                out = F.interpolate(out, size=(h,w), 
                    mode='bilinear', align_corners=self.flagAlignCorners)
                pp_sum = pp_sum + 0.25*out
            pp_sum = F.relu(pp_sum/2.,inplace=self.flagReLUInplace)

            return pp_sum
