# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-21

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
from .register import ( FEAT_EXT, register )
from .utils import ( 
    conv2DBatchNormRelu,
    residualBlock, 
    pyramidPooling )

@register(FEAT_EXT)
class UNet(BaseModule):
    @classmethod
    def get_default_init_args(cls):
        return dict(
            type=cls.__name__,
            initialChannels=32 )

    def __init__(self, initialChannels=32, freeze=False):
        super(UNet, self).__init__(freeze=freeze)

        self.flagTS = GLOBAL.torch_batch_normal_track_stat()

        self.inplanes = initialChannels

        # Encoder
        self.convbnrelu1_1 = conv2DBatchNormRelu(in_channels=3, k_size=3, n_filters=16,
                                                 padding=1, stride=2, bias=False)
        self.convbnrelu1_2 = conv2DBatchNormRelu(in_channels=16, k_size=3, n_filters=16,
                                                 padding=1, stride=1, bias=False)
        self.convbnrelu1_3 = conv2DBatchNormRelu(in_channels=16, k_size=3, n_filters=32,
                                                 padding=1, stride=1, bias=False)
        # Vanilla Residual Blocks
        self.res_block3 = self._make_layer(residualBlock,64,1,stride=2)
        self.res_block5 = self._make_layer(residualBlock,128,1,stride=2)

        self.res_block6 = self._make_layer(residualBlock,128,1,stride=2)
        self.res_block7 = self._make_layer(residualBlock,128,1,stride=2)
        self.pyramid_pooling = pyramidPooling(128, None,  fusion_mode='sum', model_name='icnet')
        # Iconvs
        self.upconv6 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1, bias=False))
        self.iconv5 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128,
                                                 padding=1, stride=1, bias=False)
        self.upconv5 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1, bias=False))
        self.iconv4 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128,
                                                 padding=1, stride=1, bias=False)
        self.upconv4 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1, bias=False))
        self.iconv3 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1, bias=False)

        self.proj6 = conv2DBatchNormRelu(in_channels=128,k_size=1,n_filters=32, padding=0,stride=1,bias=False)
        self.proj5 = conv2DBatchNormRelu(in_channels=128,k_size=1,n_filters=16, padding=0,stride=1,bias=False)
        self.proj4 = conv2DBatchNormRelu(in_channels=128,k_size=1,n_filters=16, padding=0,stride=1,bias=False)
        self.proj3 = conv2DBatchNormRelu(in_channels=64, k_size=1,n_filters=16, padding=0,stride=1,bias=False)

        # Must be called at the end of __init__().
        self.update_freeze()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion, track_running_stats=self.flagTS),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #torch.nn.init.xavier_uniform(m.weight)
                #torch.nn.init.constant(m.weight, 0.001)
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()
#            elif isinstance(m, nn.BatchNorm2d):
#                m.weight.data.fill_(1)
#                m.bias.data.zero_()
#                m.running_mean.data.fill_(0)
#                m.running_var.data.fill_(1)
        
        self.mark_initialized()

    def forward(self, x):
        # H, W -> H/2, W/2
        conv1 = self.convbnrelu1_1(x)
        conv1 = self.convbnrelu1_2(conv1)
        conv1 = self.convbnrelu1_3(conv1)

        ## H/2, W/2 -> H/4, W/4
        pool1 = F.max_pool2d(conv1, 3, 2, 1)

        # H/4, W/4 -> H/16, W/16
        conv3 = self.res_block3(pool1)
        conv4 = self.res_block5(conv3)
        conv5 = self.res_block6(conv4)
        conv6 = self.res_block7(conv5)
        conv6 = self.pyramid_pooling(conv6)

        concat5 = torch.cat((conv5,self.upconv6(conv6)),dim=1)
        conv5 = self.iconv5(concat5) 

        concat4 = torch.cat((conv4,self.upconv5(conv5)),dim=1)
        conv4 = self.iconv4(concat4) 

        concat3 = torch.cat((conv3,self.upconv4(conv4)),dim=1)
        conv3 = self.iconv3(concat3) 

        proj6 = self.proj6(conv6)
        proj5 = self.proj5(conv5)
        proj4 = self.proj4(conv4)
        proj3 = self.proj3(conv3)
        return proj6,proj5,proj4,proj3