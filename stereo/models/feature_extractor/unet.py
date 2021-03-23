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
from stereo.models.common import common_modules as cm
from stereo.models.common import pooling

from stereo.models.register import ( FEAT_EXT, register )

@register(FEAT_EXT)
class UNet(BaseModule):
    @classmethod
    def get_default_init_args(cls):
        return dict(
            type=cls.__name__,
            initialChannels=32,
            freeze=False )

    def __init__(self, initialChannels=32, freeze=False):
        super(UNet, self).__init__(freeze=freeze)

        self.flagTS = GLOBAL.torch_batch_normal_track_stat()
        self.flagReLUInplace = GLOBAL.torch_relu_inplace()

        self.inplanes = initialChannels

        # Encoder
        self.convBnReLU1_1 = cm.Conv_Half( 3, 16, 
            normLayer=cm.FeatureNormalization(16), 
            activation=nn.ReLU(inplace=self.flagReLUInplace) )
        self.convBnReLU1_2 = cm.Conv_W( 16, 16, 
            normLayer=cm.FeatureNormalization(16), 
            activation=nn.ReLU(inplace=self.flagReLUInplace) )
        self.convBnReLU1_3 = cm.Conv_W( 16, 32, 
            normLayer=cm.FeatureNormalization(32), 
            activation=nn.ReLU(inplace=self.flagReLUInplace) )
        
        # Vanilla Residual Blocks
        self.resBlock3 = self._make_layer(cm.ResidualBlock, interCh= 64, stride=2)
        self.resBlock5 = self._make_layer(cm.ResidualBlock, interCh=128, stride=2)
        self.resBlock6 = self._make_layer(cm.ResidualBlock, interCh=128, stride=2)
        self.resBlock7 = self._make_layer(cm.ResidualBlock, interCh=128, stride=2)

        # This does not make sense with very small feature size.
        self.pyramidPooling = \
            pooling.SpatialPyramidPooling(128, levels=4, 
                lastActivation=nn.ReLU(inplace=self.flagReLUInplace))

        # iConvs.
        self.upConv6 = nn.Sequential( 
            cm.Interpolate2D_FixedScale(2),
            cm.Conv_W( 128, 64, 
                normLayer=cm.FeatureNormalization(64), 
                activation=nn.ReLU(inplace=self.flagReLUInplace) ) )
        self.iConv5 = cm.Conv_W( 192, 128, 
            normLayer=cm.FeatureNormalization(128), 
            activation=nn.ReLU(inplace=self.flagReLUInplace) )
        
        self.upConv5 = nn.Sequential( 
            cm.Interpolate2D_FixedScale(2),
            cm.Conv_W( 128, 64, 
                normLayer=cm.FeatureNormalization(64), 
                activation=nn.ReLU(inplace=self.flagReLUInplace) ) )
        self.iConv4 = cm.Conv_W( 192, 128, 
            normLayer=cm.FeatureNormalization(128), 
            activation=nn.ReLU(inplace=self.flagReLUInplace) )
        
        self.upConv4 = nn.Sequential( 
            cm.Interpolate2D_FixedScale(2),
            cm.Conv_W( 128, 64, 
                normLayer=cm.FeatureNormalization(64), 
                activation=nn.ReLU(inplace=self.flagReLUInplace) ) )
        self.iConv3 = cm.Conv_W( 128, 64, 
            normLayer=cm.FeatureNormalization(64), 
            activation=nn.ReLU(inplace=self.flagReLUInplace) )

        self.proj6 = cm.Conv_W(128, 32, k=1, 
            normLayer=cm.FeatureNormalization(32), 
            activation=nn.ReLU(inplace=self.flagReLUInplace) )
        self.proj5 = cm.Conv_W(128, 16, k=1, 
            normLayer=cm.FeatureNormalization(16), 
            activation=nn.ReLU(inplace=self.flagReLUInplace) )
        self.proj4 = cm.Conv_W(128, 16, k=1, 
            normLayer=cm.FeatureNormalization(16), 
            activation=nn.ReLU(inplace=self.flagReLUInplace) )
        self.proj3 = cm.Conv_W(64, 16, k=1, 
            normLayer=cm.FeatureNormalization(16), 
            activation=nn.ReLU(inplace=self.flagReLUInplace) )

        # Must be called at the end of __init__().
        self.update_freeze()

    def _make_layer(self, block, interCh, blocks=1, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != interCh:
            downsample = cm.Conv( self.inplanes, interCh,
                k=1, s=stride, p=0, 
                normLayer=cm.FeatureNormalization(interCh) )
        
        layers = [ block(self.inplanes, interCh, stride, downsample) ]
        self.inplanes = interCh
        for i in range(1, blocks):
            layers.append(block(interCh, interCh))
        
        return nn.Sequential(*layers)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
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
        # 1 -> 1/2.
        conv1 = self.convBnReLU1_1(x)
        conv1 = self.convBnReLU1_2(conv1)
        conv1 = self.convBnReLU1_3(conv1)

        # 1/2 -> 1/4.
        pool1 = F.max_pool2d(conv1, 3, 2, 1)

        # 1/4 -> 1/64.
        conv3 = self.resBlock3(pool1)
        conv4 = self.resBlock5(conv3)
        conv5 = self.resBlock6(conv4)
        conv6 = self.resBlock7(conv5)
        conv6 = self.pyramidPooling(conv6)

        concat5 = torch.cat((conv5,self.upConv6(conv6)),dim=1)
        conv5 = self.iConv5(concat5)

        concat4 = torch.cat((conv4,self.upConv5(conv5)),dim=1)
        conv4 = self.iConv4(concat4)

        concat3 = torch.cat((conv3,self.upConv4(conv4)),dim=1)
        conv3 = self.iConv3(concat3)

        proj6 = self.proj6(conv6)
        proj5 = self.proj5(conv5)
        proj4 = self.proj4(conv4)
        proj3 = self.proj3(conv3)
        return proj6,proj5,proj4,proj3