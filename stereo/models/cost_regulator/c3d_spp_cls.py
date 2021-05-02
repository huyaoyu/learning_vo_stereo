# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-04-30

# c3d_spp_cls means
# 3D convolution + Spatial Pyramid Pooling + classification.

# Top level imports.
from stereo.models.globals import GLOBAL

# PyTorch packages.
import torch
import torch.nn as nn

# Local packages.
from stereo.models.common.base_module import ( BaseModule, WrappedModule )
from stereo.models.common import common_modules as cm
from stereo.models.common import common_modules_3d as cm3d
from stereo.models.common.pooling import SPP3D
from stereo.models.register import ( COST_PRC, register )

class ProjFeat3D(BaseModule):
    '''
    Turn 3d projection into 2d projection
    '''
    def __init__(self, inCh, outCh, stride):
        super(ProjFeat3D, self).__init__()

        self.stride = stride
        self.conv = cm.Conv( inCh, outCh, 
            k=1, s=stride[:2], p=0, 
            normLayer=cm.FeatureNormalization(outCh), 
            activation=None )

    # Override
    def initialize(self):
        super(ProjFeat3D, self).initialize()

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.conv( x.view(B, C, D, H*W) )
        return x.view( B, -1, D//self.stride[0], H, W)

class SepConv3DBlock(BaseModule):
    '''
    ResNet like 3D convolusion block.
    '''
    def __init__(self, inCh, outCh, stride=(1,1,1)):
        super(SepConv3DBlock, self).__init__()

        self.flagReLUInplace = GLOBAL.torch_relu_inplace()

        if inCh == outCh and stride==(1,1,1):
            self.downsample = None
        else:
            self.downsample = ProjFeat3D(inCh, outCh, stride)

        self.conv0 = cm3d.Conv3D( inCh, outCh, s=stride, 
            normLayer=cm3d.FeatureNorm3D(outCh), 
            activation=nn.ReLU(inplace=self.flagReLUInplace) )
        self.conv1 = cm3d.Conv3D_W( outCh, outCh, 
            normLayer=cm3d.FeatureNorm3D(outCh), 
            activation=nn.ReLU(inplace=self.flagReLUInplace) )
    
    # Override.
    def initialize(self):
        super(SepConv3DBlock, self).initialize()

    def forward(self,x):
        # Convolusion branch.
        out = self.conv0(x)

        # Short-cut branch.
        if self.downsample:
            x = self.downsample(x)
        
        return x + self.conv1(out)

class DecoderBlock(BaseModule):
    def __init__(self, 
        nConvs, inCh, interCh, outCh,
        baseStride=(1,1,1), nStrides=1, 
        outputUpSampledFeat=False, pooling=False ):
        super(DecoderBlock, self).__init__()

        # Get the global settings.
        self.flagAlignCorners = GLOBAL.torch_align_corners()
        self.flagReLUInplace  = GLOBAL.torch_relu_inplace()

        # Prepare the list of strides.
        assert( nConvs >= nStrides )
        strideList = [baseStride] * nStrides + [(1,1,1)] * (nConvs - nStrides)

        # Create the the convolusion layers.
        convs = [ SepConv3DBlock( inCh, interCh, stride=strideList[0] ) ]
        for i in range(1, nConvs):
            convs.append( SepConv3DBlock( interCh, interCh, stride=strideList[i] ) )
        self.entryConvs = WrappedModule( nn.Sequential(*convs) )
        self.append_init_here( self.entryConvs )

        # Classification layer.
        self.classify = WrappedModule(
            nn.Sequential(
                cm3d.Conv3D_W( interCh, interCh, 
                    normLayer=cm3d.FeatureNorm3D(interCh), 
                    activation=nn.ReLU(inplace=self.flagReLUInplace) ), 
                cm3d.Conv3D_W(interCh, outCh, bias=True) ) )
        self.append_init_here(self.classify)

        # Feature up-sample setting.
        self.featUpSampler = None
        if outputUpSampledFeat:
            self.featUpSampler = WrappedModule(
                nn.Sequential(
                    cm3d.Interpolate3D_FixedScale(2),
                    cm3d.Conv3D_W( interCh, interCh//2, 
                        normLayer=cm3d.FeatureNorm3D(interCh//2), 
                        activation=nn.ReLU(inplace=self.flagReLUInplace) ) ) )
            self.append_init_here(self.featUpSampler)

        # Pooling.
        if pooling:
            self.spp = SPP3D( interCh, levels=4 )
            self.append_init_here(self.spp)
        else:
            self.spp = None

    # Override.
    def initialize(self):
        if ( self.is_initialized() ):
            print('Warning: Already initialized!')
            return

        for m in self.toBeInitializedHere:
            if ( not m.is_initialized() ):
                for mm in m.modules():
                    if isinstance(mm, nn.Conv3d):
                        # n = np.prod(mm.kernel_size) * mm.out_channels
                        # nn.init.normal_(mm.weight, 0, np.sqrt( 2. / n ))
                        if hasattr(mm.bias, 'data'):
                            nn.init.zeros_(mm.bias)
                
                m.mark_initialized()

        self.mark_initialized()

    def forward(self, x):
        # Entry.
        x = self.entryConvs(x)

        # Pooling.
        if ( self.spp is not None ):
            x = self.spp(x)

        # Classification.
        cost = self.classify(x)

        # Up-sample the feature.
        if ( self.featUpSampler is not None ):
            x = self.featUpSampler(x)

        return cost, x

C3D_SPP_CLS_DEFAULT_SPECS = [ 
    { 'nEntryConvs': 6, 'inCh': 32, 'interCh': 32, 'outCh': 1 },
    { 'nEntryConvs': 6, 'inCh': 32, 'interCh': 32, 'outCh': 1 },
    { 'nEntryConvs': 6, 'inCh': 32, 'interCh': 32, 'outCh': 1 },
    { 'nEntryConvs': 6, 'inCh': 32, 'interCh': 32, 'outCh': 1 }
]

# C3D_SPP_CLS means
# 3D convolution + Spatial Pyramid Pooling + classification.
@register(COST_PRC)
class C3D_SPP_CLS(BaseModule):
    @classmethod
    def get_default_init_args(cls):
        return dict(
            specs=C3D_SPP_CLS_DEFAULT_SPECS,
            freeze=False )

    def __init__(self, specs=C3D_SPP_CLS_DEFAULT_SPECS, freeze=False):
        super(C3D_SPP_CLS, self).__init__(freeze=freeze)

        self.nLevels = len(specs)

        self.regulators = nn.ModuleList()
        for i in range(self.nLevels):
            flagUpsample = True if i != 0 else False
            flagPool     = True if i == self.nLevels - 1 else False
            spec = specs[i]
            regulator = DecoderBlock(
                spec['nEntryConvs'], 
                spec['inCh'], 
                spec['interCh'], 
                outCh=spec['outCh'], 
                outputUpSampledFeat=flagUpsample, 
                pooling=flagPool)
            self.regulators.append( regulator )
            self.append_init_impl( regulator ) # Defined in the parent class.

        # Must be called at the end of __init__().
        self.update_freeze()

    # Override.
    def initialize(self):
        super(C3D_SPP_CLS, self).initialize()

    def forward(self, costVolList):
        assert( len(costVolList) == self.nLevels ), \
            'len(costVoList) = %d, self.nLevels = %d. ' % \
                ( len(costVolList, self.nLevels) )
        
        costList = []
        start    = self.nLevels - 1
        for i in range( start, -1, -1 ):
            if ( i == start ):
                feat = costVolList[i]
            else:
                feat = torch.cat( ( feat2x, costVolList[i] ), dim=1 )
            cost, feat2x = self.regulators[i]( feat )
            costList.append(cost)
        # Make the order consistent.
        return costList[::-1]