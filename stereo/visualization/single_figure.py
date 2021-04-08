# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-04-08

import colorcet as cc
import cv2
import numpy as np

from .color_map_utils import convert_colorcet_2_array
from .image_creation import ( 
    convert_float_array_2_image, add_string_line_2_img )
from .tensor_handles import simple_array_2_rgb

from .register import ( VISUALIZERS, register )

class FigureGenerator(object):
    def __init__(self, name, getDataFunc, newShape=None, flagAddString=True ):
        super(FigureGenerator, self).__init__()

        self.name = name # The name of the plot.
        self.getDataFunc = getDataFunc # Should be a callable.
        self.descString  = ''
        self.descStringRatio = 0.03
        self.newShape = newShape
        self.flagAddString = flagAddString

    # Must be overrided.
    def create_image(self, data):
        raise NotImplementedError()

    def append_desc_string(self, s):
        self.descString = '%s %s' % ( self.descString, s )

    def __call__(self, ctx, batchIdx):
        # Clear the description string.
        self.descString = ''

        # Get data.
        data = self.getDataFunc( ctx, batchIdx )

        # Create the image.
        img = self.create_image( data )

        # Reshape if required.
        if ( self.newShape is not None ):
                img = cv2.resize( img, ( self.newShape[1], self.newShape[0] ), interpolation=cv2.INTER_LINEAR )

        # Description string.
        if ( self.flagAddString ):
            descString = '%s %s' % ( self.name, self.descString )
            img = add_string_line_2_img( img, descString, self.descStringRatio )

        return img

@register(VISUALIZERS)
class RGBFloatGenerator(FigureGenerator):
    def __init__(self, name, getDataFunc, newShape=None, flagAddString=True):
        super(RGBFloatGenerator, self).__init__( name, getDataFunc, newShape, flagAddString )

    def create_image(self, data):
        return simple_array_2_rgb(data)

@register(VISUALIZERS)
class SingleChFloatGenerator(FigureGenerator):
    def __init__(self, name, getDataFunc, 
        newShape=None, limits=None, ccMap=cc.rainbow, flagAddString=True):
        super(SingleChFloatGenerator, self).__init__(
            name, getDataFunc, newShape, flagAddString)

        self.limits = limits
        self.cMap = convert_colorcet_2_array(ccMap)

    def convert_data_with_limits(self, data, limits):
        return convert_float_array_2_image(data, limits, self.cMap)

    def create_image(self, data):
        if ( self.limits is None ):
            limits = [ data.min(), data.max() ]
        else:
            limits = self.limits

        # Convert the data to RGB image.
        return self.convert_data_with_limits( data, limits )

@register(VISUALIZERS)
class TrueDispGenerator(SingleChFloatGenerator):
    def __init__(self, name, getDataFunc, 
        newShape=None, limits=None, flagAddString=True):
        super(TrueDispGenerator, self).__init__(
            name, getDataFunc, newShape, limits, cc.rainbow, flagAddString)

@register(VISUALIZERS)
class PredDispGenerator(SingleChFloatGenerator):
    def __init__(self, name, getDataFunc, 
        newShape=None, limits=None, flagAddString=True):
        super(PredDispGenerator, self).__init__(
            name, getDataFunc, newShape, limits, cc.rainbow, flagAddString)

    def create_image(self, data):
        '''data contains "trueDisp" and "predDisp". '''
        # Get the true disparity.
        predDisp = data.pop('predDisp')
        trueDisp = data.pop('trueDisp')

        # Get the limits.
        limits = self.limits \
            if self.limits is not None \
            else [ trueDisp.min(), trueDisp.max() ]

        # Statistics.
        diff = np.abs(trueDisp - predDisp)
        s = 'em: %.2f, es: %.2f, l: [%.2f, %.2f]' \
            % ( diff.mean(), diff.std(), limits[0], limits[1] )
        self.append_desc_string( s )

        # Conver the disparity to RGB image.
        return self.convert_data_with_limits( predDisp, limits )

@register(VISUALIZERS)
class PredDispErrorGenerator(SingleChFloatGenerator):
    def __init__(self, name, getDataFunc, 
        newShape=None, limits=None, flagAddString=True):
        super(PredDispErrorGenerator, self).__init__(
            name, getDataFunc, newShape, limits, cc.coolwarm, flagAddString)

    def create_image(self, data):
        '''data contains "trueDisp" and "predDisp". '''
        # Get the true disparity.
        predDisp = data.pop('predDisp')
        trueDisp = data.pop('trueDisp')

        # Statistics.
        diff = trueDisp - predDisp
        absDiff = np.abs(diff)

        # Get the limits.
        limits = self.limits \
            if self.limits is not None \
            else [ diff.min(), diff.max() ]

        # Get the description string.
        s = 'm: %.2f, s: %.2f, l: [%.2f, %.2f]' \
            % ( absDiff.mean(), absDiff.std(), limits[0], limits[1] )
        self.append_desc_string( s )

        # Conver the disparity to RGB image.
        return self.convert_data_with_limits( diff, limits )

@register(VISUALIZERS)
class UncertaintyGenerator(SingleChFloatGenerator):
    def __init__(self, name, getDataFunc, 
        newShape=None, limits=None, flagAddString=True):
        super(UncertaintyGenerator, self).__init__( 
            name, getDataFunc, newShape, limits, cc.CET_L16, flagAddString)
