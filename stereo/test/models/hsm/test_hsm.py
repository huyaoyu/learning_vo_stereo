# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-20

# Prepare the Python environment.
import os
_CF       = os.path.realpath(__file__)
_PKG_PATH = \
os.path.dirname(
    os.path.dirname(
        os.path.dirname( 
            os.path.dirname( 
                os.path.dirname( _CF ) ) ) ) )

print(f'Adding {_PKG_PATH} to the package search path. ')

import sys
sys.path.insert(0, _PKG_PATH)

# Import the package.
import stereo
from stereo.models.globals import GLOBAL

# Configure the global settings for testing.
# Not the right setting. Only for testing.
GLOBAL.torch_align_corners(True)

if __name__ == '__main__':
    print(f'Hello, {os.path.basename(__file__)}! ')

    # Show the global settings.
    print(f'GLOBAL.torch_align_corners() = {GLOBAL.torch_align_corners()}')

    # The dummy dictionary.
    maxDisp = 192
    d = dict(
        type='HSMNet',
        maxdisp=maxDisp, 
        clean=-1, 
        level=1,
        featExtConfig=dict(
            type='UNet',
            initialChannels=32 ),
        costVolConfig=dict(
            type='CVDiff',
            refIsRight=False),
        dispRegConfigs=[
            dict(
                type='disparityregression',
                maxDisp=maxDisp,
                divisor=16 ), 
            dict(
                type='disparityregression',
                maxDisp=maxDisp,
                divisor=16 ),
            dict(
                type='disparityregression',
                maxDisp=maxDisp,
                divisor=32 ),
            dict(
                type='disparityregression',
                maxDisp=maxDisp,
                divisor=64 ),
            ] )

    # Make a HSMNet object.
    hsm = stereo.models.model_factory.make_model(d)
    hsm.initialize()

    # Make a HSMNet with default settings.
    hsm = stereo.models.model_factory.make_model(dict(type='HSMNet'))
    hsm.initialize()

    # Show the model
    # print(hsm)