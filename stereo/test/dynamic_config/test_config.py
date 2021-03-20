# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-20

# Prepare the Python environment.
import os
_CF       = os.path.realpath(__file__)
_PKG_PATH = os.path.dirname(
                os.path.dirname( 
                    os.path.dirname( 
                        os.path.dirname( _CF ) ) ) )

print(f'Adding {_PKG_PATH} to the package search path. ')

import sys
sys.path.insert(0, _PKG_PATH)

# Import the package.
import stereo

if __name__ == '__main__':
    print(f'Hello, {os.path.basename(__file__)}! ')

    # Create a dummy dict.
    d = dict( 
        featureExtractor=dict(
            type='MultiLevelFeatureExtractor', 
            stages=2,
            channels=[[3, 32], [32, 64]] ), 
        costVolume=dict(
            maxDisparity=192,
            channels=[[256, 512], [512, 256]]))

    stereo.dynamic_config.config.show_config(d)