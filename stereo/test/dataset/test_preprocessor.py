# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-28

# Prepare the Python environment.
import os
_CF = os.path.realpath(__file__)

_PKG_PATH = _CF
for _ in range(4):
    _PKG_PATH = os.path.dirname( _PKG_PATH )

print(f'Adding {_PKG_PATH} to the package search path. ')

import sys
sys.path.insert(0, _PKG_PATH)

import numpy as np

# PyTorch.
import torch
from torchvision.transforms import Compose
from torchvision.transforms.functional import to_pil_image

# Local installed packages.
from CommonPython.ImageIO.ImageWrite import write_image, write_float_image_normalized

# Import the package.
import stereo
from stereo.dataset import readers
from stereo.dataset import preprocessor

def get_data_sample():
    '''Hard coded data sample'''
    imgReader  = readers.ImageReaderPlain()
    dispReader = readers.DisparityPfmReader()
    occReader  = readers.OccPngReader()

    dirRoot = os.path.join( _PKG_PATH, 'stereo', 'test', 'sample_data', 'scene_flow' )

    disp, mask = dispReader( os.path.join( dirRoot, '0006.pfm' ) )

    return dict(
        img0=imgReader( os.path.join( dirRoot, '0006_L.png' ) ),
        img1=imgReader( os.path.join( dirRoot, '0006_R.png' ) ),
        disp0=disp,
        occ0=occReader( os.path.join( dirRoot, '0006_LOcc.png' ) ),
        useOcc0=1.0, 
        valid0=mask )

def convert_tensor_img_2_array(t):
    return np.array( to_pil_image(t) )

def write_agumented(d):
    path = os.path.join( _PKG_PATH, 'stereo', 'test', 'sample_data', 'scene_flow', 'augmented')

    write_image( os.path.join( path, 'img0.png' ), 
        convert_tensor_img_2_array( d['img0'] ) )
    write_image( os.path.join( path, 'img1.png' ), 
        convert_tensor_img_2_array( d['img1'] ) )
    write_float_image_normalized( os.path.join( path, 'disp0.png' ), 
        d['disp0'].squeeze(0).numpy() )
    write_image( os.path.join( path, 'occ0.png' ), 
        convert_tensor_img_2_array( d['occ0'] ) )

if __name__ == '__main__':
    print(f'Hello, {os.path.basename(__file__)}! ')

    # Read a data sample.
    sample = get_data_sample()

    # Compose the transform.
    trans = Compose([
        preprocessor.RecOcclusion_OCV_Dict(flagRandom=False),
        preprocessor.ToTensor_OCV_Dict(),
        preprocessor.ColorJitter_Dict( 
            brightness=(0.5, 1.5), contrast=(0.5, 1.5), 
            saturation=(0.5, 1.5), hue=(-0.5, 0.5),
            flagRandom=False ),
        preprocessor.AdjustGamma_Dict(
            gammaLimits=(0.5, 1.5), 
            flagRandom=False),
    ])

    transformed = trans( sample )

    # Write the transformed image.
    write_agumented( transformed )