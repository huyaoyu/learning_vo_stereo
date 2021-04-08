# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-28

# System packages.
import cv2
import numpy as np
import colorcet as cc

from .color_map_utils import convert_colorcet_2_array
from .image_creation import ( 
    add_string_line_2_img, s2t, convert_float_array_2_image )

def diff_statistics(diff):
    diff = np.abs(diff)

    return diff.mean(), diff.std()

def add_diff_string_2_disparity_map(trueDisp, predDisp, predDispMap):
    diff = trueDisp - predDisp
    stat = diff_statistics(diff)

    s = 'mean = %f, std = %f' % ( stat[0], stat[1] )
    predDispMap = add_string_line_2_img( predDispMap, s, 0.03 )

    return predDispMap

def compose_results( img0, img1, \
    trueDispList, predDispList ):
    '''
    img0 (NumPy array): OpenCV image.
    img1 (NumPy array): OpenCV image.
    trueDispList (list of NumPy arrays): Disparity maps, single channel.
    predDispList (list of NumPy arrays): Disparity maps, single channel.
   
    It is assumed that the number of elements in the lists are the same. 
    '''
    nDisps = len(trueDispList)
    assert(nDisps > 0), f'No disp specified. '

    # Dimensions.
    H, W = img0.shape[:2]
    cH   = 2 * H + (nDisps - 1) * H # Canvas height.
    cW   = 2 * W                    # Canvas width.

    # Create the canvas.
    canvas = np.zeros( ( cH, cW, 3 ), dtype=np.uint8 )

    # Insert images to the canvas.
    canvas[0:H,   0:W] = s2t(img0)
    canvas[H:2*H, 0:W] = s2t(img1)

    # Get the first true and predicted disparities.
    trueDisp0 = trueDispList[0]
    predDisp0 = predDispList[0]

    # Convert the full scale disparities.
    dispCMap = convert_colorcet_2_array(cc.rainbow)
    limits   = [ trueDisp0.min(), predDisp0.max() ]
    
    trueDisp0Img = convert_float_array_2_image( trueDisp0, limits, dispCMap )
    predDisp0Img = convert_float_array_2_image( predDisp0, limits, dispCMap )

    predDisp0Img = add_diff_string_2_disparity_map( trueDisp0, predDisp0, predDisp0Img )

    # Insert the full scale dispariteis.
    canvas[0:H,   W:2*W] = trueDisp0Img
    canvas[H:2*H, W:2*W] = predDisp0Img

    # The offset in the canvas.
    trueYOffset = 2*H
    
    for i in range( 1, nDisps ):
        trueDisp = trueDispList[i]
        predDisp = predDispList[i]

        limits = [ trueDisp.min(), trueDisp.max() ]
        trueDispImg = convert_float_array_2_image(trueDisp, limits, dispCMap)
        predDispImg = convert_float_array_2_image(predDisp, limits, dispCMap)
        
        predDispImg = add_diff_string_2_disparity_map( trueDisp, predDisp, predDispImg )

        trueY = trueYOffset + (i-1) * H
        predY = trueY

        canvas[ trueY:(trueY+H), 0:W ] = trueDispImg
        canvas[ predY:(predY+H), W:  ] = predDispImg

    return canvas