# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-04-08

# System packages.
import cv2
import numpy as np
import colorcet as cc

from .color_map_utils import convert_colorcet_2_array

def add_string_line_2_img(img, s, hRatio, bgColor=(70, 30, 10), fgColor=(0,255,255), thickness=1):
    """
    Add a string on top of img.
    s: The string. Only supports 1 line string.
    hRatio: The non-dimensional height ratio for the font.
    """

    assert( hRatio < 1 and hRatio > 0 )

    H, W = img.shape[:2]

    # The text size of the string in base font.
    strSizeOri, baselineOri = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1)

    # The desired height.
    hDesired = np.ceil( hRatio * H )
    strScale = hDesired / strSizeOri[1]

    # The actual font size.
    strSize, baseline = \
        cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, strScale, thickness )

    fontSize, baseLineFont = \
        cv2.getTextSize("a", cv2.FONT_HERSHEY_SIMPLEX, strScale, thickness )

    # Draw the box.
    hBox = strSize[1] + fontSize[1]
    wBox = strSize[0] + 2 * fontSize[0]

    img = img.copy()
    pts = np.array([[0,0],[wBox,0],[wBox,hBox],[0,hBox]],np.int32)
    cv2.fillConvexPoly( img, pts, color=bgColor )
    cv2.putText(img, s, (fontSize[0], int(hBox-fontSize[1]/2.0)),
        cv2.FONT_HERSHEY_SIMPLEX, strScale, color=fgColor, thickness=thickness)

    return img

def s2t(img):
    '''
    Convert single (s) channel image to (2) three (t) channel image.
    '''

    if ( img.ndim == 2 ):
        return np.stack( ( img, img, img ), axis=2 )
    elif ( img.ndim == 3 and img.shape[2] == 1 ):
        return np.concatenate( (img, img, img), axis=2 )
    else:
        return img

def convert_mask_2_image(mask):
    img = np.zeros_like(mask, dtype=np.uint8) + 255
    m = mask == 1
    img[m] = 0

    return np.stack((img, img, img), axis=2)

def convert_float_array_2_image(array, limits, cMap):
    '''
    array (NumPy array): A float single channel array.
    limits (2-element): [ min, max ] limits.
    cMap (NumPy array): N x 3 color map.

    This function convert a single floating point array into
    a RGB representation. array is first clipped by referring to
    limits. Then a RGB image is created by interplating into
    cMap.
    '''

    assert( array.ndim == 2 )

    array = array.astype(np.float32)
    array = np.clip( array, limits[0], limits[1] )

    array = array - limits[0]
    array = array / ( limits[1] - limits[0] )

    n = cMap.shape[0]
    cMap = cMap.reshape(( 1, n, -1 ))

    x = array * (n-1)
    y = np.zeros_like(array)

    img = cv2.remap( cMap, x, y, interpolation=cv2.INTER_LINEAR )

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)