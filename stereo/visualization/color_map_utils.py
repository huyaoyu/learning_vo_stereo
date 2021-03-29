# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-28

import cv2
import numpy as np

def hex_to_RGB(hex):
    ''' "#FFFFFF" -> [255,255,255] '''
    # Pass 16 to the integer function for change of base
    return [int(hex[i:i+2], 16) for i in range(1,6,2)]

def convert_colorcet_2_array(ccArray):
    cmap = np.zeros((256, 3), dtype=np.uint8)

    for i in range(len(ccArray)):
        rgb = hex_to_RGB( ccArray[i] )
        cmap[i, :] = rgb

    return cmap

def make_color_bar(shape, cMap):
    '''
    shape (NumPy array): H, W.
    cMap (NumPy array): N x 3 color map.
    '''

    H, W = shape[:]

    x = np.linspace(0, cMap.shape[0], W, endpoint=False, dtype=np.float32)
    y = np.zeros( (H,), dtype=np.float32 )

    xx, yy = np.meshgrid(x, y)

    cMap = cMap.reshape( (-1, cMap.shape[0], cMap.shape[1]) )
    img = cv2.remap( cMap, xx, yy, interpolation=cv2.INTER_LINEAR )

    return img

if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    import colorcet as cc

    cmapDisp = convert_colorcet_2_array(cc.rainbow)
    cmapDiff = convert_colorcet_2_array(cc.coolwarm)

    H = 64
    W = 512

    imgCMapDisp = make_color_bar( (H, W), cmapDisp )
    cv2.imwrite( 'DispCMap.png', cv2.cvtColor( imgCMapDisp, cv2.COLOR_RGB2BGR ) )

    imgCMapDiff = make_color_bar( (H, W), cmapDiff )
    cv2.imwrite( 'DiffCMap.png', cv2.cvtColor( imgCMapDiff, cv2.COLOR_RGB2BGR ) )

    imgStacked = np.concatenate( ( imgCMapDisp, imgCMapDiff ), axis=0 )

    plt.imshow(imgStacked)

    plt.show()
