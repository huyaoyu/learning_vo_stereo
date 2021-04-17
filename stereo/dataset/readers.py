# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-22

import cv2
import numpy as np
import os
import re

from .register import ( FILE_READERS, register_manually )

def test_file(file):
    if ( not os.path.isfile(file) ):
        raise Exception("%s does not exist. " % (file))

# readPFM() is copied from 
# # https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html#downloads
def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

class Reader(object):
    def __init__(self, invalidValue=None):
        super(Reader, self).__init__()

        self.invalidValue = invalidValue

    def get_finite_valid_mask(self, v):
        mask = np.isfinite(v)
        if ( self.invalidValue is not None ):
            mask = np.logical_and( 
                v != self.invalidValue,
                mask )
        return mask

    def read(self, fn):
        '''Child class is supposed to implement this function.
        '''
        raise NotImplementedError()

    def __call__(self, fn):
        return self.read(fn)

class DisparityReader(Reader):
    def __init__(self, fb=-1.0, invalidValue=None):
        super(DisparityReader, self).__init__(invalidValue)

        self.fb = fb

    def set_invalid_2_min_inplace(self, disp, validMask):
        m = disp[validMask].min()
        invalidMask = np.logical_not(validMask)
        disp[invalidMask] = m

class DisparityNpyReader(DisparityReader):
    def __init__(self, fb=-1.0, invalidValue=None):
        super(DisparityNpyReader, self).__init__(fb, invalidValue)

    def read(self, fn):
        test_file(fn)
        d = np.load(fn).astype(np.float32)
        mask = self.get_finite_valid_mask(d)
        self.set_invalid_2_min_inplace(d, mask)
        return d, mask

class DisparityPngReader(DisparityReader):
    def __init__(self, fb=-1.0, invalidValue=None):
        super(DisparityPngReader, self).__init__(fb, invalidValue)

    def read(self, fn):
        test_file(fn)
        d = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
        assert(d.dtype == np.uint8)
        assert(d.shape[2] == 4), '{} has a shape of {}, not suitable for disparity/depth conversion. '.format(fn, d.shape)
        d = d.view('<f4')
        d = np.squeeze(d, axis=-1)
        mask = self.get_finite_valid_mask(d)
        self.set_invalid_2_min_inplace(d, mask)
        return d, mask

class DisparityPfmReader(DisparityReader):
    def __init__(self, fb=-1.0, invalidValue=None):
        super(DisparityPfmReader, self).__init__(fb, invalidValue)

    def read(self, fn):
        test_file(fn)
        disp, scale = readPFM(fn)
        disp = disp.astype(np.float32)
        mask = self.get_finite_valid_mask(disp)
        self.set_invalid_2_min_inplace(disp, mask)
        return disp, mask

class DisparityKittiReader(DisparityReader):
    def __init__(self, fb=-1.0):
        super(DisparityKittiReader, self).__init__(fb, invalidValue=0)

    def read(self, fn):
        test_file(fn)
        d = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
        assert(d.dtype == np.uint16)
        mask = self.get_finite_valid_mask(d)
        d = d.astype(np.float32) / 256
        self.set_invalid_2_min_inplace(d, mask)
        return d, mask

class DepthNpyReader(DisparityReader):
    def __init__(self, fb=-1.0):
        super(DepthNpyReader, self).__init__(fb, invalidValue=0)

        assert(self.fb > 0), 'self.fb = {}'.format(self.fb)

    def read(self, fn):
        test_file(fn)
        d = np.load(fn).astype(np.float32)
        mask = self.get_finite_valid_mask(d)
        invalidMask = np.logical_not(mask)
        d[invalidMask] = 1
        return self.fb / d, mask

class DepthPngReader(DisparityReader):
    def __init__(self, fb=-1.0):
        super(DepthPngReader, self).__init__(fb, invalidValue=0)

        assert(self.fb > 0), 'self.fb = {}'.format(self.fb)

    def read(self, fn):
        test_file(fn)
        d = cv2.imread(fn, cv2.IMREAD_UNCHANGED)

        assert(d.dtype == np.uint8)
        assert(d.shape[2] == 4), '{} has a shape of {}, not suitable for disparity/depth conversion. '.format(fn, d.shape)

        d = d.view('<f4')
        d = np.squeeze(d, axis=-1)

        mask = self.get_finite_valid_mask(d)
        invalidMask = np.logical_not(mask)
        d[invalidMask] = 1

        return self.fb / d, mask

class OccReader(Reader):
    def __init__(self, validValue=255, outOfFOVValue=11):
        super(OccReader, self).__init__()

        self.validValue    = validValue
        self.outOfFOVValue = outOfFOVValue

    def mask_raw(self, rawValue):
        # Non-occlusion pixel is 255, out-of-view pixel is 11.
        maskOcc   = rawValue != self.validValue
        maskInFOV = rawValue != self.outOfFOVValue
        return np.logical_and( maskOcc, maskInFOV )

class OccNpyReader(OccReader):
    def __init__(self):
        super(OccNpyReader, self).__init__()

    def read(self, fn):
        test_file(fn)
        mask = np.load(fn).astype(np.uint8)
        mask = self.mask_raw(mask)
        return mask.astype(np.float32)

class OccPngReader(OccReader):
    def __init__(self):
        super(OccPngReader, self).__init__()

    def read(self, fn):
        test_file(fn)
        mask = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
        mask = self.mask_raw(mask)
        return mask.astype(np.float32)

class OccMiddReader(OccReader):
    def __init__(self):
        super(OccMiddReader, self).__init__(validValue=255, outOfFOVValue=128)

    def read(self, fn):
        test_file(fn)
        mask = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
        mask = self.mask_raw(mask)
        return mask.astype(np.float32)

class ValidMaskReader(Reader):
    def __init__(self, validValue=255):
        super(ValidMaskReader, self).__init__()

        self.validValue = validValue

    def read(self, fn):
        test_file(fn)
        mask = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
        mask = mask == self.validValue
        return mask

class ImageReader(Reader):
    def __init__(self, flagFloat=False):
        super(ImageReader, self).__init__()

        self.flagFloat = flagFloat

class ImageReaderPlain(ImageReader):
    def __init__(self, flagFloat=False):
        super(ImageReaderPlain, self).__init__(flagFloat)

    def read(self, fn):
        test_file(fn)
        img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
        if ( self.flagFloat ):
            return img.astype(np.float32)
        else:
            return img

class ImageReader3Channels(ImageReader):
    def __init__(self, flagFloat=False):
        super(ImageReader3Channels, self).__init__(flagFloat)

    def read(self, fn):
        test_file(fn)
        img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
        if ( img.ndim == 2 ):
            # img = np.tile( img.reshape( ( *img.shape, 1 ) ), 3 ) # Cause issues with Python2.7
            img = np.tile( img.reshape( ( img.shape[0], img.shap[1], 1 ) ), 3 )

        if ( self.flagFloat ):
            return img.astype(np.float32)
        else:
            return img

READERS = {
    'DispNpy'  : DisparityNpyReader,
    'DispPng'  : DisparityPngReader,
    'DispPfm'  : DisparityPfmReader,
    'DispKitti': DisparityKittiReader,
    'DepthNpy' : DepthNpyReader,
    'DepthPng' : DepthPngReader,
    'OccNpy'   : OccNpyReader,
    'OccPng'   : OccPngReader,
    'OccMidd'  : OccMiddReader,
    'VMask'    : ValidMaskReader,
    'ImgPlain' : ImageReaderPlain,
    'Img3Ch'   : ImageReader3Channels,
}

for k, v in READERS.items():
    register_manually( FILE_READERS, v, k )