# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-22

# Some of the content of this file is copied from PSMNet.
# https://github.com/JiaRenChang/PSMNet

# System packages.
import copy
import cv2
import numpy as np
import os

# PyTorch packages.
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

# Local installed packages.
from CommonPython.Filesystem.Filesystem import get_filename_parts

# Local packages.
from .readers import (
    test_file,
    ImageReaderPlain,
    DisparityPfmReader,
    OccPngReader,
    ValidMaskReader )
from .preprocessor import ToTensor_OCV_Dict

from .register import ( 
    FILE_READERS, PRE_PROCESSORS, DATASET_LOADERS,
    register, make_object )

def compose_pp_transforms(dicts):
    if ( dicts is None or len(dicts) == 0):
        return transforms.Compose([ ToTensor_OCV_Dict() ])

    transList = []
    for ppd in dicts:
        transList.append( make_object(PRE_PROCESSORS, ppd) )
    return transforms.Compose( transList )

@register(DATASET_LOADERS)
class FileListLoader(data.Dataset):
    @classmethod
    def get_default_init_args(cls, training):
        return dict(
            type=cls.__name__,
            flagUseOcc=False, 
            ppConfs=None, 
            training=training)

    def __init__(self, 
        flagUseOcc=False,
        ppConfs=None,
        training=True ):
        super(FileListLoader, self).__init__()

        self.left      = None
        self.right     = None
        self.disp_L    = None
        self.occ_L     = None
        self.valid_L   = None # The valid mask file list.
        
        self.flagUseOcc = flagUseOcc

        self.imgReader  = ImageReaderPlain()
        self.dispReader = DisparityPfmReader()
        self.occReader  = OccPngReader()
        self.validMaskReader = ValidMaskReader()
        if ( ppConfs is None ):
            self.preprocessor = transforms.Compose([ToTensor_OCV_Dict()])
        else:
            self.preprocessor = compose_pp_transforms(ppConfs)

        self.training  = training

    def set_occ_L(self, occL):
        self.occ_L = occL

    def set_valid_mask(self, validL):
        self.valid_L = validL

    def set_lists(self, left, right, dispL, occL=None, validL=None):
        self.left    = left
        self.right   = right
        self.disp_L  = dispL
        self.occ_L   = occL
        self.valid_L = validL

    def set_readers(self, 
        imgReader=None, dispReader=None,
        occReader=None, validMaskReader=None ):
        if ( imgReader is not None ):
            self.imgReader = imgReader

        if ( dispReader is not None ):
            self.dispReader = dispReader

        if ( occReader is not None ):
            self.occReader = occReader

        if ( validMaskReader is not None ):
            self.validMaskReader = validMaskReader

    def check_3_channels(self, img):
        if ( img.ndim == 2 ):
            return img

        return img[:, :, :3]

    def map_index(self, idx):
        '''The child class is expected to overload this function.
        '''
        return idx

    def __getitem__(self, index):
        index = self.map_index(index)
        
        left   = self.left[index]
        right  = self.right[index]
        disp_L = self.disp_L[index]

        imgL = self.imgReader(left)
        imgR = self.imgReader(right)

        imgL = self.check_3_channels( imgL )
        imgR = self.check_3_channels( imgR )

        dispL, mask = self.dispReader(disp_L)

        # Dictionary.
        d = {"img0": imgL, "img1": imgR, "disp0": dispL}

        if ( self.occ_L is not None ):
            occ_L = self.occ_L[index]
            occL  = self.occReader(occ_L)
            d['occ0'] = occL

        d['useOcc0'] = 1.0 if self.flagUseOcc else 0.0

        if ( self.valid_L is not None ):
            valid_L = self.valid_L[index]
            validL  = self.validMaskReader(valid_L)
            d['valid0'] = np.logical_and( validL, mask )
        else:
            d['valid0'] = mask

        # Image pre-processing.
        if ( self.preprocessor is not None ):
            d = self.preprocessor(d)

        return d

    def __len__(self):
        return len(self.left)
