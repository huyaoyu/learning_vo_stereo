# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-22

import cv2
import numpy as np
import os

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from .preprocessor import resize_bool_ocv, ToTensor_OCV_Dict
from .readers import (
    READERS,
    ImageReader3Channels )

from .register import ( 
    PRE_PROCESSORS, DATASET_LOADERS, 
    register, make_object )

class SingleData(object):
    def __init__(self, name, idx, img0, img1, disp, validMask, occ=None, useOcc=0):
        super(SingleData, self).__init__()

        self.name = name
        self.idx = idx
        self.img0 = img0
        self.img1 = img1
        self.disp0 = disp
        self.validMask = validMask
        self.occ = occ
        self.useOcc = useOcc

        self.disp0Mean = None
        self.disp0Min  = None
        self.disp0Max  = None

        self.update_stat()

    def update_stat(self):
        self.disp0Mean = self.disp0[self.validMask].mean()
        self.disp0Min  = self.disp0[self.validMask].min()
        self.disp0Max  = self.disp0[self.validMask].max()

    def __repr__(self):
        h, w = self.img0.shape[:2]
        occSum = self.occ.sum() if self.occ is not None else 0
        pixels = h*w
        valid  = self.validMask.sum()
        s = '{name}({idx}): img shape ({h}, {w}), {pixels} pix, {valid}({validP:.1f}%) valid, \
occ {occ:.1f}({occP:.1f}%) pixs. \
Disp mean/min/max {dMean:.2f}/{dMin:.2f}/{dMax:.2f}. '.format(
            name=self.name, idx=self.idx, h=h, w=w, 
            pixels=pixels, valid=valid, validP=float(valid)/pixels*100, 
            occ=occSum, occP=float(occSum)/pixels*100, 
            dMean=self.disp0Mean, dMin=self.disp0Min, dMax=self.disp0Max )

        return s

class DatasetRepresentation(object):
    def __init__(self, name, dataRoot, 
        imagesL, imagesR, filesD, 
        filesOcc=None, 
        fb=-1.0, 
        dReader='DispNpy', 
        occReader='OccPng', flagUseOcc=False):
        '''A dataset representation.
        dReader and occReader are classes that can be used as callables.

        Arguments: 
        name (str): Name of the dataset.
        dataRoot (str): Root directory of the files.
        imagesL (list of str): File list of the left images.
        imagesR (list of str): File list of the right images.
        filesD (list of str): File list of the disparity/depth files.
        filesOcc (list of str): File list of the occlusion files.
        fb (float): Focal length * baseline if depth is used. Set negaive value to use disparity.
        dReader (str): The name of the class for reading disparity/depth.
        occReader (str): The name of the class for reading occlusion.
        '''
        super(DatasetRepresentation, self).__init__()

        nImagesL = len(imagesL)

        assert( nImagesL == len(imagesR) == len(filesD) ), \
            'File lists have different length: len(imgagesL) = {}, len(imagesR) = {}, len(filesD) = {}'.format(
                nImagesL, len(imagesR), len(filesD) )
        
        if ( filesOcc is not None ):
            assert( nImagesL == len(filesOcc) ), \
                'Occlusion file list has a different length. len(imagesL) = {}, len(filesOcc) = {}'.format(
                    nImagesL, len(filesOcc) )

        self.imagesL  = imagesL
        self.imagesR  = imagesR
        self.filesD   = filesD
        self.filesOcc = filesOcc
        
        self.dReader   = READERS[ dReader ]( fb )
        self.occReader = READERS[ occReader ]()
        self.imgReader = ImageReader3Channels()

        self.flagUseOcc = flagUseOcc

        self.name = name
        self.dataRoot = dataRoot

    def __len__(self):
        return len( self.imagesL )

    def __getitem__(self, idx):
        assert( 0 <= idx < len(self.imagesL) ), 'Out of range: idx = {}'.format(idx)

        imgL = self.imgReader( os.path.join( self.dataRoot, self.imagesL[idx] ) )
        imgR = self.imgReader( os.path.join( self.dataRoot, self.imagesR[idx] ) )
        disp, validMask = self.dReader( os.path.join( self.dataRoot, self.filesD[idx] ) )
        occ  = self.occReader( os.path.join( self.dataRoot, self.filesOcc[idx] ) ) \
            if self.filesOcc is not None else None

        useOcc = 1 if self.flagUseOcc else 0

        return SingleData( self.name, idx, 
            imgL, imgR, disp, validMask, occ, useOcc )
        
class PreProcessRepresentation(object):
    def __init__(self):
        super(PreProcessRepresentation, self).__init__()

    def __call__(self, sd):
        raise NotImplementedError()

@register(PRE_PROCESSORS, 'allAreaScale')
class AreaScale(PreProcessRepresentation):
    def __init__(self, eq=877.5):
        '''
        eq is computed by sqrt(1024*752).
        '''
        super(AreaScale, self).__init__()

        self.eq = eq # The equivalent boundary length.

    def __call__(self, sd):
        '''
        This pre-processor checks the equivalent boundary length of the input.
        If the equivalent boundary length is longer than the target, then it
        scales the data in sd to match the equivalent boundary length. If the 
        equivalent boundary length is shorter than the target, then nothing happens.

        Arguments: 
        sd (SingleData): A SingleData object.

        Returns:
        The updated sd. The input sd is also updated.
        '''

        # Get the current equivalent boundary length of the input.
        hOri, wOri = sd.img0.shape[:2]
        eqOri = np.sqrt( hOri * wOri )

        # Check the equivalent boundary length. 
        if ( eqOri <= self.eq ):
            return sd

        # Longer equivalent boundary length. 
        f = self.eq / eqOri # The factor.

        # Scale the data.
        sd.img0  = cv2.resize( sd.img0,  (0, 0), fx=f, fy=f, interpolation=cv2.INTER_LINEAR )
        sd.img1  = cv2.resize( sd.img1,  (0, 0), fx=f, fy=f, interpolation=cv2.INTER_LINEAR )
        sd.disp0 = cv2.resize( sd.disp0, (0, 0), fx=f, fy=f, interpolation=cv2.INTER_LINEAR ) * f
        finiteMask = np.isfinite(sd.disp0)
        sd.validMask = np.logical_and( 
            resize_bool_ocv( sd.validMask, ( int(round(wOri*f)), int(round(hOri*f)) ) ),
            finiteMask )
        sd.occ = cv2.resize( sd.occ, (0, 0), fx=f, fy=f, interpolation=cv2.INTER_NEAREST )
        sd.update_stat()

        return sd

@register(PRE_PROCESSORS, 'allBaseCrop')
class BaseCrop(PreProcessRepresentation):
    def __init__(self, base=32):
        super(BaseCrop, self).__init__()

        self.base = int(base)
    
    def __call__(self, sd):
        '''
        Arguments:
        sd (SingleData): A SingleData object.
        
        Returns:
        The updated sd. The input sd will be updated also.
        '''

        # Get the original shape.
        hOri, wOri = sd.img0.shape[:2]

        # Get the new shape.
        hNew = hOri // self.base * self.base
        wNew = wOri // self.base * self.base

        if ( hNew == hOri and wNew == wOri ):
            return sd
        
        h0 = ( hOri - hNew ) // 2
        w0 = ( wOri - wNew ) // 2

        h1 = h0 + hNew # one pass the end.
        w1 = w0 + wNew

        sd.img0      = sd.img0[ h0:h1, w0:w1, ... ]
        sd.img1      = sd.img1[ h0:h1, w0:w1, ... ]
        sd.disp0     = sd.disp0[ h0:h1, w0:w1, ... ]
        sd.validMask = sd.validMask[ h0:h1, w0:w1, ... ]
        if ( sd.occ is not None ):
            sd.occ = sd.occ[ h0:h1, w0:w1, ... ]

        sd.update_stat()

        return sd

@register(PRE_PROCESSORS, 'allGreyRepresentation')
class GrayRepresentation(PreProcessRepresentation):
    def __init__(self):
        super(GrayRepresentation, self).__init__()

    def __call__(self, sd):
        '''sd will get changed.
        '''
        if ( sd.img0.shape.ndim == 2 ):
            sd.img0 = cv2.cvtColor( sd.img0, cv2.COLOR_BGR2GRAY )
            sd.img1 = cv2.cvtColor( sd.img1, cv2.COLOR_BGR2GRAY )
        
        return sd

@register(PRE_PROCESSORS, 'allConvert2Dict')
class Convert2Dict(PreProcessRepresentation):
    def __init__(self):
        super(Convert2Dict, self).__init__()

    def __call__(self, sd):
        '''Must be used before the preprocessing defined in ../PreProcess.py. '''
        d = { 'img0': sd.img0, 'img1': sd.img1, 
              'disp0': sd.disp0, 
              'valid0': sd.validMask }

        if ( sd.occ is not None ):
            d['occ0'] = sd.occ
        
        d['useOcc0'] = sd.useOcc

        return d

def compose_pp_transforms(dicts):
    if ( dicts is None or len(dicts) == 0):
        return transforms.Compose([ ToTensor_OCV_Dict() ])

    transList = []
    for ppd in dicts:
        transList.append( make_object(PRE_PROCESSORS, ppd) )
    return transforms.Compose( transList )

class AssembleDataset(data.Dataset):
    def __init__(self, ppConfs=None):
        super(AssembleDataset, self).__init__()

        self.nDatasets = 0 # The number of datasets.
        self.names = [] # The names of the datasets.
        self.sizes = [] # The sizes of the datasets.

        # The pre-processor.
        self.pp = compose_pp_transforms(ppConfs)

    def set_pp(self, composedPP):
        self.pp = composedPP # Not a deep copy.

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()

@register(DATASET_LOADERS)
class PreloadedAssemble(AssembleDataset):
    @classmethod
    def get_default_init_args(cls):
        return dict(
            type=cls.__name__, 
            ppConfs=[
                dict( type='ToTensor_OCV_Dict' ) ] )

    def __init__(self, ppConfs=None):
        super(PreloadedAssemble, self).__init__(ppConfs)

        self.data = [] # Holds all the data.

    def preload(self, reps):
        '''Pre-load all the data.
        Arguments: 
        reps (list of DatasetRepresentation): A collection of DatasetRepresentation objecst.
        '''

        self.names = []
        self.sizes = []
        self.data = []
        for rep in reps:
            self.names.append(rep.name)
            s = len(rep)
            self.sizes.append(s)
            for i in range(s):
                sd = rep[i]
                if ( self.pp is not None ):
                    sd = self.pp(sd)
                self.data.append( sd )

        self.nDatasets = len(reps)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = len(self.data)

        if ( 0 <= idx < s ):
            return self.data[idx]
        elif ( idx == s ): 
            raise StopIteration()
        else:
            raise Exception('Wrong idx {}, len(self.data) = {}'.format(idx, s))