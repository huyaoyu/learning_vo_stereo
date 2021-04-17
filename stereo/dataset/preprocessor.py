# -*- coding: future_fstrings -*-

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-22

# Some of the content of this file is copied from PSMNet.
# https://github.com/JiaRenChang/PSMNet

# System packages.
import cv2
import numpy as np
import random

# PyTorch.
import torch
import torchvision.transforms as transforms

# Local packages.
from .register import ( PRE_PROCESSORS, register )

imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

imagenet_stats_bgr = {
    'mean': [0.406, 0.456, 0.485], 
    'std':  [0.225, 0.224, 0.229] }

def resize_bool_ocv(img, newSize):
    '''
    Arguments: 
    img (NumPy array): Boolean image.
    newSize (2-element): ( W, H ).

    Returns:
    The resized boolean image as a NumPy array.
    '''

    assert( img.dtype == np.bool ), 'Wrong type {}, expect {}. '.format(img.dtype, np.bool)

    temp = img.copy().astype(np.float32)
    temp[np.logical_not(img)] = np.inf
    temp = cv2.resize( temp, newSize, interpolation=cv2.INTER_LINEAR )
    return np.isfinite(temp)

def resize_bool_ocv_f(img, f):
    '''Convenient version for resize_bool_ocv(). 
    Arguments: 
    img (NumPy array): Boolean image.
    f (float): Scale factor.

    Returns:
    The resized boolean image as a NumPy array.
    '''

    # Get the original size.
    H, W = img.shape[:2]
    return resize_bool_ocv( img, ( int(round(W*f)), int(round(H*f)) ) )

@register(PRE_PROCESSORS)
class ToTensor_OCV_Dict(object):
    def __init__(self):
        super(ToTensor_OCV_Dict, self).__init__()

        self.toTensor = transforms.ToTensor()

    def __call__(self, d):
        d["img0"]  = self.toTensor(d["img0"])
        d["img1"]  = self.toTensor(d["img1"])
        d["disp0"] = self.toTensor(d["disp0"])

        if ( "occ0" in d ):
            d["occ0"] = self.toTensor(d["occ0"])
            d['useOcc0'] = torch.Tensor( [ d['useOcc0'], ] ).view((1, 1))

        if ( 'valid0' in d ):
            d['valid0'] = self.toTensor( d['valid0'] )

        return d

class Float_OCV(object):
    def __call__(self, x):
        return x.astype(np.float32)

class NormalizeRGB_OCV(object):
    def __init__(self, s):
        super(NormalizeRGB_OCV, self).__init__()
        
        self.s = s

    def __call__(self, x):
        """This is the OpenCV version. The order of the color channle is BGR. The order of dimension is HWC."""

        x = x.astype(np.float32) / self.s

        # It is assumed that the data type of x is already floating point number.

        x[:, :, 0] = ( x[:, :, 0] - imagenet_stats["mean"][2] ) / imagenet_stats["std"][2]
        x[:, :, 1] = ( x[:, :, 1] - imagenet_stats["mean"][1] ) / imagenet_stats["std"][1]
        x[:, :, 2] = ( x[:, :, 2] - imagenet_stats["mean"][0] ) / imagenet_stats["std"][0]

        return x
    
    def denormalize(self, x):
        x[:, :, 0] = ( x[:, :, 0] * imagenet_stats["std"][2] ) + imagenet_stats["mean"][2]
        x[:, :, 1] = ( x[:, :, 1] * imagenet_stats["std"][1] ) + imagenet_stats["mean"][1]
        x[:, :, 2] = ( x[:, :, 2] * imagenet_stats["std"][0] ) + imagenet_stats["mean"][0]

        return x * self.s

@register(PRE_PROCESSORS)
class NormalizeRGB_OCV_Dict(object):
    def __init__(self, s):
        super(NormalizeRGB_OCV_Dict, self).__init__()

        self.s = s

        self.normalizer = NormalizeRGB_OCV(self.s)

    def __call__(self, d):
        d["img0"] = self.normalizer(d["img0"])
        d["img1"] = self.normalizer(d["img1"])

        return d

@register(PRE_PROCESSORS)
class NormalizeRGB_Dict(object):
    def __init__(self, 
        mean=imagenet_stats_bgr['mean'], 
        std=imagenet_stats_bgr['std']):
        super(NormalizeRGB_Dict, self).__init__()

        self.normalizer = transforms.Normalize(mean, std, inplace=False)

    def __call__(self, d):
        d['img0'] = self.normalizer( d['img0'])
        d['img1'] = self.normalizer( d['img1'])
        return d

class NormalizeGray_OCV_Naive(object):
    def __init__(self, s, a):
        super(NormalizeGray_OCV_Naive, self).__init__()
        self.s = s
        self.a = a

    def __call__(self, x):
        return x.astype(np.float32) / self.s - self.a
    
    def denormalize(self, x):
        return ( x + self.a ) * self.s

@register(PRE_PROCESSORS)
class NormalizeGray_OCV_Naive_Dict(object):
    def __init__(self, s, a):
        super(NormalizeGray_OCV_Naive_Dict, self).__init__()
        self.s = s
        self.a = a

        self.normalizer = NormalizeGray_OCV_Naive(self.s, self.a)

    def __call__(self, d):
        d["img0"] = self.normalizer(d["img0"])
        d["img1"] = self.normalizer(d["img1"])

        return d

class Resize_OCV(object):
    def __init__(self, h, w):
        super(Resize_OCV, self).__init__()
        
        self.h = h # The new height.
        self.w = w # The new width.

    def __call__(self, x):
        # Assuming an OpenCV image.
        return cv2.resize(x, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

class ResizeDisparity_OCV(object):
    def __init__(self, h, w):
        super(ResizeDisparity_OCV, self).__init__()
        
        self.h = h # The new height.
        self.w = w # The new width.

    def __call__(self, x):
        # Assuming an OpenCV image with float data type.
        # The factor.
        f = 1.0 * self.w / x.shape[1]

        return cv2.resize(x, (self.w, self.h), interpolation=cv2.INTER_LINEAR) * f

class ResizeOcclusion_OCV(object):
    def __init__(self, h, w):
        super(ResizeOcclusion_OCV, self).__init__()
        # The new height and width.
        self.h = h
        self.w = w

    def __call__(self, x):
        # Assuming an OpenCV image with float data type.
        return cv2.resize( x, ( self.w, self.h ), interpolation=cv2.INTER_NEAREST )

class ResizeBool_OCV(object):
    def __init__(self, h, w):
        super(ResizeBool_OCV, self).__init__()
        # The new height and width.
        self.h = h
        self.w = w

    def __call__(self, x):
        # Assuming an OpenCV image with bool data type.
        return resize_bool_ocv(x, ( self.w, self.h ))

@register(PRE_PROCESSORS)
class Resize_OCV_Dict(object):
    def __init__(self, h, w):
        super(Resize_OCV_Dict, self).__init__()

        self.h = h
        self.w = w

        self.imgResizer  = Resize_OCV(self.h, self.w)
        self.dispResizer = ResizeDisparity_OCV(self.h, self.w)
        self.occResizer  = ResizeOcclusion_OCV(self.h, self.w)
        self.boolResizer = ResizeBool_OCV(self.h, self.w)

    def __call__(self, d):
        if ( 0 == self.h or 0 == self.w ):
            return d

        img0  = d["img0"]
        img1  = d["img1"]
        disp0 = d["disp0"]

        d["img0"]  = self.imgResizer(img0)
        d["img1"]  = self.imgResizer(img1)
        d["disp0"] = self.dispResizer(disp0)
        
        if ( 'occ0' in d ):
            d['occ0'] = self.occResizer(d['occ0'])
            # No need to resize useOcc0.

        if ( 'valid0' in d ):
            finiteMask = np.isfinite(d['disp0'])
            d['valid0'] = np.logical_and( 
                self.boolResizer( d['valid0'] ), 
                finiteMask )

        return d

@register(PRE_PROCESSORS)
class RandomCropSized_OCV_Dict(object):
    def __init__(self, h, w):
        super(RandomCropSized_OCV_Dict, self).__init__()

        self.h = h
        self.w = w

    def __call__(self, d):
        if ( 0 == self.h or 0 == self.w ):
            return d
        
        img0  = d["img0"]
        img1  = d["img1"]
        disp0 = d["disp0"]

        # Allowed indices.
        ah = img0.shape[0] - self.h
        aw = img0.shape[1] - self.w

        if ( ah < 0 ):
            raise Exception("img0.shape[0] < self.h. img0.shape[0] = {}, self.h = {}. ".format( img0.shape[0], self.h ))

        if ( aw < 0 ):
            raise Exception("img0.shape[1] < self.w. img0.shape[1] = {}, self.w = {}. ".format( img0.shape[1], self.w ))

        # Get two random numbers.
        ah = ah + 1
        aw = aw + 1

        # ch = np.random.randit( 0, ah, 1 ).item()
        # cw = np.random.randit( 0, aw, 1 ).item()
        ch = torch.randint(0, ah, (1, )).item() # Random with better seed.
        cw = torch.randint(0, aw, (1, )).item()

        # # Test use!
        # ch = ah - 1
        # cw = aw - 1

        d["img0"]  = img0[ch:ch+self.h, cw:cw+self.w]
        d["img1"]  = img1[ch:ch+self.h, cw:cw+self.w]
        d["disp0"] = disp0[ch:ch+self.h, cw:cw+self.w]

        if ( "occ0" in d ):
            occ0 = d["occ0"]
            d["occ0"] = occ0[ch:ch+self.h, cw:cw+self.w]

        if ( 'valid0' in d ):
            valid0 = d['valid0']
            d['valid0'] = valid0[ch:ch+self.h, cw:cw+self.w]

        return d

@register(PRE_PROCESSORS)
class CenterCropSized_OCV_Dict(object):
    def __init__(self, h, w):
        super(CenterCropSized_OCV_Dict, self).__init__()

        self.h = h
        self.w = w

    def __call__(self, d):
        if ( 0 == self.h or 0 == self.w ):
            return d

        img0  = d["img0"]
        img1  = d["img1"]
        disp0 = d["disp0"]

        # Allowed indices.
        ah = img0.shape[0] - self.h
        aw = img0.shape[1] - self.w

        if ( ah < 0 ):
            raise Exception("img0.shape[0] < self.h. img0.shape[0] = {}, self.h = {}. ".format( img0.shape[0], self.h ))

        if ( aw < 0 ):
            raise Exception("img0.shape[1] < self.w. img0.shape[1] = {}, self.w = {}. ".format( img0.shape[1], self.w ))

        # Get two random numbers.
        ah = ah + 1
        aw = aw + 1

        ch = int( ah / 2 )
        cw = int( aw / 2 )

        d["img0"]  = img0[ch:ch+self.h, cw:cw+self.w]
        d["img1"]  = img1[ch:ch+self.h, cw:cw+self.w]
        d["disp0"] = disp0[ch:ch+self.h, cw:cw+self.w]

        if ( "occ0" in d ):
            occ0 = d["occ0"]
            d["occ0"] = occ0[ch:ch+self.h, cw:cw+self.w]

        if ( 'valid0' in d ):
            valid0 = d["valid0"]
            d['valid0'] = valid0[ch:ch+self.h, cw:cw+self.w]

        return d

class NormalizeSelf_OCV_01(object):
    def __call__(self, x):
        """
        x is an OpenCV mat.
        """

        x = x.astype(np.float32)

        x = x - x.min()
        x = x / x.max()

        return x

class NormalizeSelf_OCV(object):
    def __call__(self, x):
        """
        x is an OpenCV mat.
        This functiion perform normalization for individual channels.

        The normalized mat will have its values ranging from -1 to 1.
        """

        if ( 2 == x.ndim ):
            # Single channel mat.
            s = x.std()
            x = x - x.mean()
            x = x / s
        elif ( 3 == x.ndim ):
            # 3-channel mat.
            x = x.clone()

            for i in range(3):
                s = x[:, :, i].std()
                x[:, :, i] = x[:, :, i] - x[:, :, i].mean()
                x[:, :, i] = x[:, :, i] / s
        else:
            raise Exception("len(x.shape) = %d. ".format(len(x.shape)))

        return x

class Random_Dict(object):
    def __init__(self, flagRandom=False, randomLimit=0.5):
        super(Random_Dict, self).__init__()

        self.flagRandom  = flagRandom
        self.randomLimit = randomLimit

        self.flagForceRandom = False

    def enable_force_random(self):
        self.flagForceRandom = True

    def random(self):
        return torch.rand(1).item()

    def random_sym(self):
        return torch.rand(1).item() * 2 - 1

    def random_in_limits(self, lower, upper):
        return lower + ( upper - lower ) * self.random()

    def is_random(self):
        if ( self.flagForceRandom ):
            return True
        else:
            return self.random() >= self.randomLimit

    def __call__(self, d):
        if ( self.flagRandom ):
            if ( self.is_random() ):
                return self.augment(d) # Defined in the inherited class.
            else:
                return d
        else:
            return self.augment(d)

@register(PRE_PROCESSORS)
class VerticalFlip_OCV_Dict(Random_Dict):
    def __init__(self, flagRandom=False, randomLimit=0.5):
        super(VerticalFlip_OCV_Dict, self).__init__(flagRandom, randomLimit)

    def flip(self, img):
        H = img.shape[0]

        idx = np.linspace(H-1, 0, H, dtype=np.int32)

        if ( img.ndim == 3 ):
            return np.ascontiguousarray( img[idx, :, :] )
        else:
            return np.ascontiguousarray( img[idx, :] )

    def flip_dict(self, d):
        d['img0']  = self.flip( d['img0'] )
        d['img1']  = self.flip( d['img1'] )
        d['disp0'] = self.flip( d['disp0'] )

        if ( 'occ0' in d ):
            d['occ0'] = self.flip( d['occ0'] )

        if ( 'valid0' in d ):
            d['valid0'] = self.flip( d['valid0'] )
        
        return d

    def augment(self, d):
        return self.flip_dict(d)

@register(PRE_PROCESSORS)
class YDispAugmentation_OCV_Dict(Random_Dict):
    '''
    This class is designed by referring to the work
    Hierarchical Deep Stereo Matching on High-resolution Images.
    '''
    def __init__(self, angle=0.1, px=2, flagRandom=False, randomLimit=0.75):
        super(YDispAugmentation_OCV_Dict, self).__init__(flagRandom, randomLimit)

        self.angle     = angle
        self.px        = px

        self.flagRandom  = flagRandom
        self.randomLimit = randomLimit

    def augment(self, d):
        img1 = d['img1']

        px2    = self.random_sym() * self.px
        angle2 = self.random_sym() * self.angle

        # print('px2    = {}'.format(px2))
        # print('angle2 = {}'.format(angle2))

        imageCenter = ( 
            torch.rand(1).item() * img1.shape[0],\
            torch.rand(1).item() * img1.shape[1])

        rotMat = cv2.getRotationMatrix2D(imageCenter, angle2, 1.0)
        img1   = cv2.warpAffine(img1, rotMat, img1.shape[1::-1], flags=cv2.INTER_LINEAR)

        transMat = np.float32([[1,0,0],[0,1,px2]])
        img1     = cv2.warpAffine(img1, transMat, img1.shape[1::-1], flags=cv2.INTER_LINEAR)

        d['img1'] = img1

        return d

@register(PRE_PROCESSORS)
class RandomColor_OCV_Dict(Random_Dict):
    def __init__(self, hSpan=25, sSpan=50, vSpan=50, scale=0.2, flagRandom=False, randomLimit=0.5):
        super(RandomColor_OCV_Dict, self).__init__(flagRandom, randomLimit)

        self.hSpan = hSpan
        self.sSpan = sSpan
        self.vSpan = vSpan
        self.scale = scale

    def random_color(self, img, rh, rs, rv):
        if ( img.ndim == 2 or ( img.ndim == 3 and img.shape[2] == 1 ) ):
            img = img + rv * self.vSpan
            img = np.clip( img, 0, 255 )
            return img
        else:
            hsv = cv2.cvtColor( img, cv2.COLOR_BGR2HSV )
            h = hsv[:, :, 0] + rh * self.hSpan # The result will be promoted to float because rh is float.
            s = hsv[:, :, 1] + rs * self.sSpan
            v = hsv[:, :, 2] + rv * self.vSpan

            h = np.clip( h, 0, 180 )
            s = np.clip( s, 0, 255 )
            v = np.clip( v, 0, 255 )

            hsv = np.stack( (h, s, v), axis=2 )

            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def augment(self, d):
        rh = self.random_sym()
        rs = self.random_sym()
        rv = self.random_sym()

        rhs = self.random_sym()*self.scale + 1
        rss = self.random_sym()*self.scale + 1
        rvs = self.random_sym()*self.scale + 1

        d['img0'] = self.random_color( d['img0'], rh,     rs,     rv )
        d['img1'] = self.random_color( d['img1'], rh*rhs, rs*rss, rv*rvs )

        return d

@register(PRE_PROCESSORS)
class MeanDispBasedRandomScale_OCV_Dict(Random_Dict):
    def __init__(self, maxDisp, flagRandom=True, randomLimit=0.5):
        super(MeanDispBasedRandomScale_OCV_Dict, self).__init__(flagRandom, randomLimit)

        self.bins  = np.array( [ 0, 50, 100, 150, 200 ], dtype=np.float32 )
        self.nBins = self.bins.size
        self.scaleLimits = np.array( 
            [ [1, 3], [1, 2], [1, 1], [1, 1] ], 
            dtype=np.float32 )
        self.maxDisp = maxDisp

    def augment(self, d):
        # compute the mean disparity.
        disp0 = d['disp0']
        if ( 'valid0' in d ):
            validMask = d['valid0']
            meanDisp = disp0[validMask].mean()
            maxDisp = disp0[validMask].max()
        else:
            meanDisp = disp0.mean()
            maxDisp  = disp0.max()

        if ( not np.isfinite(meanDisp) ):
            raise Exception('Invalid value without valid mask. ')
        
        assert( meanDisp >= 0 ), 'meanDisp = {}'.format(meanDisp)

        # Find the bin index of meanDisp.
        idx = np.digitize( meanDisp, self.bins )
        idx = idx - 2 if idx == self.nBins else idx - 1

        # Get a random scale factor.
        limitThres = self.maxDisp / maxDisp
        if ( limitThres <= 1 ):
            return d

        limit0 = min( self.scaleLimits[idx, 0], limitThres )
        limit1 = min( self.scaleLimits[idx, 1], limitThres )
        f = self.random_in_limits( limit0, limit1 )

        if ( f != 1 ):
            # Scale disp0.
            disp0 = cv2.resize( disp0, (0, 0), fx=f, fy=f, interpolation=cv2.INTER_LINEAR ) * f
            d['disp0'] = disp0

            # Scale images.
            d['img0'] = cv2.resize( d['img0'], (0, 0), fx=f, fy=f, interpolation=cv2.INTER_LINEAR )
            d['img1'] = cv2.resize( d['img1'], (0, 0), fx=f, fy=f, interpolation=cv2.INTER_LINEAR )

            if ( 'occ0' in d ):
                d['occ0'] = cv2.resize( d['occ0'], (0, 0), fx=f, fy=f, interpolation=cv2.INTER_NEAREST )

            if ( 'valid0' in d ):
                finiteMask = np.isfinite(d['disp0'])
                d['valid0'] = np.logical_and(
                    resize_bool_ocv_f( d['valid0'], f ), 
                    finiteMask )

        return d

@register(PRE_PROCESSORS)
class RecOcclusion_OCV_Dict(Random_Dict):
    def __init__(self, recWidthLimits=(50, 150), occImg0=False, 
        flagRandom=False, randomLimit=0.5):
        '''Mask a rectangular region of in a single image.
        Arguments:
        recWidthLimits (2-element): The upper and lower limits of the rectangular region.
        occImg0 (bool): Set True if mask img0 instead of img1.
        '''
        super(RecOcclusion_OCV_Dict, self).__init__( flagRandom, randomLimit)

        assert( recWidthLimits[0] < recWidthLimits[1] ), \
            f'Wrong recWidthLimits: {recWidthLimits}'
        self.recWidthLimits = recWidthLimits
        self.occImg0 = occImg0

    def augment(self, d):
        imgName = "img0" if self.occImg0 else "img1"
        img = d[imgName]

        # Random box shape.
        h = int(self.random_in_limits( *self.recWidthLimits ))
        w = int(self.random_in_limits( *self.recWidthLimits ))

        # Allowed indices.
        ah = img.shape[0] - h
        aw = img.shape[1] - w

        if ( ah < 0 ):
            raise Exception("img.shape[0] < h. img.shape[0] = {}, h = {}. ".format( img.shape[0], h ))
        if ( aw < 0 ):
            raise Exception("img.shape[1] < w. img.shape[1] = {}, w = {}. ".format( img.shape[1], w ))

        # Get two random numbers.
        ah = ah + 1
        aw = aw + 1

        ch = torch.randint(0, ah, (1, )).item()
        cw = torch.randint(0, aw, (1, )).item()

        meanRec = np.mean( img[ ch:ch+h, cw:cw+w], axis=(0, 1) )
        d[imgName][ ch:ch+h, cw:cw+w, ...] = meanRec[np.newaxis, np.newaxis]

        return d

@register(PRE_PROCESSORS)
class ColorJitter_Dict(Random_Dict):
    def __init__(self, 
        brightness=(1, 1), contrast=(1, 1), saturation=(1, 1), hue=(0, 0),
        flagRandom=False, randomLimit=0.5 ):
        super(ColorJitter_Dict, self).__init__( flagRandom, randomLimit )

        self.trans = transforms.ColorJitter( 
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue )

    def augment(self, d):
        d['img0'] = self.trans( d['img0'] )
        d['img1'] = self.trans( d['img1'] )
        return d

@register(PRE_PROCESSORS)
class AdjustGamma_Dict(Random_Dict):
    def __init__(self, gammaLimits=(1, 1), 
        flagRandom=False, randomLimit=0.5 ):
        super( AdjustGamma_Dict, self).__init__(flagRandom, randomLimit)

        assert( gammaLimits[0] <= gammaLimits[1] ), \
            f'Wrong gammaLimits: {gammaLimits}'
        self.gammaLimits = gammaLimits

    def augment_single_img(self, img):
        gamma = self.random_in_limits( *self.gammaLimits )
        return transforms.functional.adjust_gamma( img, gamma )

    def augment(self, d):
        d['img0'] = self.augment_single_img( d['img0'] )
        d['img1'] = self.augment_single_img( d['img1'] )
        
        return d