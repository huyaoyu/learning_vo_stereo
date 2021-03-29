
import cv2
import numpy as np
import torch

def tensor_img_2_array(t):
    '''t: PyTorch tensor. (C, H, W). '''
    return t.permute((1,2,0)).cpu().numpy()

def batch_img_2_array_list(b):
    '''b: batched PyTorch tensor. (B, C, H, W).'''
    return [
        tensor_img_2_array(b[i, :, :, :])
        for i in range(b.shape[0]) ]

def simple_array_2_rgb(a):
    aMin = a.min()
    aMax = a.max()
    a = ( a - aMin ) / ( aMax - aMin ) * 255
    return a.astype(np.uint8)