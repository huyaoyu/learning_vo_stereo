import torch

def float_2_cls_label(x):
    mask = x > 0.5
    x[mask] = 1
    x[torch.logical_not(mask)] = 0
    return x.type(torch.long)