# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-20

import copy

# from .hsm_ori import HSMNet as HSM_Ori

# MODELS = dict(
#     HSM_Ori=HSM_Ori )

from .register import ( MODELS, make_object )

def make_model(d):
    return make_object(MODELS, d)