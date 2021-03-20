# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-20

from .components.register import ( MODELS, make_object )

def make_model(d):
    return make_object(MODELS, d)