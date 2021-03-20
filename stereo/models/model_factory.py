# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-20

import copy

from .hsm import HSMNet

MODELS = dict(
    HSMNet=HSMNet )

def make_model(d):
    assert( isinstance(d,  dict) ), f'd must be dict. d is {type(d)}'
    
    # Make a deep copy of the input dict.
    d = copy.deepcopy(d)

    # Get the type.
    typeName = MODELS[ d['type'] ]

    # Remove the type string from the input dictionary.
    d.pop('type')

    # Create the model.
    return typeName( **d )