# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-22

import copy

LR_SCHEDULERS=dict()

def register(dst, name=None):
    '''Register a class to a dstination dictionary. '''
    def dec_register(cls):
        if ( name is None ):
            dst[cls.__name__] = cls
        else:
            dst[name] = cls
        return cls
    return dec_register

def register_manually(dst, cls, name=None):
    if ( name is None ):
        dst[cls.__name__] = cls
    else:
        dst[name] = cls

def make_scheduler(typeD, optimizer, argD):
    '''Make an object from type collection typeD. '''

    assert( isinstance(typeD, dict) ), f'typeD must be dict. typeD is {type(typeD)}'
    assert( isinstance(argD,  dict) ), f'argD must be dict. argD is {type(argD)}'
    
    # Make a deep copy of the input dict.
    d = copy.deepcopy(argD)

    # Get the type.
    typeName = typeD[ d['type'] ]

    # Remove the type string from the input dictionary.
    d.pop('type')

    # Create the model.
    return typeName( optimizer, **d )