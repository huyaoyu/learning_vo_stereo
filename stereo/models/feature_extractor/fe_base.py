# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-04-04

# Top level imports.
from stereo.models.globals import GLOBAL

# PyTorch packages.
import torch

# Local packages.
from stereo.models.common.base_module import BaseModule

class FEBase(BaseModule):
    def __init__(self, levels=[1, 2, 4, 8], freeze=False):
        '''
        levels (list of ints): The level factor. The feature map of level i is 1/levels[i] the size of the input feature.
        freeze (bool): Use True to freeze this modudle.
        '''
        super(FEBase, self).__init__(freeze=freeze)

        self.levels = levels

    def n_levels(self):
        return len( self.levels )