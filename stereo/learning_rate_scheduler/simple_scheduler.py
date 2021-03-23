# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-22

import numpy as np

from .base_scheduler import BaseScheduler
from .register import ( LR_SCHEDULERS, register )

@register(LR_SCHEDULERS)
class FixedLength(BaseScheduler):
    def __init__(self, 
        optimizer, steps, lrs):
        '''
        Arguments:
        steps (list of ints): The steps of each section.
        lrs (list of floats): The learning rate used for each section. Must have one more element than argument setps.
        '''
        super(FixedLength, self).__init__(optimizer)

        self.steps  = np.concatenate( ( np.array([0,], dtype=np.int), np.cumsum(steps).astype(np.int) ) )
        self.lrs    = lrs
        self.nSteps = self.steps.size
        self.nLRs   = len(self.lrs)

        assert( self.nLRs == self.nSteps ), 'self.nSteps = {}, self.nLRs = {}'.format(
            self.nSteps, self.nLRs )
        
        self.currentStepIdx   = 0 # Current idx for self.steps.
        self.currentStepThres = self.steps[ self.currentStepIdx ]
        self.currentLRIdx     = 0 # Current idx for self.lrs.
        self.currentLR        = self.lrs[ self.currentLRIdx ]
        self.count            = 0 # The number of times the step() function is called.

        self.reprStrInit = 'FixedLength: \n\
acc steps={steps}, \nlrs={lrs}'.format( steps=self.steps, lrs=self.lrs )

    def step(self, loss):
        # Check if we have reached the last learning rate section.
        if ( self.currentStepIdx >= self.nSteps ):
            return

        if ( self.count >= self.currentStepThres ):
            # Find the learning rate.
            if ( self.currentLRIdx >= self.nLRs ):
                self.currentLRIdx = self.nLRs - 1
            
            self.currentLR = self.lrs[ self.currentLRIdx ]
            self.update_learning_rate(self.currentLR)

            self.currentLRIdx += 1
            self.currentStepIdx += 1
            if ( self.currentStepIdx < self.nSteps ):
                self.currentStepThres = self.steps[ self.currentStepIdx ]
        
        self.count += 1

    def __repr__(self):
        s = '{}\n\
current step thres: {}, curent lr: {}, count: {}. '.format( 
    self.reprStrInit, self.currentStepThres, self.currentLR, self.count )

        return s