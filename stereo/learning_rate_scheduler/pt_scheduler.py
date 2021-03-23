# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-22

import torch

from .base_scheduler import BaseScheduler
from .register import ( LR_SCHEDULERS, register )

@register(LR_SCHEDULERS)
class PT_ReduceLROnPlateau(BaseScheduler):
    def __init__(self, optimizer, 
        mode='min', 
        factor=0.75, 
        patience=1500, 
        threshold=0.9, 
        threshold_mode='rel', 
        cooldown=100, 
        min_lr=1e-6 ):
        super(PT_ReduceLROnPlateau, self).__init__(optimizer)

        self.lrs = torch.optim.lr_scheduler.ReduceLROnPlateau( 
            self.optimizer, 
            mode=mode, 
            factor=factor, 
            patience=patience, 
            threshold=threshold, 
            threshold_mode=threshold_mode,
            cooldown=cooldown, 
            min_lr=min_lr )
    
        self.reprStr = 'PT_ReduceLROnPlateau: \n\
mode=\'{mode}\', factor={factor}, patience={patience}, \
threshold={threshold}, threshold_mode={threshold_mode}, cooldown={cooldown}, min_lr={min_lr}. '.format(
    mode=mode, factor=factor, patience=patience, 
    threshold=threshold, threshold_mode=threshold_mode, cooldown=cooldown, min_lr=min_lr )

    def step(self, loss):
        self.lrs.step(loss)

    def __repr__(self):
        return self.reprStr

