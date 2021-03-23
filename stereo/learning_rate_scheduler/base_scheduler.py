# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-22

class BaseScheduler(object):
    def __init__(self, optimizer):
        super(BaseScheduler, self).__init__()

        self.optimizer = optimizer

    def update_learning_rate(self, lr):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def step(self, loss):
        raise NotImplementedError()
