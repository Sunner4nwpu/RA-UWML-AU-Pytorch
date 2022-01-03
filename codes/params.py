
import torch
import torch.nn as nn
from bisect import bisect_right
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
import numpy as np
import argparse


class EpochBaseLR(_LRScheduler):
    def __init__(self, optimizer, milestones, lrs, last_epoch=-1, ):
        if len(milestones)+1 != len(lrs):
            raise ValueError('The length of milestones must equal to the '
                             ' length of lr + 1. Got {} and {} separately', len(milestones)+1, len(lrs))
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)

        self.milestones = milestones
        self.lrs = lrs
        super(EpochBaseLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.lrs[bisect_right(self.milestones, self.last_epoch)]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        lr = self.get_lr()

        for g in self.optimizer.param_groups:
            g['lr'] = lr


class ParamsControl(object):
    def __init__(self, opts, model):

        self.model = model
        self.opts = opts

        params_backbone = list(self.model.backbone.parameters())
        params_new = list(self.model.new.parameters())
        params_patch = list(self.model.patch_proposal.parameters())
        params_attention = list(self.model.self_attention.parameters())            

        if opts.use_Adam:
            self.optimizers = [
                torch.optim.Adam(params_backbone, lr=0.),
                torch.optim.Adam(params_new, lr=0.),
                torch.optim.Adam(params_patch, lr=0.),
                torch.optim.Adam(params_attention, lr=0.)
            ]
        else:
            self.optimizers = [
                torch.optim.SGD(params_backbone, lr=0., momentum=opts.momentum, weight_decay=opts.weight_decay),
                torch.optim.SGD(params_new, lr=0., momentum=opts.momentum, weight_decay=opts.weight_decay),
                torch.optim.SGD(params_patch, lr=0., momentum=opts.momentum, weight_decay=opts.weight_decay),
                torch.optim.SGD(params_attention, lr=0., momentum=opts.momentum, weight_decay=opts.weight_decay)
            ]

        lr_backbone = [float(lr) for lr in opts.lr_backbone.split(',')]
        lr_new      = [float(lr) for lr in opts.lr_new.split(',')]
        lr_patch    = [float(lr) for lr in opts.lr_patch.split(',')]
        lr_attention    = [float(lr) for lr in opts.lr_attention.split(',')]
        decay_backbone = [float(decay) for decay in opts.decay_backbone.split(',')]
        decay_new      = [float(decay) for decay in opts.decay_new.split(',')]
        decay_patch    = [float(decay) for decay in opts.decay_patch.split(',')]
        decay_attention    = [float(decay) for decay in opts.decay_attention.split(',')]     

        self.schedulers = [
            EpochBaseLR(self.optimizers[0], milestones=decay_backbone, lrs=lr_backbone),
            EpochBaseLR(self.optimizers[1], milestones=decay_new, lrs=lr_new),
            EpochBaseLR(self.optimizers[2], milestones=decay_patch, lrs=lr_patch),
            EpochBaseLR(self.optimizers[3], milestones=decay_attention, lrs=lr_attention)
        ]
     

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def back_grad(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def update(self, epoch):
        for scheduler in self.schedulers:
            scheduler.step(epoch)


