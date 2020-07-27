#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===============================================================
# Copyright (C) 2018 HuangYk.
# Licensed under The MIT Lincese.
#
# Filename     : BatchLoggers.py
# Author       : HuangYK
# Last Modified: 2018-10-23 17:40
# Description  :
# ===============================================================


from __future__ import print_function, division
from .VisdomLoggers import LossVisdom


class TRSNetBatchLogger(object):
    def __init__(self):
        self.loss = {}
        self.loss['total'] = None
        self.loss['rfn'] = None
        self.loss['trsn'] = None
        self.train_iter_idx = 0
        self.test_iter_idx = 0
        self.train_loss_visloggers = {}
        self.train_loss_visloggers['total'] = LossVisdom('Total Loss IN Batch')
        self.train_loss_visloggers['rfn'] = LossVisdom('RFN Loss IN Batch')
        self.train_loss_visloggers['trsn'] = LossVisdom('TRSN Loss IN Batch')

        self.test_loss_visloggers = {}
        self.test_loss_visloggers['total'] = LossVisdom('Total Loss IN Batch')
        self.test_loss_visloggers['rfn'] = LossVisdom('RFN Loss IN Batch')
        self.test_loss_visloggers['trsn'] = LossVisdom('TRSN Loss IN Batch')

    def __call__(self, state):
        stage = 'train' if state['train'] else 'test'
        if state['train']:
            self.loss['total'] = state['accumulate_total_loss'].data.item()
            self.loss['rfn'] = state['accumulate_rfn_loss'].data.item()
            self.loss['trsn'] = state['accumulate_trsn_loss'].data.item()
            iteration = self.train_iter_idx
            self.train_iter_idx += 1
        else:

            self.loss['total'] = state['loss']['total'].data.item()
            self.loss['rfn'] = state['loss']['rfn'].data.item()
            self.loss['trsn'] = state['loss']['trsn'].data.item()
            iteration = self.test_iter_idx
            self.test_iter_idx += 1

        for loss_key in ['total', 'rfn', 'trsn']:
            if state['train']:
                self.train_loss_visloggers[loss_key].log(
                    iteration, self.loss[loss_key], state['train'])
            else:
                self.test_loss_visloggers[loss_key].log(
                    iteration, self.loss[loss_key], state['train'])

        self.loss['total'] = None
        self.loss['rfn'] = None
        self.loss['trsn'] = None
