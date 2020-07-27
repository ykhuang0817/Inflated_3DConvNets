#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===============================================================
# Copyright (C) 2018 HuangYk.
# Licensed under The MIT Lincese.
#
# Filename     : EpochLoggers.py
# Author       : HuangYK
# Last Modified: 2018-10-23 17:58
# Description  :
# ===============================================================


from __future__ import print_function, division

from .VisdomLoggers import LossVisdom, AccuracyVisdom, ConfusionVisdom


class TRSNetEpochLogger(object):
    def __init__(self, num_classes):
        self.loss_visloggers = {}
        self.loss_visloggers['total'] = LossVisdom('Total')
        self.loss_visloggers['rfn'] = LossVisdom('RFN')
        self.loss_visloggers['trsn'] = LossVisdom('TRSN')

        self.accuracy_visloggers = {}
        self.accuracy_visloggers['fusion'] = AccuracyVisdom('Fusion')
        self.accuracy_visloggers['rfn'] = AccuracyVisdom('RFN')
        self.accuracy_visloggers['trsn'] = AccuracyVisdom('TRSN')

        self.confusion_visloggers = {}
        self.confusion_visloggers['fusion'] = ConfusionVisdom(
            num_classes, 'Fusion')
        self.confusion_visloggers['rfn'] = ConfusionVisdom(
            num_classes, 'RFN')
        self.confusion_visloggers['trsn'] = ConfusionVisdom(
            num_classes, 'TRSN')

    def __call__(self, epoch_idx, avg_loss, avg_accuracy,
                 confusion_matrix, is_train=True):
        for loss_key in ['total', 'rfn', 'trsn']:
            self.loss_visloggers[loss_key].log(
                epoch_idx, avg_loss[loss_key], is_train
            )
        for out_key in ['fusion', 'rfn', 'trsn']:
            self.accuracy_visloggers[out_key].log(
                epoch_idx, avg_accuracy[out_key], is_train
            )
            self.confusion_visloggers[out_key].log(
                confusion_matrix[out_key], is_train
            )
