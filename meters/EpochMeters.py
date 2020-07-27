#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===============================================================
# Copyright (C) 2018 HuangYk.
# Licensed under The MIT Lincese.
#
# Filename     : EpochMeters.py
# Author       : HuangYK
# Last Modified: 2018-10-23 15:22
# Description  :
# ===============================================================


from __future__ import print_function, division

import torch
import torchnet as tnt


class TRSNetEpochMeters(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.loss_meters = {}
        self.loss_meters['trsn'] = tnt.meter.AverageValueMeter()
        self.loss_meters['rfn'] = tnt.meter.AverageValueMeter()
        self.loss_meters['total'] = tnt.meter.AverageValueMeter()

        self.accuracy_meters = {}
        self.accuracy_meters['fusion'] = tnt.meter.ClassErrorMeter(
            accuracy=True)
        self.accuracy_meters['rfn'] = tnt.meter.ClassErrorMeter(accuracy=True)
        self.accuracy_meters['trsn'] = tnt.meter.ClassErrorMeter(accuracy=True)

        self.confusion_meters = {}
        self.confusion_meters['fusion'] = tnt.meter.ConfusionMeter(
            num_classes, normalized=True)
        self.confusion_meters['rfn'] = tnt.meter.ConfusionMeter(
            num_classes, normalized=True)
        self.confusion_meters['trsn'] = tnt.meter.ConfusionMeter(
            num_classes, normalized=True)

    @property
    def loss(self):
        loss = {}
        loss['trsn'] = self.loss_meters['trsn'].value()[0]
        loss['rfn'] = self.loss_meters['rfn'].value()[0]
        loss['total'] = self.loss_meters['total'].value()[0]

        return loss

    @property
    def accuracy(self):
        accuracy = {}
        accuracy['trsn'] = self.accuracy_meters['trsn'].value()[0]
        accuracy['rfn'] = self.accuracy_meters['rfn'].value()[0]
        accuracy['fusion'] = self.accuracy_meters['fusion'].value()[0]

        return accuracy

    @property
    def confusion_matrix(self):
        confusion_matrix = {}
        confusion_matrix['trsn'] = self.confusion_meters['trsn'].value()
        confusion_matrix['rfn'] = self.confusion_meters['rfn'].value()
        confusion_matrix['fusion'] = self.confusion_meters['fusion'].value()

        return confusion_matrix

    def reset_meters(self):
        for meter in self.loss_meters.values():
            meter.reset()
        for meter in self.accuracy_meters.values():
            meter.reset()
        for meter in self.confusion_meters.values():
            meter.reset()

    def add_loss_average(self, total_loss, trsn_loss, rfn_loss):
        self.loss_meters['trsn'].add(trsn_loss.data.item())
        self.loss_meters['rfn'].add(rfn_loss.data.item())
        self.loss_meters['total'].add(total_loss.data.item())

    def add_accurary_average(self, fusion_out, trsn_out, rfn_out, target,
                             region_target):
        self.accuracy_meters['fusion'].add(fusion_out.data, target)
        self.accuracy_meters['trsn'].add(trsn_out.data, target)
        self.accuracy_meters['rfn'].add(rfn_out.data, region_target)

    def add_confusion_matrix(self, fusion_out, trsn_out, rfn_out, target,
                             region_target):
        self.confusion_meters['fusion'].add(fusion_out.data, target)
        self.confusion_meters['trsn'].add(trsn_out.data, target)
        self.confusion_meters['rfn'].add(rfn_out.data, region_target)

    def add_meters(self, state, anchor_num):
        region_target = torch.cat([state['output']['target']]*anchor_num)
        self.add_loss_average(state['loss']['total'], state['loss']['trsn'],
                              state['loss']['rfn'])
        self.add_accurary_average(
            state['output']['fusion'], state['output']['trsn'],
            state['output']['rfn'], state['output']['target'], region_target
        )
        self.add_confusion_matrix(
            state['output']['fusion'], state['output']['trsn'],
            state['output']['rfn'], state['output']['target'], region_target
        )
