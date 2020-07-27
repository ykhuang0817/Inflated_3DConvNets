#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===============================================================
# Copyright (C) 2018 HuangYk.
# Licensed under The MIT Lincese.
#
# Filename     : VisdomLoggers.py
# Author       : HuangYK
# Last Modified: 2018-10-23 17:52
# Description  :
# ===============================================================


from __future__ import print_function, division

import time
from torchnet.logger import VisdomPlotLogger, VisdomLogger


class LossVisdom(object):
    '''Plot train and test loss curve together in a VisdomPlotLogger
    '''
    def __init__(self, title='TBD'):
        self._loss = VisdomPlotLogger('line', opts={
            'title': '{:s} Loss Curve'.format(title)
        })
        check_visdom_server(self._loss.viz)

    def log(self, epoch, loss, train=None):
        assert train is not None,\
            'train should be True or False, not {}'.format(train)
        name = 'train' if train else 'test'
        try:
            self._loss.log(epoch, loss, name=name)
        except BaseException as e:
            check_visdom_server(self._loss.viz)
            print("***Retry LossVisdom")
            self.log(epoch, loss, train)


class AccuracyVisdom(object):
    '''Plot train and test accuracy curve together in a VisdomPlotLogger
    '''
    def __init__(self, title='TBD'):
        self._acc = VisdomPlotLogger('line', opts={
            'title': '{:s} Accuracy Curve'.format(title)
        })
        check_visdom_server(self._acc.viz)

    def log(self, epoch, accuracy, train=None):
        assert train is not None,\
            'train should be True or False, not {}'.format(train)
        name = 'train' if train else 'test'
        try:
            self._acc.log(epoch, accuracy, name=name)
        except BaseException as e:
            check_visdom_server(self._acc.viz)
            print("***Retry AccuracyVisdom")
            self.log(epoch, accuracy, train)


class ConfusionVisdom(object):
    '''Plot test confusion matrix in a VisdomLogger
    '''
    def __init__(self, num_classes, title='TBD'):
        self._confusion = VisdomLogger('heatmap', opts={
            'title': '{:s} Confusion Matrix'.format(title),
            'columnnames': list(range(num_classes)),
            'rownames': list(range(num_classes))
        })
        check_visdom_server(self._confusion.viz)

    def log(self, confusion, train=None):
        assert train is not None,\
            'train should be True or False, not {}'.format(train)
        if train:
            pass
        else:
            try:
                self._confusion.log(confusion)
            except BaseException as e:
                check_visdom_server(self._confusion.viz)
                print("***Retry ConfusionVisdom")
                self.log(confusion, train)


def check_visdom_server(vis):
    '''check if visdom server start up

    Args:
    -----
    vis: visdom.Visdom isinstance

    Return:
    -------
    Throw a assert exception if visdom server not work,
    return none if visdom server is running
    '''
    startup_sec = 1
    while not vis.check_connection() and startup_sec > 0:
        time.sleep(0.1)
        startup_sec -= 0.1
    assert vis.check_connection(), 'No visdom server found, \
use python -m visdom.server to start a visdom server'
