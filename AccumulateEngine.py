#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===============================================================
# Copyright (C) 2018 HuangYk.
# Licensed under The MIT Lincese.
#
# Filename     : AccumulateEngine.py
# Author       : HuangYK
# Last Modified: 2018-10-28 15:18
# Description  :
# ===============================================================


from __future__ import print_function, division, absolute_import
import torch
import torchnet as tnt


class AccumulateEngine(tnt.engine.Engine):
    def __init__(self):
        super(AccumulateEngine, self).__init__()

    def train(self, network, iterator, maxepoch, optimizer, accumulate_step):
        state = {
            'network': network,
            'iterator': iterator,
            'maxepoch': maxepoch,
            'optimizer': optimizer,
            'accumulate_step': accumulate_step,
            'accumulate_loss': 0,
            'epoch': 0,
            't': 0,
            'train': True,
        }

        self.hook('on_start', state)
        while state['epoch'] < state['maxepoch']:
            self.hook('on_start_epoch', state)
            for sample in state['iterator']:
                state['sample'] = sample
                self.hook('on_sample', state)

                def closure():
                    loss, output = state['network'](state['sample'])
                    state['output'] = output
                    state['loss'] = loss
                    loss.backward()
                    self.hook('on_forward', state)
                    # to free memory in save_for_backward
                    state['output'] = None
                    state['loss'] = None
                    return loss

                loss = closure()
                state['accumulate_loss'] += loss/state['accumulate_step']

                if state['accumulate_step'] == 1 or (state['t']+1) % state['accumulate_step'] == 0:
                    state['optimizer'].step()
                    state['optimizer'].zero_grad()
                    self.hook('on_update', state)
                    state['accumulate_loss'] = 0

                #if (state['t']+1) % 3 == 0:
                #    break

                state['t'] += 1
            state['epoch'] += 1
            self.hook('on_end_epoch', state)
        self.hook('on_end', state)
        return state

    def test(self, network, iterator):
        state = {
            'network': network,
            'iterator': iterator,
            't': 0,
            'train': False,
        }

        self.hook('on_start', state)
        with torch.no_grad():
            for sample in state['iterator']:
                state['sample'] = sample
                self.hook('on_sample', state)

                def closure():
                    loss, output = state['network'](state['sample'])
                    state['output'] = output
                    state['loss'] = loss
                    self.hook('on_forward', state)
                    self.hook('on_update', state)
                    # to free memory in save_for_backward
                    state['output'] = None
                    state['loss'] = None

                closure()
                #if (state['t']+1) % 3 == 0:
                #   break

                state['t'] += 1
            self.hook('on_end', state)
            return state
