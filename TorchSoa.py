#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===============================================================
# Copyright (C) 2018 HuangYk.
# Licensed under The MIT Lincese.
#
# Filename     : TorchSoa.py
# Author       : HuangYK
# Last Modified: 2019-03-21 14:19
# Description  :
#
# ===============================================================


from __future__ import print_function, division, absolute_import
import os
import copy
import shutil
import math
import torch
import torchnet as tnt
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
import time
import numpy as np
import pandas as pd

from tqdm import tqdm  # progress bar using in python shell
from pandas import DataFrame
from collections import defaultdict


class TorchSoaEngine(object):
    '''A architecture of training process

    Inherit TorchSoaEngine to build a neural network training processor for
    specific dataset, and override abstract method get_iterator to provide a
    batch sample iterator from dataset.

    Attribute:
    ----------
    meters: Caculate loss, class accuracy, class confusion performance of
            neural networks
    model: Neural networks model at gpu device
    parameters: Total number of parameters in model

    Example:
    --------
    >> kw={'model':neural_network_instance,
           'optimizer':optimizer_instance,
           'loss_func':loss_function
           'maxepoch':max_epoch, 'batch_size':batch_size,
           'num_workers':num_workers}
    >> net_engine = TorchSoaEngine(**kw)
    >> net_engine.meters = ClassifyMeter(num_classes)
    >> net_engine.train()
    '''
    def __init__(self, model, optimizer, loss_func, maxepoch, batch_size,
                 num_workers, net_name, snap_epoch, model_dir=None, logs_dir=None,
                 gpu_id=None, loss_scheduler=None, step_scheduler=None, sgdr=False,
                 init_lr=None, T_max=10, reset_lr_params=None, dataset=None, resume=False
                ):
        '''Init with training parameters, add hooks in torchnet

        Training hooks function sequence is:
            --> hook['on_start']
              --> maxepoch iteration(
                --> hook['on_start_epoch']
                --> batch data iteration(
                  --> state['sample'] --> hook['on_sample']
                  --> state['optimizer'].zero
                  --> forward: state['network'](state['sample'])
                  --> state['output'], state['loss']
                  --> hook['on_forward'] with state['output'] and state['loss']
                  --> state['output'].zero, state['loss'].zero
                  --> backprop: state['optimizer'] with loss
                  --> hook['on_upadte']
                  --> state['t'].add
                ) # one epoch
                --> state['epoch'].add
                --> hook['on_end_epoch']
              ) # one training
            --> hook['on_end']

        Args:
        -----
        model: torch.nn.Module A nerual networks inherit nn.Module
        optimizer: torch.optim Optim method for training
        loss_func： torch.nn.functional, Loss function for nerual networks
        max_epoch: int, Epoch number for training process
        batch_size: int, Sample batch in a iteration
        num_workers: int, Number of processors for get sample
        net_name: str,

        Return:
        -------
        A normalized torch net training architecture
        '''
        self._model = model
        self._optimizer = optimizer
        self._max_epoch = maxepoch
        self._loss_func = loss_func
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._net_name = net_name
        self._snap_epoch = snap_epoch
        self._model_dir = model_dir if model_dir is not None else './epochs'
        self._logs_dir = logs_dir if logs_dir is not None else './logs'
        self._gpu_id = gpu_id
        self._dataset = dataset

        self._loss_scheduler = loss_scheduler
        self._step_scheduler = step_scheduler
        self._init_lr = init_lr
        self._use_sgdr = sgdr
        self._T_max = T_max
        self._iteration_len = None

        self._best_accuracy = 0
        self._best_epoch = 0
        self._reset_epoch = 0
        self._reset_lr_params = reset_lr_params

        self._epoch_meters = None
        self._epoch_recorder = None
        self._batch_logger = None
        self._epoch_logger = None
        self._resume = resume

        self._engine = None

    @property
    def engine(self):
        return self._engine

    @engine.setter
    def engine(self, engine):
        self._engine = engine
        self._init_engine

    @property
    def epoch_meters(self):
        return self._epoch_meters

    @epoch_meters.setter
    def epoch_meters(self, meters):
        self._epoch_meters = meters

    @property
    def epoch_rec(self):
        return self._epoch_recorder

    @epoch_rec.setter
    def epoch_rec(self, epoch_rec):
        self._epoch_recorder = epoch_rec

    @property
    def model(self):
        return self._model

    @property
    def parameters(self):
        return sum(param.numel() for param in self._model.parameters())

    def get_loggers(self, stage):
        if stage == 'epoch':
            logger = self._epoch_logger
        elif stage == 'batch':
            logger = self._batch_logger
        return logger

    def set_loggers(self, stage, logger):
        if stage == 'epoch':
            self._epoch_logger = logger
        elif stage == 'batch':
            self._batch_logger = logger

    def init_module(self, engine, epcoh_meters, epoch_recorder, epcoh_logger, batch_logger):
        self.engine = engine
        self.epoch_meters = epcoh_meters
        self.epoch_rec = epoch_recorder
        self.set_loggers('epoch', epcoh_logger)
        self.set_loggers('batch', batch_logger)

        self._init_engine()._init_recorder()

    def _init_engine(self):
        self._engine.hooks['on_start'] = self._on_start
        self._engine.hooks['on_start_epoch'] = self._on_start_epoch
        self._engine.hooks['on_sample'] = self._on_sample
        self._engine.hooks['on_forward'] = self._on_forward
        self._engine.hooks['on_end_epoch'] = self._on_end_epoch
        self._engine.hooks['on_end'] = self._on_end
        self._engine.hooks['on_update'] = self._on_update_batch

        return self

    def _init_recorder(self):
        self.epoch_rec.add_item(
            kind='confusion',
            num_classes=self.epoch_meters.num_classes
        )
        return self

    def _on_start(self, state):
        if state['train']:
            self._iteration_len = len(state['iterator'])
            print("**Iteration Numbers: {}".format(self._iteration_len))
            if self._reset_lr_params:
                print("**Restart Epochs: {}".format(self._reset_lr_params['epoch']))
            if self._resume:
                state['epoch']=self._best_epoch
                print('***Resume lr: {}'.format(
                        self._optimizer.param_groups[0]['lr'])
                )

    def _on_sample(self, state):
        '''Attach train(True) or test(False) label to samples

        Args:
        -----
        state: dict, a state dict in torchnet, state['sample'] will provide
               a list contain data, target
        '''
        state['sample'].append(state['train'])
#        if state['train'] and self._use_sgdr:
#            batch_lr = self._init_lr*sgdr(
#                self._T_max*self._iteration_len, state['t']
#            )
#            set_optimizer_lr(self._optimizer, batch_lr)
#            current_lr = self._optimizer.param_groups[0]['lr']
#            self._batch_logger.log_lr(state, current_lr)

    def _on_start_epoch(self, state):
        print('Epoch {} start'.format(state['epoch']+1))
        self._epoch_meters.reset_meters()
        #state['t'] = 0
        if self._step_scheduler and state['epoch']>0:
            if not self._resume:
                self._step_scheduler.step()
            self._resume = False
            print("**Step schedule")
        if self._use_sgdr:
            print("**SGDR schedule")
            gamma = pow(0.1, state['epoch'] // self._T_max)
            epoch_init = self._init_lr*gamma
            epoch_lr = epoch_init*sgdr(self._T_max, state['epoch'])
            set_optimizer_lr(self._optimizer, epoch_lr)
        # reset lr strategy
        if self._reset_lr_params:
            if state['epoch'] >= min(self._reset_lr_params['epoch']):
                reset_lr = self._reset_lr_params['lr']
                self._reset_epoch += 1
                # reset lr
                if state['epoch'] in self._reset_lr_params['epoch']:
                    self._reset_epoch = 0
                set_optimizer_lr(self._optimizer, reset_lr*np.power(
                    self._reset_lr_params['gamma'],
                    self._reset_epoch//self._reset_lr_params['step']
                    )
                )

        print('***current lr: {}'.format(self._optimizer.param_groups[0]['lr']))

        state['iterator'] = tqdm(state['iterator'])

    def _on_forward(self, state):
        '''Process forward output, loss before reset

        Args:
        -----
        state: dict, provide output tensor and loss in state['output'],
               state['loss']
        '''
        self._epoch_meters.add_meters(state)

    def _on_update_batch(self, state):
        self._batch_logger(state)

    def _on_end_epoch(self, state):
        stage = 'train'
        epoch_idx = state['epoch']
        print('[Epoch {}] {} end'.format(epoch_idx, stage))

        loss = self._epoch_meters.loss
        accuracy = self._epoch_meters.accuracy
        confusion = self._epoch_meters.confusion

        print_meters(epoch=epoch_idx, loss=loss, accuracy= accuracy, train=True)
        self._epoch_logger(
            epoch_idx=epoch_idx, loss=loss, accuracy=accuracy,
            confusion=confusion, train=True
        )

        self._epoch_recorder.record(
            index=epoch_idx, train=True,
            loss=loss, accuracy=accuracy,
            diag=self._epoch_meters.get_confusion_diag()[0],
            num=self._epoch_meters.get_confusion_diag()[1]
        )

        self._epoch_meters.reset_meters()

        # release gpu memory
        torch.cuda.empty_cache()

        self.test()
        stage='test'
        print('[Epoch {}] {} end'.format(epoch_idx, stage))

        loss = self._epoch_meters.loss
        accuracy = self._epoch_meters.accuracy
        confusion = self._epoch_meters.confusion

        print_meters(epoch=epoch_idx, loss=loss, accuracy= accuracy, train=False)
        self._epoch_logger(
            epoch_idx=epoch_idx, loss=loss, accuracy=accuracy,
            confusion=confusion, train=False
        )

        self._epoch_recorder.record(
            index=epoch_idx, train=False,
            loss=loss, accuracy=accuracy,
            diag=self._epoch_meters.get_confusion_diag()[0],
            num=self._epoch_meters.get_confusion_diag()[1],
            conf=self._epoch_meters.get_confusion_matrix()

        )

        if self._loss_scheduler is not None:
            self._loss_scheduler.step(loss)
            print("**Loss schedule")

        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)

        torch.save(
            {'epoch': epoch_idx, 'arch': self._model.__class__.__name__,
             'optim_state_dict': self._optimizer.state_dict(),
             'model_state_dict': self._model.state_dict(),
             'best_acc':accuracy
             }, '{:s}/{:s}_checkpoint.pth.tar'.format(
             self._model_dir, self._net_name)
        )


        if accuracy>self._best_accuracy:
            # save static params
            torch.save(
                self._model.state_dict(),
                '{:s}/{:s}_best_acc_state_dict.pth'.format(
                    self._model_dir, self._net_name
                )
            )
            self._best_accuracy = accuracy
            self._best_epoch = state['epoch']

            shutil.copy(
                '{:s}/{:s}_checkpoint.pth.tar'.format(
                        self._model_dir, self._net_name),
                '{:s}/{:s}_model_best.pth.tar'.format(
                        self._model_dir, self._net_name),
            )

        accuracy_baseline = {'hmdb51':59.2,'ucf101':87.3}.get(self._dataset, None)

        if accuracy>(self._best_accuracy-0.3) and accuracy_baseline:
            # save static params
            if accuracy > accuracy_baseline:
                print("save more best state_dict")
                shutil.copy(
                    '{:s}/{:s}_checkpoint.pth.tar'.format(
                            self._model_dir, self._net_name),
                    '{:s}/{:s}_model_{}.pth.tar'.format(
                            self._model_dir, self._net_name, accuracy),
                )

        print("[Best] Epoch {:02d} (Accuracy: {:.2f})".format(
                self._best_epoch, self._best_accuracy))

        if (epoch_idx) % self._snap_epoch == 0:
            if not os.path.exists(self._model_dir):
                os.makedirs(self._model_dir)
            torch.save(
                self._model.state_dict(),
                '{:s}/{:s}_epoch_{:02d}_state_dict.pth'.format(
                    self._model_dir, self._net_name, state['epoch']
                )
            )
            # self._epoch_logger.upate_cache()

        csv_folder = self._logs_dir
        csv_file = '_'.join([self._net_name, 'epoch', '{:02d}'.format(epoch_idx)])
        csv_file = os.path.join(csv_folder, csv_file)
        self._epoch_recorder.save_csv(csv_file, state['train'])

        # release gpu memory
        torch.cuda.empty_cache()

    def _on_end(self, state):
        '''Save training record
        '''
        csv_folder = self._logs_dir
        if not os.path.exists(csv_folder):
            os.makedirs(csv_folder)

        if state['train']:
            csv_file = '_'.join(
                [self._net_name, 'max_epoch', str(self._max_epoch)]
            )
            torch.save(
                self._model.state_dict(),
                '{:s}/{:s}_max_epoch_{:02d}_state_dict.pth'.format(
                    self._model_dir, self._net_name, self._max_epoch
                )
            )
        else:
            csv_file = '_'.join([self._net_name, 'epoch', 'test', 'tmp'])

        csv_file = os.path.join(csv_folder, csv_file)
        self._epoch_recorder.save_csv(csv_file, state['train'])

    def _network_processor(self, sample):
        data, target, train = sample

        data, target = data.cuda(self._gpu_id), target.cuda(self._gpu_id)
        if train:
            self._model.train()
        else:
            self._model.eval()

        output = self._model(data)
        loss = self._loss_func(output, target)

        return loss, output

    def get_iterator(self, train):
        raise NotImplementedError(
            'get_iterator not implemented for TorchSoaEngine, which is an \
            abstract class')


    def train(self):
        assert self._engine is not None, 'Need to set engine'
        assert self._epoch_meters is not None, 'Need to set epoch_meters'
        assert self._epoch_recorder is not None, 'Need to set epoch_recorder'
        assert self._batch_logger is not None, 'Need to set batch_logger'
        assert self._epoch_logger is not None, 'Need to set epoch_logger '

        raise NotImplementedError(
            'get_iterator not implemented for TorchSoaEngine, which is an \
            abstract class')

        self._engine.train(
            self._network_processor, self.get_iterator(True),
            maxepoch=self._max_epoch, optimizer=self._optimizer,
        )


    def test(self):

        raise NotImplementedError(
            'get_iterator not implemented for TorchSoaEngine, which is an \
            abstract class')
        self._engine.test(self._network_processor, self.get_iterator(False))


class EpochMeter(object):
    '''Classify task performance evaluation with loss curve, accuracy curve,
    confusion matrix

    This class provides loss, accuracy, confusion

    Attribute:
    ----------
    vis: ClassifyVisdom instance for plot loss, accuracy, confusion in
         visdom server in real time during training
    loss: float, average loss
    accuracy: float, average accuracy of total samples
    confusion: [k x k] np.array, class confusion matrix
    '''
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.loss_meter = tnt.meter.AverageValueMeter()
        self.acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
        self.confusion_meter = tnt.meter.ConfusionMeter(
            num_classes, normalized=True)

        self._meters = [self.loss_meter, self.acc_meter, self.confusion_meter]

    @property
    def loss(self):
        '''
        Return average loss
        '''
        return self.loss_meter.value()[0]

    @property
    def accuracy(self):
        '''
        Return average class accuracy
        '''
        return self.acc_meter.value()[0]

    @property
    def confusion(self):
        '''
        Return confusion matrix of [num_classes x num_classes]
        '''
        self.confusion_meter.normalized = True
        return self.confusion_meter.value()

    def get_confusion_diag(self):
        confusion = self.confusion_meter.conf
        return np.diag(confusion), confusion.sum(1).clip(min=1e-12)

    def get_confusion_matrix(self):
        return self.confusion_meter.conf

    def reset_meters(self):
        for meter in self._meters:
            meter.reset()

    def add_loss(self, loss):
        self.loss_meter.add(loss.data.item())

    def add_accuracy(self, output, target):
        try:
            self.acc_meter.add(output.data, target)
        except IndexError as e:
            print(e)
            print(target.shape)
            print(output.data.shape)

    def add_confusion(self, output, target):
        self.confusion_meter.add(output.data, target)

    def add_meters(self, state):
        '''Add output, target to meters(loss, acc, confusion) per batch iter

        Args:
        -----
        state: dict, provide loss, output, target
        '''
        self.add_loss(state['loss'])
        self.add_accuracy(state['output'], state['sample'][-2])
        self.add_confusion(state['output'], state['sample'][-2])


def print_meters(epoch, loss, accuracy, train):
    process = 'Training' if train else 'Test'
    print('[{:s}][Epoch {:02d}] {:s} Loss: {:.4f} (Accuracy: {:.2f}%)'.
            format(get_time(), epoch, process, loss, accuracy))


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class EpochLogger(object):
    '''Visdom logger for classify task, contain loss curve, accuracy curve and
    confusion matrix, plot in visdom server
    '''
    def __init__(self, num_classes, title='TBD'):
        self._loss_logger = LossVisdom(title=title)
        self._acc_logger = AccuracyVisdom(title=title)
        self._confusion_logger = ConfusionVisdom(num_classes=num_classes, title=title)
        self._loss_cache = defaultdict(lambda: None)
        self._acc_cache = defaultdict(lambda: None)

    def __call__(self, epoch_idx, loss, accuracy, confusion, train=None):
        if self._loss_cache[epoch_idx] == None:
            self._loss_cache[epoch_idx] = []
        self._loss_cache[epoch_idx].append((loss, train))

        if self._acc_cache[epoch_idx] == None:
            self._acc_cache[epoch_idx] = []
        self._acc_cache[epoch_idx].append((accuracy, train))

        self._loss_logger.log(epoch_idx, loss, train)
        self._acc_logger.log(epoch_idx, accuracy, train)
        self._confusion_logger.log(confusion, train)

    def upate_cache(self):
        for epoch_idx in range(1, len(self._loss_cache.items())+1):
            for loss, train in self._loss_cache[epoch_idx]:
                self._loss_logger.log(epoch_idx, loss, train)

        for epoch_idx in range(1, len(self._acc_cache.items())+1):
            for acc, train in self._acc_cache[epoch_idx]:
                self._acc_logger.log(epoch_idx, acc, train)


class BatchLogger(object):
    '''Visdom logger for classify task, contain loss curve, accuracy curve and
    confusion matrix, plot in visdom server
    '''
    def __init__(self, title='TBD'):
        self._train_iter_idx = 0
        self._test_iter_idx = 0

        self._train_loss_logger = LossVisdom(title=title+' Train')
        self._test_loss_logger = LossVisdom(title=title+' Test')
        self._lr_logger = BatchLRVisdom(title=title+' Test')

    def __call__(self, state):
        if state['train']:
            loss = state['accumulate_loss'].data.item()
            iteration = self._train_iter_idx
            self._train_iter_idx += 1
            self._train_loss_logger.log(iteration, loss, state['train'])
        else:
            loss = state['loss'].data.item()
            iteration = self._test_iter_idx
            self._test_iter_idx += 1
            self._test_loss_logger.log(iteration, loss, state['train'])

    def log_lr(self, state, lr):
        if state['train']:
            iteration = self._train_iter_idx
            self._lr_logger.log(iteration, lr, state['train'])
        else:
            pass

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
            print(e)
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
            print(e)
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
                print(e)
                print("***Retry ConfusionVisdom")
                self.log(confusion, train)


class BatchLRVisdom(object):
    def __init__(self, title='TBD'):
        self._lr = VisdomPlotLogger('line', opts={
            'title': '{:s} lr Curve'.format(title)
        })
        check_visdom_server(self._lr.viz)

    def log(self, idx, lr, train=None):
        assert train is not None,\
            'train should be True or False, not {}'.format(train)
        name = 'train' if train else 'test'
        try:
            self._lr.log(idx, lr, name=name)
        except BaseException as e:
            check_visdom_server(self._lr.viz)
            print(e)
            print("***Retry LossVisdom")
            self.log(idx, lr, train)


class EpochRecorder(object):
    '''Record loss and accuracy of a training process as csv

    '''
    items = ['loss-acc']

    def __init__(self, record_step='epoch', root_dir='./logs'):
        assert self.check_default_save_folder(), 'Save folder created failed'
        self.record_step = record_step
        self._recs = defaultdict(lambda: 'N/A')
        self._recs['loss-acc'] = LossAccRecorder(record_step, root_dir)
        self._root_dir = root_dir

    def check_default_save_folder(self, path='./logs'):
        if os.path.exists(path):
            return True
        else:
            os.makedirs(path)
            self.check_default_save_folder(path)

    def add_item(self, kind, num_classes):
        assert kind in ['confusion'], 'Record type not support'
        if kind == 'confusion':
            self.items.append(kind)
            self._recs[kind] = ConfusionRecorder(
                self.record_step, num_classes, root_dir=self._root_dir
            )

    def get_record(self):
        '''
        Return: A dict of DataFrame, which index in items
        '''
        return self._recs

    def record(self, index, train, loss=np.nan, accuracy=np.nan,
               diag=np.nan, num=np.nan, conf=None):
        '''Add loss, accuracy to DataFrame

        Args:
        -----
        index: int, epoch or batch iteration number
        loss: float, loss of net forward process in this index
        accuracy: float, average accuracy among classes in this index
        train: boolean, if this index is a training process
        '''
        kws = {'index': index, 'train': train, 'loss': loss, 'conf': conf,
               'accuracy': accuracy, 'diag': diag, 'num': num}
        for kind in self.items:
            self._recs[kind].record(**kws)

    def save_csv(self, path, train=None):
        for item in self.items:
            if not self._recs[item] == 'N/A':
                self._recs[item].save_csv(path, train=None)
            else:
                print('{} not used'.format(item))


class LossAccRecorder(object):
    '''
    '''
    def __init__(self, record_step, root_dir):
        self.record_step = record_step
        self._df = DataFrame(
            columns=[['loss', 'loss', 'accuracy', 'accuracy'],
                     ['train', 'test', 'train', 'test']]
            )

        self._df.index.name = record_step
        self._root_dir = root_dir

    def record(self, index, train, loss, accuracy, **kws):
        c_level1 = 'train' if train else 'test'
        self._df.loc[index, ('loss', (c_level1))] = loss
        self._df.loc[index, ('accuracy', (c_level1))] = accuracy

    def save_csv(self, path, train=None):
        self._df.to_csv('{0:s}_loss-acc.csv'.format(path))


class ConfusionRecorder(object):
    '''
    '''
    items = ['diag_train', 'diag_test', 'num_train', 'num_test']

    def __init__(self, record_step, num_classes, root_dir):
        self.record_step = record_step
        self._dfs = defaultdict(lambda: 'N/A')
        self._confs = []
        self._confs_keys = []
        self._conf_df = None
        self._root_dir = root_dir
        for k in self.items:
            self._dfs[k] = DataFrame(columns=np.arange(num_classes))

    def record(self, index, train, diag, num, conf=None, **kws):
        diag_key = 'diag_train' if train else 'diag_test'
        num_key = 'num_train' if train else 'num_test'
        self._dfs[diag_key].loc[index] = diag
        self._dfs[num_key].loc[index] = num
        if conf is not None and not train:
            self._conf_df  = DataFrame(conf)
            self._conf_df .to_csv(
                './{2:s}/{0:s}_{1:d}_test_confusion.csv'.format(
                    self.record_step, index, self._root_dir)
            )
            self._confs.append(copy.deepcopy(self._conf_df))
            self._confs_keys.append('epoch_{:d}'.format(index))

    def save_csv(self, path, train=None):
        df = pd.concat(
            [self._dfs['diag_train'], self._dfs['diag_test'],
             self._dfs['num_train'], self._dfs['num_test']],
            axis=1, keys=self.items
        )
        df.index.name = self.record_step
        df.to_csv('{:s}_diag.csv'.format(path))
        if len(self._confs) > 0 and not train:
            conf_concat_df = pd.concat(
                self._confs, axis=1, keys=self._confs_keys
            )
            conf_concat_df.index.name = 'Target'
            conf_concat_df.to_csv('{:s}_confusion.csv'.format(path))


def set_optimizer_lr(optimizer, lr):
    # callback to set the learning rate in an optimizer, without rebuilding
    # the whole optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def sgdr(period, batch_idx):
    # returns normalised anytime sgdr schedule given period and batch_idx
    # best performing settings reported in paper are T_0 = 10, T_mult=2
    # so always use T_mult=2
    batch_idx = float(batch_idx)
    restart_period = period
    while batch_idx/restart_period > 1.:
        batch_idx = batch_idx - restart_period
        restart_period = restart_period * 2.

    radians = math.pi*(batch_idx/restart_period)
    return 0.5*(1.0 + math.cos(radians))


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
