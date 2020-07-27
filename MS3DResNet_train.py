#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===============================================================
# Copyright (C) 2018 HuangYk.
# Licensed under The MIT Lincese.
#
# Filename     : MS3DRes_train.py
# Author       : HuangYK
# Last Modified: 2018-11-05 17:37
# Description  :
# ===============================================================

from __future__ import print_function, division, absolute_import

import os
import torch
import argparse
import time

from torchsummary import summary
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from config.config import cfg
from prettytable import PrettyTable
from focalloss import FocalLoss
import numpy as np

#from network import MS3DResnet_copy20190528 as MS3DResnet
from network import MS3DResnet
#from network import MS3DResnetB as MS3DResnet
#from network import CrossMS3DResnet as MS3DResnet
#from network import SynMS3DResnet as MS3DResnet
from network import BoostMS3DResnet
from network import densenet3d
#from network.MS3DResnet import get_1x_lr_params, get_5x_lr_params, get_10x_lr_params
#from network.resnext_3d import resnext3D50_32x4d, resnext3D101_32x4d, resnext3D101_64x4d
#from network.resnext_3d import get_1x_lr_params, get_5x_lr_params, get_10x_lr_params
from VideoDataLoader import get_dataloader
from TorchSoa import TorchSoaEngine
from TorchSoa import EpochLogger, EpochMeter, EpochRecorder, BatchLogger
from AccumulateEngine import AccumulateEngine


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a MS3DRes network')
    parser.add_argument('--dataset', dest='dataset', help='training dataset',
                        default='hmdb51', type=str)
    parser.add_argument('--split', dest='split', help='training split',
                        default='s1', type=str)
    parser.add_argument('--epochs', dest='max_epoch',
                        help='number of epoch to train',
                        default=25, type=int)
    parser.add_argument('--batch', dest='batch_size',
                        help='number of sample per batch',
                        default=16, type=int)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=4, type=int)
    parser.add_argument('--snap', dest='snap_epoch',
                        help='number of epoch to save model',
                        default=100, type=int)
    parser.add_argument('--id', dest='gpu_id',
                        help='choose GPU to train',
                        default=0, type=int)
    parser.add_argument('--ua', dest='use_anchor',
                        help='whether use temporal anchor',
                        default=False, action='store_true')
    parser.add_argument('--save_dir', dest='model_dir',
                        help='directory to save models',
                        default="./epochs", type=str)
    parser.add_argument('--log_dir', dest='log_dir',
                        help='directory to save logs',
                        default="./logs", type=str)
#   config net
    parser.add_argument('--net', dest='net_depth',
                        help='set resnet depth',
                        default='18', type=str, 
                        choices=['18', '34', '50', '101', '152', '50_32', 
                                 '101_32', '101_32x4', '101_64x4', 'd121',
                                 'd161', 'd169', 'd201'])

    parser.add_argument('--pretrained', dest='pretrained',
                        help='set pretrained networks',
                        default=False, action='store_true')

    parser.add_argument('--boost', dest='boost_model',
                        help='use boost model',
                        default=False, action='store_true')
    
    parser.add_argument('--gr', dest='global_rate',
                        help='fusion rate for global',
                        default=0.5, type=float)

    parser.add_argument('--ddp', dest='densen_drop_out',
                        help='densen block drop out',
                        default=0, type=float)

    parser.add_argument('--rdr', dest='res_dr',
                        help='resliner drop out rate',
                        default=0.2, type=float)

#   config loss type
    parser.add_argument('--loss', dest='loss_func',
                        help='loss function used',
                        default="cross_entropy", type=str, 
                        choices=["cross_entropy", "focal"])
    parser.add_argument('--fgamma', dest='focal_gamma',
                        help='focal loss gamma',
                        default=1, type=int)
    
#   config spatial_transforms
    parser.add_argument('--sm', dest='s_crop_method',
                        help='spatial crop method',
                        default="random", type=str, 
                        choices=["random", "center", "corner"])
    parser.add_argument('--ims', dest='im_size',
                        help='image resize value',
                        default=224, type=int)
    parser.add_argument('--norm', dest='norm_value',
                        help='If 1, range of inputs is [0-255]. \
                        If 255, range of inputs is [0-1].',
                        default=255, type=int, choices=[1, 255, 128])

#   config temporal_transforms
    parser.add_argument('--tm', dest='temporal_crop_method',
                        help='temporal crop method',
                        default="random", type=str, 
                        choices=["random", "center", "begin", "randomstep"])
    parser.add_argument('--ts', dest='temporal_size',
                        help='video sample frame size',
                        default=16, type=int)
    parser.add_argument('--tsm', dest='temporal_step_method',
                        help='temporal sample step method',
                        default="fixed", type=str, 
                        choices=['random', 'fixed'])
    parser.add_argument('--tstep', dest='temporal_step',
                        help='temporal sample step size',
                        default=1, type=int, choices=[1, 2, 3])
    
#   config anchor_transforms
    parser.add_argument('--am', dest='anchor_method',
                        help='anchor propose method',
                        default='random', type=str, 
                        choices=["random", "center", "begin", "conseq"])
    parser.add_argument('--an', dest='anchor_num',
                        help='number of anchor to propose',
                        default=1, type=int)
    parser.add_argument('--ad', dest='anchor_duration',
                        help='frames in per anchor',
                        default=16, type=int)
    parser.add_argument('--sldt', dest='slice_distribution',
                        help='choose anchor slice distribution',
                        default='unit', type=str, 
                        choices=["unit", "norm"])

#   config accumulate_engine
    parser.add_argument('--acst', dest='accumulate_step',
                        help='accumulate step for larger batch with limit GPU',
                        default=1, type=int)
    parser.add_argument('--acth', dest='accumulate_batch',
                        help='accumulate batch size',
                        default=1, type=int)

#   config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str, 
                        choices=["sgd"])
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)

#   config lr scheduler
    parser.add_argument('--sch', dest='scheduler',
                        help='lr scheduler',
                        default="step", type=str, 
                        choices=["step", "mul", "loss", 'sgdr'])
    parser.add_argument('--step_size', dest='step_size',
                        help='number of epoch to decay lr',
                        default=1, type=int)
    parser.add_argument('--mul_size', dest='mul_size',
                        help='number of epoch to decay lr',
                        default='30,100', type=str)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.8, type=float)
    parser.add_argument('--tmax', dest='T_max',
                        help='epoch period for sgdr',
                        default=10, type=int)

#   config reset_lr scheduler
    parser.add_argument('--url', dest='use_reset_lr',
                        help='use reset lr scheduler',
                        default=False, action='store_true')
    parser.add_argument('--reset_epoch', dest='reset_epoch',
                        help='reset epoch in scheduler',
                        default='100', type=str)
    parser.add_argument('--reset_lr', dest='reset_lr',
                        help='reset learning rate',
                        default=0.0001, type=float)
    parser.add_argument('--reset_step', dest='reset_step',
                        help='step size in reset lr scheduler',
                        default=1, type=int)
    parser.add_argument('--reset_gamma', dest='reset_gamma',
                        help='reset learning rate decay ratio',
                        default=0.78, type=float)
    
#   config distribute training
    parser.add_argument('--ngpu', dest='n_gpu',
                        help='number of gpu to use',
                        default=1, type=int)

#   config opticflow training
    parser.add_argument('--opf', dest='use_opf',
                        help='use optic flow for training',
                        default=False, action='store_true')

#   resume params
    parser.add_argument('--resume', dest='resume',
                        help='resume checkpoint'
                        ) 

    args = parser.parse_args()
    return args


def model_factory(depth='18', num_classes=51, pretrained=False, t_dim=1, 
                  duration=16, pretrained_dataset = "imagenet",
                  boost_model=False, gr=0.5, ddp=0, res_dr=0, use_opf=False):
    _method, _kargs = {
            '18': (MS3DResnet.MS3DResnet18 if not boost_model else BoostMS3DResnet.MS3DResnet18,
                   {'num_classes': num_classes, 't_dim': t_dim, 'gr':gr, 'res_dr':res_dr,
                    'duration': duration, 'pretrained':pretrained, 'IS_OPF_NET': use_opf}),
            '34': (MS3DResnet.MS3DResnet34 if not boost_model else BoostMS3DResnet.MS3DResnet34, 
                   {'num_classes': num_classes, 't_dim': t_dim, 'gr':gr, 'res_dr':res_dr,
                    'duration': duration, 'pretrained':pretrained, 'IS_OPF_NET': use_opf}),
            '50': (MS3DResnet.MS3DResnet50 if not boost_model else BoostMS3DResnet.MS3DResnet50, 
                   {'num_classes': num_classes, 't_dim': t_dim, 'gr':gr, 'res_dr':res_dr,
                    'duration': duration, 'pretrained':pretrained, 'IS_OPF_NET': use_opf}),
            '101': (MS3DResnet.MS3DResnet101 if not boost_model else BoostMS3DResnet.MS3DResnet101, 
                    {'num_classes': num_classes, 't_dim': t_dim, 'gr':gr, 'res_dr':res_dr,
                    'duration': duration, 'pretrained':pretrained, 'IS_OPF_NET': use_opf}),
            '152': (MS3DResnet.MS3DResnet152 if not boost_model else BoostMS3DResnet.MS3DResnet152, 
                    {'num_classes': num_classes, 't_dim': t_dim, 'gr':gr, 'res_dr':res_dr,
                    'duration': duration, 'pretrained':pretrained, 'IS_OPF_NET': use_opf}),
#            '50_32': (MS3DResnet.MS3DResnext3D50_32x4d, 
#            		{'num_classes': num_classes, 't_dim': t_dim, 'res_dr':res_dr,
#                    'duration': duration, 'pretrained':pretrained}),
#            '101_32': (MS3DResnet.MS3DResnext3D101_32x8d, 
#            		{'num_classes': num_classes, 't_dim': t_dim, 'res_dr':res_dr,
#                    'duration': duration, 'pretrained':pretrained}),
#            '101_32x4': (MS3DResnet.MS3DResnext3D101_32x4d, 
#            		{'num_classes': num_classes, 't_dim': t_dim, 'res_dr':res_dr,
#                    'duration': duration, 'pretrained':pretrained}),
#            '101_64x4': (MS3DResnet.MS3DResnext3D101_64x4d, 
#            		{'num_classes': num_classes, 't_dim': t_dim, 'res_dr':res_dr,
#                    'duration': duration, 'pretrained':pretrained}),
            'd121':(densenet3d.densenet3D121,
                    {'num_classes': num_classes, 't_dim': t_dim, 'drop_rate':ddp, 'res_dr':res_dr,
                    'duration': duration, 'pretrained':pretrained, 'IS_OPF_NET': use_opf}),   
            'd161':(densenet3d.densenet3D161,
                    {'num_classes': num_classes, 't_dim': t_dim, 'drop_rate':ddp, 'res_dr':res_dr,
                    'duration': duration, 'pretrained':pretrained, 'IS_OPF_NET': use_opf}),    
            'd169':(densenet3d.densenet3D169,
                    {'num_classes': num_classes, 't_dim': t_dim, 'drop_rate':ddp, 'res_dr':res_dr,
                    'duration': duration, 'pretrained':pretrained, 'IS_OPF_NET': use_opf}),
            'd201':(densenet3d.densenet3D201,
                    {'num_classes': num_classes, 't_dim': t_dim, 'drop_rate':ddp, 'res_dr':res_dr,
                    'duration': duration, 'pretrained':pretrained, 'IS_OPF_NET': use_opf}),
#            '101_64': (resnext3D101_64x4d, {
#                    'num_classes': num_classes, 'pretrained':pretrained}),    
    }.get(depth)
    print("**Load ResNet{:s}".format(depth))
    model = _method(**_kargs)
    
    return model

class MS3DResEngine(TorchSoaEngine):
    def __init__(self, init_kws, dataloader_kws, accumulate_step):
        super(MS3DResEngine, self).__init__(**init_kws)
        self.dataloader_kws = dataloader_kws
        self.accumulate_step = accumulate_step

    def get_iterator(self, train):
        subset = 'train' if train else 'test'
        self.dataloader_kws['subset'] = subset
        self.dataloader_kws['spatial_transforms_param']['subset'] = subset
        if self.dataloader_kws['temporal_transforms_param'] is not None:
            self.dataloader_kws['temporal_transforms_param']['subset'] = subset
        if self.dataloader_kws['anchor_transforms_param'] is not None:
            self.dataloader_kws['anchor_transforms_param']['subset'] = subset
        return get_dataloader(**self.dataloader_kws)

    def train(self):
        assert self._engine is not None, 'Need to set engine'
        assert self._epoch_meters is not None, 'Need to set epoch_meters'
        assert self._epoch_recorder is not None, 'Need to set epoch_recorder'
        assert self._batch_logger is not None, 'Need to set batch_logger'
        assert self._epoch_logger is not None, 'Need to set epoch_logger '
        self._engine.train(
            self._network_processor, self.get_iterator(True),
            maxepoch=self._max_epoch, optimizer=self._optimizer,
            accumulate_step=self.accumulate_step
        )

    def test(self):
        self._engine.test(
            self._network_processor, self.get_iterator(False)
        )


def print_args_table(args):
    assert isinstance(args, dict)
    args_table = PrettyTable(["args_name", "args_param"])
    args_table.padding_width=1
    for args_name, args_param in args.items():
        args_table.add_row([args_name, args_param])
    print(args_table)

def get_mean(norm_value=255, dataset='activitynet'):
    assert dataset in ['activitynet', 'kinetics', 'imagenet', 'opticflow']
    if dataset == 'activitynet':
        return [
            114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value
        ]
    elif dataset == 'kinetics':
        # Kinetics (10 videos for each class)
        return [
            110.63666788 / norm_value, 103.16065604 / norm_value,
            96.29023126 / norm_value
        ]
    elif dataset == 'imagenet':
        return [0.485, 0.456, 0.406]
        # return [90, 98, 102]
    elif dataset == 'opticflow':
        return [0.5, 0.5]


def get_std(norm_value=255, dataset='activitynet'):
    # Kinetics (10 videos for each class)
    assert dataset in ['activitynet', 'kinetics', 'imagenet', 'opticflow']
    if dataset == 'imagenet':
        return [0.229, 0.224, 0.225]
        #return [1, 1, 1]
    elif dataset == 'kinetics':
        return [
            38.7568578 / norm_value, 37.88248729 / norm_value,
            40.02898126 / norm_value
        ]
    elif dataset == 'activitynet':
        return [
            38.7568578 / norm_value, 37.88248729 / norm_value,
            40.02898126 / norm_value
        ]
    elif dataset == 'opticflow':
        return [np.mean([0.229, 0.224, 0.225]), 
                np.mean([0.229, 0.224, 0.225]),
                ]

if __name__ == '__main__':
    args = parse_args()
    
    assert torch.cuda.is_available()
# config dataloader
    dataset = args.dataset
    use_anchor = args.use_anchor
    split = args.split
    batch_size = args.batch_size
    num_workers = args.num_workers
    max_epoch = args.max_epoch

    snap_epoch = args.snap_epoch
    gpu_id = args.gpu_id

    subset = 'train'
    drop_last=False
    num_classes = {'hmdb51': 51, 'ucf101':101, 'ucf50':50}.get(dataset)

#   config transforms
    '''
    If fine-tuning in imagenet, norm_value=255.
    If fine-tuning in kinetics, norm_value=1.
    '''
    norm_value = args.norm_value
    if norm_value == 1:
        input_mean = get_mean(norm_value, 'imagenet')
        input_std = get_std(norm_value, 'imagenet')
    elif norm_value == 255:
        input_mean = get_mean(norm_value, 'imagenet')
        input_std = get_std(norm_value, 'imagenet')
#    elif norm_value == 128:
#        input_mean = get_mean(norm_value, 'imagenet')
#        input_std = get_std(norm_value, 'imagenet')
    
    scales = [1.0, 0.875, 0.75, 0.66]
    #scales = [1.15, 1.0, 0.95, 0.875]
    if args.use_opf:
    	opt_mean = get_mean(norm_value, 'opticflow')
    	opt_std = get_std(norm_value, 'opticflow')

    print("**Norm Value:{}".format(norm_value))
    print("**Input Mean:{}".format(input_mean))
    print("**Input Std:{}".format(input_std))
    if args.use_opf:
    	print("**Opticflow Mean:{}".format(opt_mean))
    	print("**Opticflow Std:{}".format(opt_std))
    print("**Crop Scale :{}".format(scales))
    print("**{:s}: {}".format('connection fc type', cfg.FC_TYPE))
    
    spatial_transforms_param = {
        'subset': subset,
        'method':args.s_crop_method, 'scales': scales,
        'im_size': args.im_size, 'norm_value': norm_value,
        'input_mean': input_mean,
        'input_std': input_std
        }

    temporal_transforms_param = {
        'crop_method': args.temporal_crop_method, 'step': args.temporal_step,
        'step_method': args.temporal_step_method, 'subset': subset, 
        'size': args.temporal_size} if not use_anchor else None

    anchor_transforms_param = {
        'anchor_method': args.anchor_method, 'anchor_num': args.anchor_num,
        'anchor_duration': args.anchor_duration, 
        'slice_distribution': args.slice_distribution,
        'subset':subset} if use_anchor else None

    opf_transforms_param = {
        'subset': subset,
        'method':args.s_crop_method, 'scales': scales,
        'im_size': args.im_size, 'norm_value': norm_value,
        'input_mean': opt_mean,
        'input_std': opt_std
        } if args.use_opf else None

#   config train param
    lr = args.lr
    weight_decay = 1e-3

#   config model
    net_depth = args.net_depth
    pretrained_dataset = "imagenet"
    # pretrained_dataset = "kinetics"
    if args.n_gpu == 1:
        model = model_factory(
                depth=net_depth, num_classes=num_classes, t_dim=args.anchor_num,
                duration=args.anchor_duration, pretrained=args.pretrained, 
                pretrained_dataset=pretrained_dataset, boost_model=args.boost_model,
                gr=args.global_rate, ddp=args.densen_drop_out, res_dr=args.res_dr,
                use_opf=args.use_opf).cuda(gpu_id)
#        train_params = [
#            {'params': get_1x_lr_params(model), 'lr': lr},
#            #{'params': get_5x_lr_params(model), 'lr': lr * 5},
#            {'params': get_10x_lr_params(model), 'lr': lr * 10}]
        train_params = model.parameters()
    else:
        gpu_id = None
        model = torch.nn.DataParallel(
                model_factory(
                depth=net_depth, num_classes=num_classes, t_dim=args.anchor_num,
                duration=args.anchor_duration, pretrained=args.pretrained, 
                pretrained_dataset=pretrained_dataset, boost_model=args.boost_model,
                gr=args.global_rate, ddp=args.densen_drop_out, res_dr=args.res_dr,
                use_opf=args.use_opf), device_ids=list(range(args.n_gpu))
            ).cuda()
        #model = convert_model(model).cuda()
        #train_params = [
        #    {'params': get_1x_lr_params(model.module), 'lr': lr},
            #{'params': get_5x_lr_params(model.module), 'lr': lr * 5},
        #    {'params': get_10x_lr_params(model.module), 'lr': lr * 10}]
        train_params = model.parameters() 
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("**Resumed model loaded")
    #model = convert_model(model).cuda()
    #train_params = model.parameters()  
#   config optimizer
    #optim_params = {'momentum':0.9, 'lr': lr}
    if args.resume:
        lr=checkpoint['optim_state_dict']['param_groups'][0]['lr']
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            train_params, lr=lr, momentum=0.9, weight_decay=weight_decay)

    elif args.optimizer == 'adam':
        optimizer = optim.Adam(train_params, lr=lr)
    
#   config scheduler
    step_size = args.step_size
    mul_size = map(int, args.mul_size.split(','))
    lr_decay_gamma = args.lr_decay_gamma
    lr_patience = args.step_size
    T_max = args.T_max
    use_sgdr = False
    if args.scheduler == 'step':
        step_scheduler = lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=lr_decay_gamma
        )
        loss_scheduler = None
    elif args.scheduler == 'mul':
        step_scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=mul_size, gamma=lr_decay_gamma
        )
        print('**Multi-step size:{}'.format(mul_size))
        loss_scheduler = None
    elif args.scheduler == 'loss':
        step_scheduler = None
        loss_scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=lr_patience
        )
    elif args.scheduler == 'sgdr':
        step_scheduler = None
        loss_scheduler = None
        use_sgdr = True
    
    if args.resume:
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        print("**Resumed optimizer loaded")
    ## lr = 0.1
#    loss_scheduler = lr_scheduler.ReduceLROnPlateau(
#        optimizer, 'min', patience=lr_patience
#        ) if args.scheduler == 'loss' else None

#   dataloader params
    dataloader_kws = {
        'dataset': dataset, 'subset':subset, 'split': split,
        'batch_size': batch_size, 'num_workers': num_workers,
        'drop_last': drop_last,
        'spatial_transforms_param': spatial_transforms_param,
        'temporal_transforms_param': temporal_transforms_param,
        'anchor_transforms_param': anchor_transforms_param,
        'opf_transforms_param': opf_transforms_param
    }

#   engine params
    dataset_name =  dataset+"_opf_{}".format(cfg.OPTIC_FLOW_TYPE) if args.use_opf else dataset
    if args.pretrained:
        net_name = 'MS3DRes{:s}_{:s}_init_{:s}_pretrained'.format(
                net_depth, dataset_name, pretrained_dataset)
    else :
        net_name = 'MS3DRes{:s}_{:s}_init_no_pretrained'.format(
                net_depth, dataset_name)

    log_subdir = "{:s}-{:s}-{:s}-{:s}".format(
            time.strftime("%Y-%m-%d-%H"), dataset, split, net_name)
    model_subdir = "{:s}-{:s}-{:s}-{:s}".format(
            time.strftime("%Y-%m-%d-%H"), dataset, split, net_name)

    log_dir = "{:s}/{:s}".format(args.log_dir, log_subdir)
    model_dir = "{:s}/{:s}".format(args.model_dir, model_subdir)
    
    loss_func = {
        "cross_entropy": F.cross_entropy, 
        #"focal": FocalLoss(gamma=args.focal_gamma)
    }.get(args.loss_func)
    
    if args.use_reset_lr:
		reset_lr_params={
            'epoch':map(int, args.reset_epoch.split(',')), 'lr':args.reset_lr,
            'gamma':args.reset_gamma, 'step':args.reset_step
	    }
    else :
    	reset_lr_params = None
    
    MS3DResEngine_params = {
        'model': model, 'optimizer': optimizer, 'maxepoch': max_epoch,
        'loss_func': loss_func, 'batch_size': batch_size,
        'num_workers': num_workers, 'net_name': net_name,
        'snap_epoch':snap_epoch,
        'model_dir': model_dir, 'logs_dir': log_dir, 'gpu_id': gpu_id,
        'loss_scheduler':loss_scheduler, 'step_scheduler':step_scheduler,
        'sgdr': use_sgdr, 'init_lr': lr, 'T_max': T_max, 
        'reset_lr_params': reset_lr_params
    }

#   torch config
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True #不加则显存会爆掉
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(4)
    torch.cuda.manual_seed(4)
    torch.cuda.manual_seed_all(4)

#   engine config
    MS3DRes_engine = MS3DResEngine(
            init_kws=MS3DResEngine_params, dataloader_kws=dataloader_kws, 
            accumulate_step=args.accumulate_step)
    MS3DRes_engine.init_module(
        AccumulateEngine(), EpochMeter(num_classes=num_classes),
        EpochRecorder(record_step='{:s}_epoch'.format(dataset),
                      root_dir=log_dir),
        EpochLogger(num_classes, title='MS3DRes{:s}_{:s}_{:s}'.format(net_depth, dataset, split)),
        BatchLogger(title='MS3DRes{:s}_{:s}_{:s}'.format(net_depth, dataset, split))
    )
    
    if args.resume:
        MS3DRes_engine._best_accuracy = checkpoint['best_acc']
        MS3DRes_engine._best_epoch = checkpoint['epoch']
        MS3DRes_engine._resume = True
        
    args.accumulate_batch = batch_size*args.accumulate_step
    
    args_file = os.path.join(log_dir, "Args_{:s}.txt".format(net_name))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(args_file, 'w') as f:
        f.write("Args Name: Args Param\n")
        for args_name, args_param in vars(args).items():
            f.write("{:s}: {}\n".format(args_name, args_param))
        f.write("{:s}: {}\n".format('input_mean', input_mean))
        f.write("{:s}: {}\n".format('input_std', input_std))
        f.write("{:s}: {}\n".format('net parameters', MS3DRes_engine.parameters))
        f.write("{:s}: {}\n".format('scales', scales))
        f.write("{:s}: {}\n".format('optic flow type', cfg.OPTIC_FLOW_TYPE))
        f.write("{:s}: {}\n".format('connection fc type', cfg.FC_TYPE))
        if args.resume:
            f.write("{:s}: {}\n".format('resume path', args.resume))

    print("**Net parameters: {}".format(MS3DRes_engine.parameters))
    print("**Optic flow type: {}".format(cfg.OPTIC_FLOW_TYPE))
    print_args_table(vars(args))
    
    if use_anchor:
        temporal_size = args.anchor_num * args.anchor_duration
    else:
        temporal_size = args.temporal_size
    summary_file = os.path.join(log_dir, "Summary_{:s}.txt".format(net_name))
    in_channel = 2 if args.use_opf else 3
    lines = summary(model, (in_channel, temporal_size, args.im_size, args.im_size))
    torch.cuda.empty_cache()
    with open(summary_file, 'w') as f:
        for summary_line in lines:
            f.write(summary_line+"\n")
            
    MS3DRes_engine.train()
