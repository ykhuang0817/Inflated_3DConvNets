#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===============================================================
# Copyright (C) 2018 HuangYk.
# Licensed under The MIT Lincese.
#
# Filename     : MS3DResNet_test.py
# Author       : HuangYK
# Last Modified: 2019-03-21 15:40
# Description  :
# ===============================================================

from __future__ import print_function, division, absolute_import

import os
import torch
import argparse
import time
import numpy as np
#from torchsummary import summary
import torch.nn.functional as F
#from torch import optim
#from torch.optim import lr_scheduler
from prettytable import PrettyTable
from tqdm import tqdm
from pandas import DataFrame

#from network.resnet_3d_E import resnet3D18, resnet3D34, resnet3D50, resnet3D101, resnet3D152
from network import MS3DResnet
#from network import BoostMS3DResnet
from network import densenet3d
from VideoDataLoader import get_ensemble_dataloader as get_dataloader
from TorchSoa import TorchSoaEngine, print_meters, get_time
from TorchSoa import EpochLogger, EpochMeter, EpochRecorder, BatchLogger
from AccumulateEngine import AccumulateEngine
from config.config import cfg


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Res3D network')
    parser.add_argument('--dataset', dest='dataset', help='training dataset',
                        default='hmdb51', type=str)
    parser.add_argument('--split', dest='split', help='training split',
                        default='s1', type=str)
    parser.add_argument('--batch', dest='batch_size',
                        help='number of sample per batch',
                        default=16, type=int)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=4, type=int)
    parser.add_argument('--id', dest='gpu_id',
                        help='choose GPU to train',
                        default=0, type=int)
    parser.add_argument('--ua', dest='use_anchor',
                        help='whether use temporal anchor',
                        default=False, action='store_true')
    parser.add_argument('--log_dir', dest='log_dir',
                        help='directory to save logs',
                        default="./logs", type=str)

#   config net
    parser.add_argument('--net', dest='net_depth',
                        help='set resnet depth',
                        default='101', type=str,
                        choices=['18', '34', '50', '101', '152', 'd169'])

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
                        choices=["random", "center", "begin"])
    parser.add_argument('--an', dest='anchor_num',
                        help='number of anchor to propose',
                        default=2, type=int)
    parser.add_argument('--ad', dest='anchor_duration',
                        help='frames in per anchor',
                        default=16, type=int)
    parser.add_argument('--sldt', dest='slice_distribution',
                        help='anchor choose slice distribution',
                        default='norm', type=str,
                        choices=["unit", "norm"])


#   config distribute training
    parser.add_argument('--ngpu', dest='n_gpu',
                        help='number of gpu to use',
                        default=1, type=int)

##   use state_dict model params
    parser.add_argument('--state_dict', dest='state_dict',
                        help='whether use state dict parameters',
                        default=False, action='store_true')
    parser.add_argument('--state_path', dest='state_path',
                        help='directory to state dict path',
                        default="./state_dict/", type=str)
    
    #   config opticflow training
    parser.add_argument('--opf', dest='use_opf',
                        help='use optic flow for training',
                        default=False, action='store_true')

    args = parser.parse_args()
    return args

class EnsembleEngine(AccumulateEngine):
    
    def __init__(self):
        super(EnsembleEngine, self).__init__()
        
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


def model_factory(depth='18', num_classes=51, pretrained=False, t_dim=1, 
                  duration=16, pretrained_dataset = "imagenet",
                  boost_model=False, gr=0.5, ddp=0, res_dr=0, use_opf=False):
    _method, _kargs = {
            '18': (MS3DResnet.MS3DResnet18,
                   {'num_classes': num_classes, 't_dim': t_dim, 'gr':gr, 'res_dr':res_dr,
                    'duration': duration, 'pretrained':pretrained, 'IS_OPF_NET': use_opf}),
            '34': (MS3DResnet.MS3DResnet34, 
                   {'num_classes': num_classes, 't_dim': t_dim, 'gr':gr, 'res_dr':res_dr,
                    'duration': duration, 'pretrained':pretrained, 'IS_OPF_NET': use_opf}),
            '50': (MS3DResnet.MS3DResnet50, 
                   {'num_classes': num_classes, 't_dim': t_dim, 'gr':gr, 'res_dr':res_dr,
                    'duration': duration, 'pretrained':pretrained, 'IS_OPF_NET': use_opf}),
            '101': (MS3DResnet.MS3DResnet101, 
                    {'num_classes': num_classes, 't_dim': t_dim, 'gr':gr, 'res_dr':res_dr,
                    'duration': duration, 'pretrained':pretrained, 'IS_OPF_NET': use_opf}),
            '152': (MS3DResnet.MS3DResnet152, 
                    {'num_classes': num_classes, 't_dim': t_dim, 'gr':gr, 'res_dr':res_dr,
                    'duration': duration, 'pretrained':pretrained, 'IS_OPF_NET': use_opf}),
#            '50_32': (MS3DResnet.MS3DResnext3D50_32x4d, 
#                   {'num_classes': num_classes, 't_dim': t_dim, 'res_dr':res_dr,
#                    'duration': duration, 'pretrained':pretrained}),
#            '101_32': (MS3DResnet.MS3DResnext3D101_32x8d, 
#                   {'num_classes': num_classes, 't_dim': t_dim, 'res_dr':res_dr,
#                    'duration': duration, 'pretrained':pretrained}),
#            '101_32x4': (MS3DResnet.MS3DResnext3D101_32x4d, 
#                   {'num_classes': num_classes, 't_dim': t_dim, 'res_dr':res_dr,
#                    'duration': duration, 'pretrained':pretrained}),
#            '101_64x4': (MS3DResnet.MS3DResnext3D101_64x4d, 
#                   {'num_classes': num_classes, 't_dim': t_dim, 'res_dr':res_dr,
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

class Res3DEngine(TorchSoaEngine):
    def __init__(self, init_kws, dataloader_kws, accumulate_step):
        super(Res3DEngine, self).__init__(**init_kws)
        self.dataloader_kws = dataloader_kws
        self.accumulate_step = accumulate_step
        
    @property
    def parameters(self):
        return "Ensemble models"

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
#        self._engine.train(
#            self._network_processor, self.get_iterator(True),
#            maxepoch=self._max_epoch, optimizer=self._optimizer,
#            accumulate_step=self.accumulate_step
#        )

    def test(self):
        self._engine.test(
            self._network_processor, tqdm(self.get_iterator(False))
        )
        
    def _network_processor(self, sample):
        if len(self._model) == 2:
            rgb_data, opf_data, target, train = sample
    
            rgb_data, opf_data, target = rgb_data.cuda(), opf_data.cuda(), target.cuda()
            
            #assert len(self._model) >= 2
            
            outputs = []
            for model in self._model["rgb"]:
                model.eval()
                #output = model(rgb_data)
                outputs.append(model(rgb_data))
                
            for model in self._model["opf"]:
                model.eval()
                output = model(opf_data)
                outputs.append(output)
            
            output = 0
            for out in outputs:
                output += out
            
            output = output/len(outputs)
            loss = self._loss_func(output, target)
        elif len(self._model) == 3:
            rgb_data, opf_data, warp_data, target, train = sample
    
            rgb_data, opf_data, warp_data, target = rgb_data.cuda(), opf_data.cuda(), warp_data.cuda(), target.cuda()
            
            #assert len(self._model) >= 2
            
            outputs_rgb = []
            outputs_opt = []
            #outputs_3 = []
            for model in self._model["rgb"]:
                model.eval()
                #output = model(rgb_data)
                outputs_rgb.append(model(rgb_data))
                
            for model in self._model["opf"]:
                model.eval()
                output = model(opf_data)
                outputs_opt.append(output)
                
            for model in self._model["warp"]:
                model.eval()
                output = model(warp_data)
                outputs_opt.append(output)
            
            output_opt = 0
            for out in outputs_opt:
                output_opt += out
                
            output_rgb = 0
            for out in outputs_rgb:
                output_rgb += out
            
            output_opt = output_opt/len(outputs_opt)
            output_rgb = output_rgb/len(outputs_rgb)
            alpha=0.5
            output = alpha*output_opt + (1-alpha)*output_rgb
            
            loss = self._loss_func(output, target)

        return loss, output


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
                np.mean([0.229, 0.224, 0.225])]


if __name__ == '__main__':
    args = parse_args()

    assert torch.cuda.is_available()
# config dataloader
    dataset = args.dataset
    use_anchor = True
    split = args.split
    batch_size = args.batch_size
    num_workers = args.num_workers
#    max_epoch = args.max_epoch

#    snap_epoch = args.snap_epoch
    gpu_id = args.gpu_id

    subset = 'train'
    drop_last=False
    num_classes = {'hmdb51': 51, 'ucf101':101}.get(dataset)

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

    opt_mean = get_mean(norm_value, 'opticflow')
    opt_std = get_std(norm_value, 'opticflow')
    scales = [1.0, 0.875, 0.75, 0.66]
    cfg.FC_TYPE = 'resfc'
    #scales = [1.15, 1.0, 0.95, 0.875]
    print("**Norm Value:{}".format(norm_value))
    print("**Input Mean:{}".format(input_mean))
    print("**Input Std:{}".format(input_std))
    print("**Optic Mean:{}".format(opt_mean))
    print("**Optic Std:{}".format(opt_std)) 
    print("**Crop Scale :{}".format(scales))
    print("**{:s}: {}".format('connection fc type', cfg.FC_TYPE))
    
    model_path = "./epochs"
    s1_rgb_ucf_state_dict_path = {
        # 87.44
        "2019-04-11-10-ucf101-s1-MS3DRes101_ucf101_init_imagenet_pretrained/MS3DRes101_ucf101_init_imagenet_pretrained_best_acc_state_dict.pth":('101',6,6),
        # 87.76
        #"2019-05-16-10-ucf101-s1-MS3DRes152_ucf101_init_imagenet_pretrained/MS3DRes152_ucf101_init_imagenet_pretrained_best_acc_state_dict.pth":('152',6,6),
        # 87.63
        #"2019-09-12-20-ucf101-s1-MS3DResd169_ucf101_init_imagenet_pretrained/MS3DResd169_ucf101_init_imagenet_pretrained_best_acc_state_dict.pth":('d169',6,6)
    }
    s2_rgb_ucf_state_dict_path = {
        # 87.55-101
        "2019-09-27-21-ucf101-s2-MS3DRes101_ucf101_init_imagenet_pretrained/MS3DRes101_ucf101_init_imagenet_pretrained_best_acc_state_dict.pth":('101',6,6),
        # 87.55-152
        #"2019-10-05-17-ucf101-s2-MS3DRes152_ucf101_init_imagenet_pretrained/MS3DRes152_ucf101_init_imagenet_pretrained_best_acc_state_dict.pth":('152',6,6),
        # 87.33-d169
        #"2019-09-27-17-ucf101-s2-MS3DResd169_ucf101_init_imagenet_pretrained/MS3DResd169_ucf101_init_imagenet_pretrained_best_acc_state_dict.pth":('d169',6,6)
    }
    s3_rgb_ucf_state_dict_path = {
        # 86.85-101
        "2019-10-31-11-ucf101-s3-MS3DRes101_ucf101_init_imagenet_pretrained/MS3DRes101_ucf101_init_imagenet_pretrained_best_acc_state_dict.pth":('101',6,6),
        # 87.23-152
        #"2019-10-26-15-ucf101-s3-MS3DRes152_ucf101_init_imagenet_pretrained/MS3DRes152_ucf101_init_imagenet_pretrained_best_acc_state_dict.pth":('152',6,6),
        # 87.58-d169
        #"2019-10-23-16-ucf101-s3-MS3DResd169_ucf101_init_imagenet_pretrained/MS3DResd169_ucf101_init_imagenet_pretrained_best_acc_state_dict.pth":('d169',6,6)
    }
    #ucf_state_dict_path = {
    #    "2019-04-11-10-ucf101-s1-MS3DRes101_ucf101_init_imagenet_pretrained/MS3DRes101_ucf101_init_imagenet_pretrained_best_acc_state_dict.pth":('101',6,6),
    #    "2019-03-29-ucf101-s1-MS3DRes152_ucf101_init_imagenet_pretrained/MS3DRes152_ucf101_init_imagenet_pretrained_best_acc_state_dict.pth":('152',6,6)
    #}
    s1_rgb_hmdb_state_dict_path = {
        # 59.80-101
        "2019-07-03-21-hmdb51-s1-MS3DRes101_hmdb51_init_imagenet_pretrained/MS3DRes101_hmdb51_init_imagenet_pretrained_best_acc_state_dict.pth":('101',6,6),
        # 59.93-152
        #"2019-07-09-14-hmdb51-s1-MS3DRes152_hmdb51_init_imagenet_pretrained/MS3DRes152_hmdb51_init_imagenet_pretrained_best_acc_state_dict.pth":('152',6,6),
        # 60.13-d169
        #"2019-06-14-15-hmdb51-s1-MS3DResd169_hmdb51_init_imagenet_pretrained/MS3DResd169_hmdb51_init_imagenet_pretrained_best_acc_state_dict.pth":('d169',6,6)
    }
    s2_rgb_hmdb_state_dict_path = {
        # 57.71-101
        "2019-09-12-23-hmdb51-s2-MS3DRes101_hmdb51_init_imagenet_pretrained/MS3DRes101_hmdb51_init_imagenet_pretrained_best_acc_state_dict.pth":('101',6,6),
        # 58.56-152
        #"2019-09-17-22-hmdb51-s2-MS3DRes152_hmdb51_init_imagenet_pretrained/MS3DRes152_hmdb51_init_imagenet_pretrained_best_acc_state_dict.pth":('152',6,6),
        # 57.58-d169
        #"2019-09-14-22-hmdb51-s2-MS3DResd169_hmdb51_init_imagenet_pretrained/MS3DResd169_hmdb51_init_imagenet_pretrained_best_acc_state_dict.pth":('d169',6,6)
    }
    s3_rgb_hmdb_state_dict_path = {
        # 58.37-101
        "2019-11-19-13-hmdb51-s3-MS3DRes101_hmdb51_init_imagenet_pretrained/MS3DRes101_hmdb51_init_imagenet_pretrained_best_acc_state_dict.pth":('101',6,6),
        # 58.89-152
        #"2019-11-21-16-hmdb51-s3-MS3DRes152_hmdb51_init_imagenet_pretrained/MS3DRes152_hmdb51_init_imagenet_pretrained_best_acc_state_dict.pth":('152',6,6),
        # 57.52-d169
        #"2019-11-19-10-hmdb51-s3-MS3DResd169_hmdb51_init_imagenet_pretrained/MS3DResd169_hmdb51_init_imagenet_pretrained_best_acc_state_dict.pth":('d169',6,6)
    }
    
    if split == 's1':
        rgb_state_dict_path = {
            'hmdb51': s1_rgb_hmdb_state_dict_path, 
            'ucf101':s1_rgb_ucf_state_dict_path
        }.get(dataset)
    elif split == 's2':
        rgb_state_dict_path = {
            'hmdb51': s2_rgb_hmdb_state_dict_path, 
            'ucf101':s2_rgb_ucf_state_dict_path
        }.get(dataset)
    elif split == 's3':
        rgb_state_dict_path = {
            'hmdb51': s3_rgb_hmdb_state_dict_path, 
            'ucf101':s3_rgb_ucf_state_dict_path
        }.get(dataset)

    # TVL1 opf
    s1_opf_ucf_state_dict_path = {
        # 85.83-101
        "2019-10-11-12-ucf101-s1-MS3DRes101_ucf101_opf_TVL1_init_imagenet_pretrained/MS3DRes101_ucf101_opf_TVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('101',6,6),
        # 84.98-152
        #"2019-10-24-14-ucf101-s1-MS3DRes152_ucf101_opf_TVL1_init_imagenet_pretrained/MS3DRes152_ucf101_opf_TVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('152',6,6),
        # 85.73-d169
        #"2019-10-16-20-ucf101-s1-MS3DResd169_ucf101_opf_TVL1_init_imagenet_pretrained/MS3DResd169_ucf101_opf_TVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('d169',6,6)
    }
    s2_opf_ucf_state_dict_path = {
        # 87.68-101
        "2019-11-07-02-ucf101-s2-MS3DRes101_ucf101_opf_TVL1_init_imagenet_pretrained/MS3DRes101_ucf101_opf_TVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('101',6,6),
        # 87.57-152
        #"2019-11-08-15-ucf101-s2-MS3DRes152_ucf101_opf_TVL1_init_imagenet_pretrained/MS3DRes152_ucf101_opf_TVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('152',6,6),
        # 88.46-d169
        #"2019-11-07-02-ucf101-s2-MS3DResd169_ucf101_opf_TVL1_init_imagenet_pretrained/MS3DResd169_ucf101_opf_TVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('d169',6,6)
    }
    s3_opf_ucf_state_dict_path = {
        # 88.20-101
        "2019-11-01-15-ucf101-s3-MS3DRes101_ucf101_opf_TVL1_init_imagenet_pretrained/MS3DRes101_ucf101_opf_TVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('101',6,6),
        # 87.91-152
        #"2019-11-01-22-ucf101-s3-MS3DRes152_ucf101_opf_TVL1_init_imagenet_pretrained/MS3DRes152_ucf101_opf_TVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('152',6,6),
        # 88.63-d169
        #"2019-11-01-15-ucf101-s3-MS3DResd169_ucf101_opf_TVL1_init_imagenet_pretrained/MS3DResd169_ucf101_opf_TVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('d169',6,6)
    }
    #ucf_state_dict_path = {
    #    "2019-04-11-10-ucf101-s1-MS3DRes101_ucf101_init_imagenet_pretrained/MS3DRes101_ucf101_init_imagenet_pretrained_best_acc_state_dict.pth":('101',6,6),
    #    "2019-03-29-ucf101-s1-MS3DRes152_ucf101_init_imagenet_pretrained/MS3DRes152_ucf101_init_imagenet_pretrained_best_acc_state_dict.pth":('152',6,6)
    #}
    s1_opf_hmdb_state_dict_path = {
        # 59.74-101
        "2019-07-18-20-hmdb51-s1-MS3DRes101_hmdb51_opf_init_imagenet_pretrained/MS3DRes101_hmdb51_opf_init_imagenet_pretrained_best_acc_state_dict.pth":('101',6,6),
        # 59.02-152
        #"2019-07-23-16-hmdb51-s1-MS3DRes152_hmdb51_opf_init_imagenet_pretrained/MS3DRes152_hmdb51_opf_init_imagenet_pretrained_best_acc_state_dict.pth":('152',6,6),
        # 59.93-d169
        #"2019-07-23-08-hmdb51-s1-MS3DResd169_hmdb51_opf_init_imagenet_pretrained/MS3DResd169_hmdb51_opf_init_imagenet_pretrained_best_acc_state_dict.pth":('d169',6,6)
    }
    s2_opf_hmdb_state_dict_path = {
        # 59.28-101
        "2019-11-09-17-hmdb51-s2-MS3DRes101_hmdb51_opf_TVL1_init_imagenet_pretrained/MS3DRes101_hmdb51_opf_TVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('101',6,6),
        # 59.35-152
        #"2019-11-11-22-hmdb51-s2-MS3DRes152_hmdb51_opf_TVL1_init_imagenet_pretrained/MS3DRes152_hmdb51_opf_TVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('152',6,6),
        # 59.74-d169
        #"2019-11-11-15-hmdb51-s2-MS3DResd169_hmdb51_opf_TVL1_init_imagenet_pretrained/MS3DResd169_hmdb51_opf_TVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('d169',6,6)
    }
    s3_opf_hmdb_state_dict_path = {
        # 60.65-101
        "2019-11-16-21-hmdb51-s3-MS3DRes101_hmdb51_opf_TVL1_init_imagenet_pretrained/MS3DRes101_hmdb51_opf_TVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('101',6,6),
        # 61.96-152
        #"2019-11-21-23-hmdb51-s3-MS3DRes152_hmdb51_opf_TVL1_init_imagenet_pretrained/MS3DRes152_hmdb51_opf_TVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('152',6,6),
        # 61.31-d169
        #"2019-11-16-21-hmdb51-s3-MS3DResd169_hmdb51_opf_TVL1_init_imagenet_pretrained/MS3DResd169_hmdb51_opf_TVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('d169',6,6)
    }
    
    if split == 's1':
        opf_state_dict_path = {
            'hmdb51': s1_opf_hmdb_state_dict_path, 
            'ucf101':s1_opf_ucf_state_dict_path
        }.get(dataset)
    elif split == 's2':
        opf_state_dict_path = {
            'hmdb51': s2_opf_hmdb_state_dict_path, 
            'ucf101':s2_opf_ucf_state_dict_path
        }.get(dataset)
    elif split == 's3':
        opf_state_dict_path = {
            'hmdb51': s3_opf_hmdb_state_dict_path, 
            'ucf101':s3_opf_ucf_state_dict_path
        }.get(dataset)
    
    # warp opf
    s1_warp_ucf_state_dict_path = {
        # 85.54-101
        "2019-11-26-14-ucf101-s1-MS3DRes101_ucf101_opf_WarpTVL1_init_imagenet_pretrained/MS3DRes101_ucf101_opf_WarpTVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('101',6,6),
        # 85.43-152
        #"2020-03-15-11-ucf101-s1-MS3DRes152_ucf101_opf_WarpTVL1_init_imagenet_pretrained/MS3DRes152_ucf101_opf_WarpTVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('152',6,6),
        # 85.67-d169
        #"2020-02-28-00-ucf101-s1-MS3DResd169_ucf101_opf_WarpTVL1_init_imagenet_pretrained/MS3DResd169_ucf101_opf_WarpTVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('d169',6,6)
    }
    s2_warp_ucf_state_dict_path = {
        # 87.63-101
        "2020-02-17-16-ucf101-s2-MS3DRes101_ucf101_opf_WarpTVL1_init_imagenet_pretrained/MS3DRes101_ucf101_opf_WarpTVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('101',6,6),
        # 87.25-152
        #"2020-03-16-22-ucf101-s2-MS3DRes152_ucf101_opf_WarpTVL1_init_imagenet_pretrained/MS3DRes152_ucf101_opf_WarpTVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('152',6,6),
        # 87.98-d169
        #"2020-02-21-12-ucf101-s2-MS3DResd169_ucf101_opf_WarpTVL1_init_imagenet_pretrained/MS3DResd169_ucf101_opf_WarpTVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('d169',6,6)
    }
    s3_warp_ucf_state_dict_path = {
        # 86.85-101
        "2020-02-25-10-ucf101-s3-MS3DRes101_ucf101_opf_WarpTVL1_init_imagenet_pretrained/MS3DRes101_ucf101_opf_WarpTVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('101',6,6),
        # 88.85-152
        #"2020-03-24-12-ucf101-s3-MS3DRes152_ucf101_opf_WarpTVL1_init_imagenet_pretrained/MS3DRes152_ucf101_opf_WarpTVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('152',6,6),
        # 87.74-d169
        #"2020-02-29-16-ucf101-s3-MS3DResd169_ucf101_opf_WarpTVL1_init_imagenet_pretrained/MS3DResd169_ucf101_opf_WarpTVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('d169',6,6)
    }
    #ucf_state_dict_path = {
    #    "2019-04-11-10-ucf101-s1-MS3DRes101_ucf101_init_imagenet_pretrained/MS3DRes101_ucf101_init_imagenet_pretrained_best_acc_state_dict.pth":('101',6,6),
    #    "2019-03-29-ucf101-s1-MS3DRes152_ucf101_init_imagenet_pretrained/MS3DRes152_ucf101_init_imagenet_pretrained_best_acc_state_dict.pth":('152',6,6)
    #}
    s1_warp_hmdb_state_dict_path = {
        # 59.28-101
        "2020-02-11-18-hmdb51-s1-MS3DRes101_hmdb51_opf_WarpTVL1_init_imagenet_pretrained/MS3DRes101_hmdb51_opf_WarpTVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('101',6,6),
        # 59.34-152
        #"2020-02-23-23-hmdb51-s1-MS3DRes152_hmdb51_opf_WarpTVL1_init_imagenet_pretrained/MS3DRes152_hmdb51_opf_WarpTVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('152',6,6),
        # 60.26-d169
        #"2020-02-14-00-hmdb51-s1-MS3DResd169_hmdb51_opf_WarpTVL1_init_imagenet_pretrained/MS3DResd169_hmdb51_opf_WarpTVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('d169',6,6)
    }
    s2_warp_hmdb_state_dict_path = {
        # 60.65-101
        "2020-02-15-17-hmdb51-s2-MS3DRes101_hmdb51_opf_WarpTVL1_init_imagenet_pretrained/MS3DRes101_hmdb51_opf_WarpTVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('101',6,6),
        # 60.26-152
        #"2020-02-23-23-hmdb51-s2-MS3DRes152_hmdb51_opf_WarpTVL1_init_imagenet_pretrained/MS3DRes152_hmdb51_opf_WarpTVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('152',6,6),
        # 60.98-d169
        #"2020-02-17-22-hmdb51-s2-MS3DResd169_hmdb51_opf_WarpTVL1_init_imagenet_pretrained/MS3DResd169_hmdb51_opf_WarpTVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('d169',6,6)
    }
    s3_warp_hmdb_state_dict_path = {
        # 61.31-101
        "2020-02-19-21-hmdb51-s3-MS3DRes101_hmdb51_opf_WarpTVL1_init_imagenet_pretrained/MS3DRes101_hmdb51_opf_WarpTVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('101',6,6),
        # 60.78-152
        #"2020-04-03-19-hmdb51-s3-MS3DRes152_hmdb51_opf_WarpTVL1_init_imagenet_pretrained/MS3DRes152_hmdb51_opf_WarpTVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('152',6,6),
        # 62.68-d169
        #"2020-02-21-22-hmdb51-s3-MS3DResd169_hmdb51_opf_WarpTVL1_init_imagenet_pretrained/MS3DResd169_hmdb51_opf_WarpTVL1_init_imagenet_pretrained_best_acc_state_dict.pth":('d169',6,6)
    }
    
    if split == 's1':
        warp_state_dict_path = {
            'hmdb51': s1_warp_hmdb_state_dict_path, 
            'ucf101':s1_warp_ucf_state_dict_path
        }.get(dataset)
    elif split == 's2':
        warp_state_dict_path = {
            'hmdb51': s2_warp_hmdb_state_dict_path, 
            'ucf101':s2_warp_ucf_state_dict_path
        }.get(dataset)
    elif split == 's3':
        warp_state_dict_path = {
            'hmdb51': s3_warp_hmdb_state_dict_path, 
            'ucf101':s3_warp_ucf_state_dict_path
        }.get(dataset)
#    opf_state_dict_path = {
#            'hmdb51': opf_hmdb_state_dict_path, 
#            'ucf101':opf_ucf_state_dict_path
#    }.get(dataset)

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
        }

#   config train param
    weight_decay = 1e-5
    torch.nn.Module.dump_patches = True
#   config model
    net_depth = args.net_depth
    pretrained_dataset = "imagenet"

    rgb_models = []
    opf_models = []
    warp_models = []
    # pretrained_dataset = "kinetics"
    if args.n_gpu == 1:
        model = model_factory(
                depth=net_depth, num_classes=num_classes, t_dim=args.anchor_num,
                duration=args.anchor_duration, pretrained=False,
                pretrained_dataset=pretrained_dataset
                ).cuda(gpu_id)
    else:
        gpu_id = None
        # rgb
        for subpath, params in rgb_state_dict_path.items():
            sub_model_path = os.path.join(model_path, subpath)
            net_depth = params[0]
            #sub_model_path = os.path.join(sub_model_path, model_name)
            model = model_factory(
                    depth=net_depth, num_classes=num_classes, t_dim=args.anchor_num,
                    duration=args.anchor_duration, pretrained=False,
                    pretrained_dataset=pretrained_dataset, use_opf=False
                    ).cuda()
            model = torch.nn.DataParallel(
                    model, device_ids=list(range(args.n_gpu))
                    )
            model.load_state_dict(torch.load(sub_model_path))
            rgb_models.append(model)
        print("**RGB Model loaded")
        # opf models
        for subpath, params in opf_state_dict_path.items():
            sub_model_path = os.path.join(model_path, subpath)
            net_depth = params[0]
            #sub_model_path = os.path.join(sub_model_path, model_name)
            model = model_factory(
                    depth=net_depth, num_classes=num_classes, t_dim=args.anchor_num,
                    duration=args.anchor_duration, pretrained=False,
                    pretrained_dataset=pretrained_dataset, use_opf=True
                    ).cuda()
            model = torch.nn.DataParallel(
                    model, device_ids=list(range(args.n_gpu))
                    )
            model.load_state_dict(torch.load(sub_model_path))
            opf_models.append(model)
        print("**OPF Model loaded")
        # warp models
        for subpath, params in warp_state_dict_path.items():
            sub_model_path = os.path.join(model_path, subpath)
            net_depth = params[0]
            #sub_model_path = os.path.join(sub_model_path, model_name)
            model = model_factory(
                    depth=net_depth, num_classes=num_classes, t_dim=args.anchor_num,
                    duration=args.anchor_duration, pretrained=False,
                    pretrained_dataset=pretrained_dataset, use_opf=True
                    ).cuda()
            model = torch.nn.DataParallel(
                    model, device_ids=list(range(args.n_gpu))
                    )
            model.load_state_dict(torch.load(sub_model_path))
            warp_models.append(model)
        print("**WARP Model loaded")

    for model in rgb_models:
        model.eval()
    
    
    for model in opf_models:
        model.eval()
    

    ensemble_models = {
        "rgb": rgb_models,
        "opf": opf_models,
        "warp":warp_models,
        #"opf": warp_models,
    }

    if len(ensemble_models) == 2:
        print("**2-stream model")
        stream_num = 2
        net_nums = len(ensemble_models["rgb"]) + len(ensemble_models["opf"])
    elif len(ensemble_models) == 3:
        print("**3-stream model")
        stream_num = 3
        net_nums = len(ensemble_models["rgb"]) + len(ensemble_models["opf"]) + len(ensemble_models["warp"])
#   dataloader params
    dataloader_kws = {
        'dataset': dataset, 'subset':subset, 'split': split,
        'batch_size': batch_size, 'num_workers': num_workers,
        'drop_last': drop_last, 'IS_ENSEMBLE': True,
        'spatial_transforms_param': spatial_transforms_param,
        'temporal_transforms_param': temporal_transforms_param,
        'anchor_transforms_param': anchor_transforms_param,
        'opf_transforms_param': opf_transforms_param,
        'stream_num':stream_num
    }

#   engine params
    #net_nums = len(opf_models) + len(rgb_models)
    net_name = 'EnsembleOpfMS3D{}_{:s}_test'.format(
                net_nums, dataset)

    log_subdir = "{:s}-{:s}-{:s}-{:s}".format(
            time.strftime("%Y-%m-%d"), dataset, split, net_name)
    model_subdir = "{:s}-{:s}-{:s}-{:s}".format(
            time.strftime("%Y-%m-%d"), dataset, split, net_name)

    log_dir = "{:s}/{:s}".format(args.log_dir, log_subdir)

    loss_func = {
        "cross_entropy": F.cross_entropy
    }.get(args.loss_func)

    Res3DEngine_params = {
        'model': ensemble_models, 'optimizer': None, 'maxepoch': None,
        'loss_func': loss_func, 'batch_size': batch_size,
        'num_workers': num_workers, 'net_name': net_name,
        'snap_epoch':None,
        'model_dir': None, 'logs_dir': log_dir, 'gpu_id': gpu_id,
        'loss_scheduler':None, 'step_scheduler':None,
        'sgdr': None, 'init_lr': None, 'T_max': None
    }

#   torch config
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True #不加则显存会爆掉
    torch.backends.cudnn.deterministic = True
    
    torch.manual_seed(4)
    torch.cuda.manual_seed(4)

#   engine config
    Res3D_engine = Res3DEngine(
            init_kws=Res3DEngine_params, dataloader_kws=dataloader_kws,
            accumulate_step=1)
    Res3D_engine.init_module(
        AccumulateEngine(), EpochMeter(num_classes=num_classes),
        EpochRecorder(record_step='{:s}_epoch'.format(dataset),
                      root_dir=log_dir),
        EpochLogger(num_classes, title='TestEnsOpfMS3D{}_{:s}'.format(
                net_nums, dataset)),
        BatchLogger(title='TestEnsOpfMS3D{}_{:s}'.format(
                net_nums, dataset))
    )

    args.accumulate_batch = batch_size*1

    args_file = os.path.join(log_dir, "Test_Args_{:s}.txt".format(net_name))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(args_file, 'w') as f:
        f.write("Args Name: Args Param\n")
        for args_name, args_param in vars(args).items():
            f.write("{:s}: {}\n".format(args_name, args_param))
        f.write("{:s}: {}\n".format('input_mean', input_mean))
        f.write("{:s}: {}\n".format('input_std', input_std))
        f.write("{:s}: {}\n".format('net parameters', Res3D_engine.parameters))
        f.write("{:s}: {}\n".format('scales', scales))

    print("**Net parameters: {}".format(Res3D_engine.parameters))
    print_args_table(vars(args))

    if use_anchor:
        temporal_size = args.anchor_num * args.anchor_duration
        print('**Temporal size: {}'.format(temporal_size))
    else:
        temporal_size = args.temporal_size
        
#    summary_file = os.path.join(log_dir, "Test_Summary_{:s}.txt".format(net_name))
#    lines = summary(model, (3, temporal_size, args.im_size, args.im_size))
#    torch.cuda.empty_cache()
#    with open(summary_file, 'w') as f:
#        for summary_line in lines:
#            f.write(summary_line+"\n")

    Res3D_engine.test()
    loss = Res3D_engine.epoch_meters.loss
    accuracy = Res3D_engine.epoch_meters.accuracy
    confusion = Res3D_engine.epoch_meters.confusion
    print_meters(epoch=1, loss=loss, accuracy= accuracy, train=False)

    ## save in txt
    end_time = get_time()
    result = '[{:s}][Epoch {:02d}] {:s} Loss: {:.4f} (Accuracy: {:.2f}%)\n'.format(
            end_time, 1, 'Test', loss, accuracy)
    result_file = os.path.join(log_dir, 'Test_at_{:s}'.format(end_time))
    with open(result_file, 'w') as f:
        # result
        f.write(result)
        f.write("="*70+"\n")
        # paramters
        f.write("Args Name: Args Param\n")
        for args_name, args_param in vars(args).items():
            f.write("{:s}: {}\n".format(args_name, args_param))
        f.write("{:s}: {}\n".format('input_mean', input_mean))
        f.write("{:s}: {}\n".format('input_std', input_std))
        f.write("{:s}: {}\n".format('scales', scales))
        f.write("{:s}: {}\n".format('state_dict', rgb_state_dict_path))
        f.write("{:s}: {}\n".format('state_dict', opf_state_dict_path))
        f.write("{:s}: {}\n".format('state_dict', warp_state_dict_path))
        f.write("{:s}: {}\n".format('stream_num', stream_num))
        f.write("{:s}: {}\n".format('connection fc type', cfg.FC_TYPE))
        f.write("="*70+"\n")
        # net summary
#        for summary_line in lines:
#            f.write(summary_line+"\n")

    conf_df = DataFrame(confusion)
    conf_df.to_csv(
            os.path.join(
                log_dir, 'TestConfusion_at_{:s}.csv'.format(end_time)
            )
    )
    print("[{:s}] Result File Saved in {:s}".format(get_time(), result_file))

