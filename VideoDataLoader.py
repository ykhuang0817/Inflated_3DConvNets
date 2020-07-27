#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===============================================================
# Copyright (C) 2018 HuangYk.
# Licensed under The MIT Lincese.
#
# Filename     : VideoDataLoader.py
# Author       : HuangYK
# Last Modified: 2018-11-20 14:45
# Description  :
# ===============================================================


from __future__ import print_function, division

import torch

from config.config import cfg
from datasets.hmdb51 import HMDB51
from datasets.ucf101 import UCF101
from datasets.ucf50 import UCF50
from video_transform import spatial_transforms, temporal_transforms
from anchor_generator.temporal_anchor import TemporalCenterAnchor
from anchor_generator.temporal_anchor import TemporalBeginAnchor
from anchor_generator.temporal_anchor import TemporalRandomAnchor
from anchor_generator.temporal_anchor import TemporalConseqAnchor


def multiscale_crop_factory(crop_method, scales, size):
    assert crop_method in ['random', 'corner', 'center']
    crop_class, param = {
        'random': (spatial_transforms.MultiScaleRandomCrop, {
            'scales': scales, 'size': size}),
        'corner': (spatial_transforms.MultiScaleCornerCrop, {
            'scales': scales, 'size': size}),
        'center': (spatial_transforms.MultiScaleCornerCrop, {
            'scales': scales, 'size': size, 'crop_positions': ['c']})
    }.get(crop_method)
    return crop_class(**param)


def crop_factory(crop_method, size, crop_pos='c'):
    assert crop_method in ['random', 'corner', 'center']
    crop_class, param = {
        'random': (spatial_transforms.RandomCrop, {
            'size': size}),
        'corner': (spatial_transforms.CornerCrop, {
            'size': size, 'crop_position': crop_pos}),
        'center': (spatial_transforms.CenterCrop, {
            'size': size})
    }.get(crop_method)
    return crop_class(**param)


def get_dataloader(dataset, subset, split, batch_size=20, num_workers=4,
                   drop_last=True, IS_ENSEMBLE=False, IS_IDT=False,
                   spatial_transforms_param=None,
                   opf_transforms_param = None,
                   temporal_transforms_param=None,
                   anchor_transforms_param=None):
    assert dataset in ['hmdb51', 'ucf101', 'ucf50']

    dataset_class = {'hmdb51': HMDB51, 'ucf101': UCF101, 'ucf50':UCF50}.get(dataset)

    assert isinstance(spatial_transforms_param, dict)
    transform_in_spatial = get_spatial_transform(**spatial_transforms_param)

    if anchor_transforms_param is not None:
        assert isinstance(anchor_transforms_param, dict)
        transform_in_temporal = get_temporal_anchor_transform(
            **anchor_transforms_param
        )
        opf_in_temporal = None
#        opf_in_temporal = get_temporal_anchor_transform(
#            anchor_method=anchor_transforms_param['anchor_method'],
#            anchor_num=anchor_transforms_param['anchor_num'],
#            anchor_duration=anchor_transforms_param['anchor_duration'],
#            slice_distribution='unit'
#        )
        if opf_in_temporal:
            print("**Use Diff Temporal Transform")
        else:
            print("**Use Same Temporal Transform")
    else:
        assert isinstance(temporal_transforms_param, dict)
        transform_in_temporal = get_temporal_transform(
            **temporal_transforms_param)

    if opf_transforms_param:
        assert isinstance(opf_transforms_param, dict)
        opf_transforms_param["subset"] = "opf"
        transform_in_opticflow = get_spatial_transform(**opf_transforms_param)
    else:
        transform_in_opticflow = None

    kwargs = {'num_workers': num_workers, 'pin_memory': True,
              'drop_last': drop_last}
    
    if cfg.OPTIC_FLOW_TYPE=="TVL1":
        opt_type_key = 0
    elif cfg.OPTIC_FLOW_TYPE=="WarpTVL1":
        opt_type_key = 1
    
    data_loader = torch.utils.data.DataLoader(
        dataset_class(
            subset, split, spatial_transform=transform_in_spatial,
            opf_spatial_transform=transform_in_opticflow,
            opf_temporal_transform = opf_in_temporal,
            temporal_transform=transform_in_temporal,
            is_ensemble=IS_ENSEMBLE, is_idt=IS_IDT, opt_type=opt_type_key
            ),
        batch_size=batch_size, shuffle=True, **kwargs
    )

    return data_loader

def get_ensemble_dataloader(dataset, subset, split, batch_size=20, num_workers=4,
                   drop_last=True, IS_ENSEMBLE=False, IS_IDT=False,
                   spatial_transforms_param=None,
                   opf_transforms_param = None,
                   temporal_transforms_param=None,
                   anchor_transforms_param=None,stream_num=2):
    assert dataset in ['hmdb51', 'ucf101', 'ucf50']

    dataset_class = {'hmdb51': HMDB51, 'ucf101': UCF101, 'ucf50': UCF50}.get(dataset)

    assert isinstance(spatial_transforms_param, dict)
    transform_in_spatial = get_spatial_transform(**spatial_transforms_param)

    if anchor_transforms_param is not None:
        assert isinstance(anchor_transforms_param, dict)
        transform_in_temporal = get_temporal_anchor_transform(
            **anchor_transforms_param
        )
        opf_in_temporal = None
        opf_in_temporal = get_temporal_anchor_transform(
            anchor_method=anchor_transforms_param['anchor_method'],
            anchor_num=anchor_transforms_param['anchor_num'],
            anchor_duration=anchor_transforms_param['anchor_duration'],
            slice_distribution='unit'
        )
        if opf_in_temporal:
            print("**Use Diff Temporal Transform")
        else:
            print("**Use Same Temporal Transform")
    else:
        assert isinstance(temporal_transforms_param, dict)
        transform_in_temporal = get_temporal_transform(
            **temporal_transforms_param)

    if opf_transforms_param:
        assert isinstance(opf_transforms_param, dict)
        opf_transforms_param["subset"] = "opf"
        transform_in_opticflow = get_spatial_transform(**opf_transforms_param)
    else:
        transform_in_opticflow = None

    kwargs = {'num_workers': num_workers, 'pin_memory': True,
              'drop_last': drop_last}
    
    if cfg.OPTIC_FLOW_TYPE=="TVL1":
        opt_type_key = 0
    elif cfg.OPTIC_FLOW_TYPE=="WarpTVL1":
        opt_type_key = 1

    data_loader = torch.utils.data.DataLoader(
        dataset_class(
            subset, split, spatial_transform=transform_in_spatial,
            opf_spatial_transform=transform_in_opticflow,
            opf_temporal_transform = opf_in_temporal,
            temporal_transform=transform_in_temporal,opt_type=opt_type_key,
            is_ensemble=IS_ENSEMBLE, is_idt=IS_IDT, opt_stream_num=stream_num
            ),
        batch_size=batch_size, shuffle=True, **kwargs
    )

    return data_loader


def get_spatial_transform(subset='train', method='random',
                          scales=[1.0, 0.875, 0.75, 0.66],
                          im_size=224, norm_value=255,
                          input_mean=[0.485, 0.456, 0.406],
                          input_std=[0.229, 0.224, 0.225]
                          ):
    assert subset in ['train', 'test', 'val', 'opf']
    transform = {
        'train': spatial_transforms.Compose([
            # spatial_transforms.Scale(size=(324, 256)),
            multiscale_crop_factory(method, scales, im_size),
            spatial_transforms.RandomHorizontalFlip(),
            spatial_transforms.ToTensor(norm_value),
            spatial_transforms.Normalize(input_mean, input_std)
        ]),
        'test': spatial_transforms.Compose([
            spatial_transforms.Scale(size=(324, 256)),
            crop_factory('center', im_size),
            spatial_transforms.ToTensor(norm_value),
            spatial_transforms.Normalize(input_mean, input_std)
        ])
    }

    if subset == 'train':
        transform_in_spatial = transform['train']
        print('**Train Transform')
    elif subset == 'test':
        transform_in_spatial = transform['test'] if not \
            cfg.SAME_TRANSFORM else transform['train']
        print('**Test Transform')
    elif subset == 'opf':
        transform_in_spatial = transform['test'] if not \
            cfg.SAME_TRANSFORM else transform['train']
        print('**{} Opf Transform'.format(cfg.OPTIC_FLOW_TYPE))

    print('**Use Same Transform') if \
        cfg.SAME_TRANSFORM else print('**Use Diff Transform')

    return transform_in_spatial


def get_temporal_transform(crop_method='random', size=16, **kwargs):
    assert crop_method in ['random', 'begin', 'center', 'randomstep']
    transform = {
        'random': (temporal_transforms.TemporalRandomCrop, {'size': size}),
        'begin': (temporal_transforms.TemporalBeginCrop, {'size': size}),
        'center': (temporal_transforms.TemporalCenterCrop, {'size': size}),
        'randomstep': (temporal_transforms.TemporalRandomStepCrop, {
            'size': size, 'step_method': kwargs['step_method'],
            'step': kwargs['step']}
        )
    }

    if kwargs['subset'] == 'train':
        crop_class, param = transform[crop_method]
    elif kwargs['subset'] == 'test':
        crop_class, param  = transform['random'] if not \
            cfg.SAME_TRANSFORM else transform[crop_method]

    return crop_class(**param)


def get_temporal_anchor_transform(anchor_method, anchor_num, anchor_duration,
                                  **kwargs):
    # TODO{ykh}: need random class
    assert anchor_method in ['begin', 'center', 'random', 'conseq']
    anchor_class, param = {
        'begin': (TemporalBeginAnchor,
                  {'anchor_num': anchor_num,
                   'duration': anchor_duration}),
        'center': (TemporalCenterAnchor,
                   {'anchor_num': anchor_num,
                    'duration': anchor_duration}),
        'random': (TemporalRandomAnchor,
                   {'anchor_num': anchor_num,
                    'duration': anchor_duration,
                    'slice_distribution':kwargs['slice_distribution']}),
        'conseq': (TemporalConseqAnchor,
                   {'anchor_num': anchor_num,
                    'duration': anchor_duration,
                    'slice_distribution':kwargs['slice_distribution']})
    }.get(anchor_method)

    return anchor_class(**param)
