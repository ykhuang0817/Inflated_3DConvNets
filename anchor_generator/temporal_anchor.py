#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===============================================================
# Copyright (C) 2018 HuangYk.
# Licensed under The MIT Lincese.
#
# Filename     : temporal_anchor.py
# Author       : HuangYK
# Last Modified: 2019-03-25 20:26
# Description  :
# ===============================================================


from __future__ import print_function, division, absolute_import
import numpy as np
from numpy.random import randint


class TemporalCenterAnchor(object):
    """
    """
    def __init__(self, anchor_num, duration):
        self.anchor_num = anchor_num
        self.duration = duration

    def __call__(self, frame_indices):
        vid_length = len(frame_indices)
        anchor_interval = vid_length // self.anchor_num
        # print('video base: {}, {}'.format(anchor_interval, vid_length))
        if self.duration > len(frame_indices):
            anchor_out = frame_indices
            for index in anchor_out:
                if len(anchor_out) >= self.duration:
                    break
                anchor_out.append(index)
            out = []
            [out.extend(anchor_out) for i in range(self.anchor_num)]
            # print('out: {}'.format(out))
            return out
        else:
            anchor_range = [[max(1, i*anchor_interval//2 - self.duration//2),
                             min(vid_length,
                                 i*anchor_interval//2 + self.duration//2)]
                            for i in range(1, self.anchor_num+1)]
            # print('anchor_range: {}'.format(anchor_range))
            for i in range(len(anchor_range)):
                if anchor_range[i][0] == 1:
                    anchor_range[i][1] = 1 + self.duration
            out = []
            if self.duration == 1:
                [out.extend([anchor_start])
                 for anchor_start, anchor_end in anchor_range]
            else:
                [out.extend(range(anchor_start, anchor_end))
                 for anchor_start, anchor_end in anchor_range]
            # print('out: {}'.format(out))
            return out


class TemporalBeginAnchor(object):
    """
    """
    def __init__(self, anchor_num, duration):
        self.anchor_num = anchor_num
        self.duration = duration

    def __call__(self, frame_indices):
        vid_length = len(frame_indices)
        anchor_interval = vid_length // self.anchor_num
        if self.duration > len(frame_indices):
            anchor_out = frame_indices
            for index in anchor_out:
                if len(anchor_out) >= self.duration:
                    break
                anchor_out.append(index)
            out = []
            [out.extend(anchor_out) for i in range(self.anchor_num)]
            return out
        else:
            anchor_range = [[1+i*anchor_interval,
                             min(vid_length+1,
                                 1+i*anchor_interval+self.duration)]
                            for i in range(self.anchor_num)]
            out = []
            for anchor_start, anchor_end in anchor_range:
                anchor_slices = []
                if anchor_end == vid_length+1:
                    anchor_slices = range(anchor_start, anchor_end)
                    for index in anchor_slices:
                        if len(anchor_slices) >= self.duration:
                            break
                        anchor_slices.append(index)
                else:
                    anchor_slices = range(anchor_start, anchor_end)
                out.extend(anchor_slices)
            return out   


class TemporalRandomAnchor(object):
    """
    """
    def __init__(self, anchor_num, duration, slice_distribution='unit'):
        self.anchor_num = anchor_num
        self.duration = duration
        self.slice_distribution = slice_distribution

    def __call__(self, frame_indices):
        vid_length = len(frame_indices)
        anchor_interval = vid_length // self.anchor_num
        if self.duration > len(frame_indices):
            anchor_out = frame_indices
            for index in anchor_out:
                if len(anchor_out) >= self.duration:
                    break
                anchor_out.append(index)
            out = []
            [out.extend(anchor_out) for i in range(self.anchor_num)]
            return out
        else:
#            assert (anchor_interval-1)>0, 'anchor_interval:{},vid_length:{}'.format(
#                    anchor_interval, vid_length)
            random_offset = anchor_interval-1 if (anchor_interval-1)>0 else 1
            anchor_range = [[1+i*anchor_interval,
            				 choose_slice(random_offset, self.slice_distribution)
                             ]
                            for i in range(self.anchor_num)]
            out = []
            for interval_start, anchor_offset in anchor_range:
                anchor_slices = []
                # generate random anchor
                anchor_start = interval_start+anchor_offset
                anchor_start = 1 if anchor_start == 0 else anchor_start
                anchor_end = anchor_start+self.duration
                if anchor_end >= vid_length+1:
                    anchor_slices = list(range(anchor_start, vid_length+1))
                    for index in anchor_slices:
                        if len(anchor_slices) >= self.duration:
                            break
                        anchor_slices.append(index)
                else:
                    anchor_slices = range(anchor_start, anchor_end)
                out.extend(anchor_slices)
            return out


class TemporalConseqAnchor(object):
    """
    """
    def __init__(self, anchor_num, duration, slice_distribution='unit'):
        self.anchor_num = anchor_num
        self.duration = duration
        self.slice_distribution = slice_distribution

    def __call__(self, frame_indices):
        vid_length = len(frame_indices)
        anchor_interval = vid_length
        if self.duration > len(frame_indices):
            anchor_out = frame_indices
            for index in anchor_out:
                if len(anchor_out) >= self.duration:
                    break
                anchor_out.append(index)
            out = []
            [out.extend(anchor_out) for i in range(self.anchor_num)]
            return out
        else:
            if (anchor_interval-self.duration*self.anchor_num)>0:
                random_offset = anchor_interval-self.duration*self.anchor_num
            else:
                random_offset = 1
            
            anchor_start = choose_slice(random_offset, self.slice_distribution)
            anchor_start = anchor_start if anchor_start>0 else 1
            anchor_end = anchor_start+self.duration*self.anchor_num
            if anchor_end >= vid_length+1:
                anchor_slices = list(range(anchor_start, vid_length+1))
                for index in anchor_slices:
                    if len(anchor_slices) >= self.duration*self.anchor_num:
                        break
                    anchor_slices.append(index)
            else:
                anchor_slices = range(anchor_start, anchor_end)
                
            return anchor_slices 

            
def choose_slice(slice_range, distribution='unit'):
    assert distribution in ['unit', 'norm']
    if distribution == 'unit':
        return randint(slice_range)
    elif distribution == 'norm':
        return int((np.random.randn(1)*0.6+slice_range/2)[0])
