#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===============================================================
# Copyright (C) 2018 HuangYk.
# Licensed under The MIT Lincese.
#
# Filename     : ucf101.py
# Author       : HuangYK
# Last Modified: 2018-09-23 21:22
# Description  :
# ===============================================================


import os
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from sklearn.externals import joblib #模型保存与加载

from datasets.optic_flow_loader import construct_opticflow_image
from datasets.video_loader import make_dataset
from datasets.video_loader import get_default_video_loader


class UCF101(data.Dataset):
    '''UCF101 dataset of 3 splits

    Args:
    -----
    subset: str, ['train', 'test', 'val'] dataset type
    split: str, ['s1', 's2', 's3'], splits of dataset
    n_samples_for_each_video: int, sample number from a video
    sample_duration: int, sample indices number if n_samples_for_each_video
                     is not one
    spatial_transform: callable instance, same as attribute
    temporal_transform: callable instance, same as attribute
    target_transform: callable instance, same as attribute
    is_ensemble: use for different streams ensemble

    Attribute:
    ----------
    data: dict, annotation of subset split
    idx_to_class: dict, key is idx, value id class name
    spatial_transform: callable instance, takes in images returns transformed
                       version
    temporal_transform: callable instance, takes in indices returns transformed
                        version
    target_transform: callable instance, takes in targets returns transformed
                      version
    video_loader: fucntion, get image clips by video dir and frame indices
    '''
    def __init__(self, subset, split,
                 n_samples_for_each_video=1, sample_duration=16,
                 spatial_transform=None, temporal_transform=None,
                 target_transform=None, opf_spatial_transform=None, is_ensemble=False, 
                 is_idt=False, opf_temporal_transform=None,opt_type=0, opt_stream_num=2):
        '''
        '''
        self.data, self.idx_to_class = make_dataset(
            'ucf101', subset, split, n_samples_for_each_video, sample_duration)
        
        self.is_ensemble = is_ensemble
        if is_idt:
            self.pca_transform = joblib.load(self.data[0]['pca_transform'])
        else:
            self.pca_transform = None
        self.spatial_transform = spatial_transform
        self.opf_spatial_transform = opf_spatial_transform
        self.opf_temporal_transform = opf_temporal_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.opt_type = opt_type
        self.opt_stream_num = opt_stream_num
        self.opt_kws = {0:"TVL1",1:"WarpTVL1"}
        self.video_loader = get_default_video_loader()

    def __getitem__(self, index):
        '''return video frame images and target by [] or iterable call

        Args:
        -----
        index: int, sample index

        Return:
        -------
        clips: tensor, video frame images in [C, N, H, W]
               C--Channel, N--frame num, H--frame height, W--frame width
        target: int, video class cid num
        '''
        video_path = self.data[index]['dir']

        all_frame_indices = list(self.data[index]['indices'][0])
        if self.temporal_transform:
            frame_indices = self.temporal_transform(all_frame_indices)
            if self.opf_temporal_transform:
                opt_indices = self.opf_temporal_transform(all_frame_indices)
            else:
                opt_indices = frame_indices
        
        target = self.data[index]['cid']
        if self.target_transform:
            target = self.target_transform(target)
        
        # Optic Flow
        if self.opf_spatial_transform:
            opt_clips = self.video_loader(
                    video_path, 
                    opt_indices,
                    USE_OPF=True,
                    OPTIC_TYPE=self.opt_kws[self.opt_type]
            )
            self.opf_spatial_transform.randomize_parameters()
            opt_clips = [torch.cat(
                (self.opf_spatial_transform(opt_x), 
                 self.opf_spatial_transform(opt_y)), 
                0) for opt_x, opt_y in opt_clips]
#            opt_clips = [self.opf_spatial_transform(
#                construct_opticflow_image(opt_x, opt_y, 'TV-L1')
#            ) for opt_x, opt_y in opt_clips]
            #flow_color = flow_vis.flow_to_color(flow_out, convert_to_bgr=False)

            opt_clips = torch.stack(opt_clips, 0).permute(1, 0, 2, 3)    
            if not self.is_ensemble:
                return opt_clips, target
            if self.opt_stream_num == 3:
                warp_clips = self.video_loader(
                        video_path, 
                        opt_indices,
                        USE_OPF=True,
                        OPTIC_TYPE=self.opt_kws[1]
                )
                self.opf_spatial_transform.randomize_parameters()
                warp_clips = [torch.cat(
                    (self.opf_spatial_transform(opt_x), 
                     self.opf_spatial_transform(opt_y)), 
                    0) for opt_x, opt_y in warp_clips]
                warp_clips = torch.stack(warp_clips, 0).permute(1, 0, 2, 3)
        
        # RGB
        clips = self.video_loader(video_path, frame_indices)
        if self.spatial_transform:
            self.spatial_transform.randomize_parameters()
            clips = [self.spatial_transform(img) for img in clips]
        clips = torch.stack(clips, 0).permute(1, 0, 2, 3)
            
        if self.pca_transform:
            idt_sample = self.make_FV_matrix(self.data[index]['idt_fisher'])
            idt_sample = idt_sample.reshape(1, idt_sample.shape[0])
            idt_pca = self.pca_transform.transform(idt_sample)
            if not self.is_ensemble:
                return clips, idt_pca, target
        
        if self.is_ensemble:
            if self.opt_stream_num <=2:
                return clips, opt_clips, target
            elif self.opt_stream_num ==3:
                return clips, opt_clips, warp_clips, target
        else:
            return clips, target


    def __len__(self):
        return len(self.data)
    
    def make_FV_matrix(self, fisher_path):
        vid_path = os.path.join(fisher_path)
        return np.load(vid_path)['fish']

    def get_video_image(self, video_idx, indice_idx, mean, std):
        '''return a frame image of sample[video_idx], indice[indice_idx]

        Args:
        -----
        video_idx: int, sample index
        indice_idx: int, frame index

        Return:
        -------
        image: PIL.Image, frame image
        class_name: str, sample cid corresponding class name
        '''
        if self.pca_transform:
            clips, idt_pca, target = self[video_idx]
        elif self.opf_spatial_transform:
            clips, opt_clips, target = self[video_idx]
        else:
            clips, target = self[video_idx]
        clips = clips.permute(1, 0, 2, 3)
        indice_idx = indice_idx \
            if indice_idx < clips.shape[0] else clips.shape[0]-1

        pil_transform = transforms.ToPILImage()
        image_tensor = clips[indice_idx]
        for t, m, s in zip(image_tensor, mean, std):
            t.mul_(s).add_(m)
        image = pil_transform(image_tensor)
        class_name = self.idx_to_class[target]
        return image, class_name
