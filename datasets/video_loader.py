#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===============================================================
# Copyright (C) 2018 HuangYk.
# Licensed under The MIT Lincese.
#
# Filename     : video_loader.py
# Author       : HuangYK
# Last Modified: 2018-09-18 20:43
# Description  :
# ===============================================================


import numpy as np
import pandas as pd
import glob
from PIL import Image
import functools
import os
import math
import copy

from config.config import cfg


annotation_root_dir = cfg.ANNOTATION_ROOT


def load_annotation(db):
    '''load csv annotation and return a split dataframe

    Annotation contains [sample, cid, class, db, dir, num, split, tag, vid]
    columns.

    Args:
    -----
    db: str, ['hmdb51', 'ucf101'], annotation used

    Return:
    -------
    annotation_splits: dict contain 3 splits dataframe, db data
    annotation_class: dataframe, db class to idx
    '''
    assert db in ['hmdb51', 'ucf101', 'ucf50']
    csv_annotation = glob.glob('{:s}/{:s}_s*.csv'.format(
        annotation_root_dir, db))
    csv_class = glob.glob('{:s}/{:s}_c*.csv'.format(annotation_root_dir, db))
    splits = ['s1', 's2', 's3'] if db is not 'ucf50' else ['s1','s2','s3']
    annotation_splits = {}
    for split in splits:
        csv_file = [anno for anno in csv_annotation if split in anno]
        #print(csv_file)
        assert len(csv_file) == 1, 'No such split in annotation'
        annotation_splits[split] = pd.read_csv(csv_file[0])
    assert len(csv_class) == 1
    annotation_class = pd.read_csv(csv_class[0])
    drop_column = [
        col for col in annotation_class.columns if col not in ['class', 'idx']
    ][0]
    annotation_class = annotation_class.drop(columns=drop_column)
    return annotation_splits, annotation_class


def get_subset_annotation(annotation, subset):
    '''split datset annotation to test, train, val

    Args:
    -----
    annotation: dataframe, all data from a dataset
    subset: str, ['train', 'test', 'val']

    Return:
    -------
    subset_annotation: dataframe, subset of annotation fit [tag]
    '''
    assert subset in ['train', 'test', 'val']
    return annotation.loc[lambda df: df.tag == subset]


def pil_loader(path):
    '''load image in pil
    '''
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    '''load image in accimage
    '''
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    '''image_loader generation
    '''
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader
    


def video_loader(video_dir_path, frame_indices, image_loader, USE_OPF=False, 
                 OPTIC_TYPE=cfg.OPTIC_FLOW_TYPE):
    '''load video in slices way

    Args:
    -----
    video_dir_path: str, video frames folder
    frame_indices: np.array or list, frame slices
    image_loader: function, provide image load way
    '''
    video = []
    opt_video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'img_{:05d}.jpg'.format(i))
        # Optic flow ['TVL1', 'WarpTVL1']
        if USE_OPF:
            if OPTIC_TYPE == 'TVL1':
                flow_x_dir = '/flow_x'
                flow_y_dir = '/flow_y'
            elif OPTIC_TYPE == 'WarpTVL1':
                flow_x_dir = '/warp_flow_x'
                flow_y_dir = '/warp_flow_y'
            elif OPTIC_TYPE == 'Liteflow':
                liteflow_dir = '/lite_flow'
                #liteflow_data = np.load(liteflow_path)
                # todo
            #print(flow_x_dir)
            if cfg.OPTIC_FLOW_TYPE == 'Liteflow':
                liteflow_path = image_path.replace(
                        '/img', liteflow_dir, 1
                        ).replace('img', 'liteflow').replace('.jpg', '.npy')
                if os.path.exists(liteflow_path):
                    lite_flow = np.load(liteflow_path)
                    lite_flow_x = Image.fromarray(
                            np.floor(255*(lite_flow[:,:,0]/2+0.5)), mode='F'
                            ).convert('I')
                    lite_flow_y = Image.fromarray(
                            np.floor(255*(lite_flow[:,:,1]/2+0.5)), mode='F'
                            ).convert('I')
                    opt_video.append(
                        (lite_flow_x, lite_flow_y)
                    )
                else:
                    print('No data find in {}'.format(liteflow_path))
                    return opt_video
            else:
                opt_ims = {
                    'flow_x': image_path.replace('/img', flow_x_dir),
                    'flow_y': image_path.replace('/img', flow_y_dir)
                }
                if os.path.exists(opt_ims['flow_x']) \
                        and os.path.exists(opt_ims['flow_y']):
                    opt_video.append((
                        image_loader(opt_ims['flow_x']).convert('I'),
                        image_loader(opt_ims['flow_y']).convert('I')
                        )
                    )
                    #opt_video.append((opt_ims['flow_x'], opt_ims['flow_y']))
                else:
                    print('No video image find in {}'.format(image_path))
                    return opt_video
        # RGB
        else:
            if os.path.exists(image_path):
                video.append(image_loader(image_path))
            else:
                print('No video image find in {}'.format(image_path))
                return video
    if USE_OPF:
        return opt_video
    else:
        return video


def get_default_video_loader():
    '''video loader interface
    '''
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def create_indices(num_frames, n_samples_for_each_video, sample_duration):
    '''make slices of video

    Args:
    -----
    num_frames: int, frames of video
    n_samples_for_each_video: int, sample numbers of a video
    sample_duration: int, frame numbers of a sample

    Return:
    -------
    indices: nd.array, [n_samples_for_each_video, sample_duration], slices list
    '''
    if n_samples_for_each_video == 1:
        return (np.arange(num_frames).reshape(1, -1)+1).astype(int)
    elif n_samples_for_each_video > 1:
        step = max(1, math.ceil(
            (num_frames - 1 - sample_duration)/(n_samples_for_each_video - 1)))
    else:
        step = sample_duration
        n_samples_for_each_video = -1
    indices = np.arange(1, num_frames, step).reshape(-1, 1) + \
        np.arange(sample_duration)
    indices = np.where(indices > num_frames, -1, indices)
    return indices.astype(int)


def create_dataset(datasets, subset, split,
                   n_samples_for_each_video=1, sample_duration=1):
    '''create subset of datasets dict length same as video number by adding
       indices nd.array

    Args:
    -----
    datasets: str, ['hmdb51', 'ucf101']
    subset: str, ['train', 'test', 'val']
    split: str, ['s1', 's2', 's3']
    n_samples_for_each_video: int, sample number of a video
    sample_duration: int, frame number of a sample

    Return:
    -------
    annotation_data: dict, video folder, class id, frames indices eg.
    cid_to_class: dict, class id map class name
    '''
    annotation_loader = {'hmdb51': load_annotation,
                         'ucf101': load_annotation,
                         'ucf50': load_annotation}.get(datasets)
    annotation_splits, annotation_class = annotation_loader(datasets)
    cid_to_class = annotation_class.set_index('idx').to_dict('index')
    annotation_data = get_subset_annotation(
        annotation_splits[split], subset
    )[['sample', 'cid', 'dir', 'num']].reset_index().to_dict('index')
    for vid_idx, vid_annotation in annotation_data.items():
        num_frames = vid_annotation['num']
        annotation_data[vid_idx]['indices'] = create_indices(
            num_frames, n_samples_for_each_video, sample_duration)
    return annotation_data, cid_to_class


def make_dataset(datasets, subset, split,
                 n_samples_for_each_video=1, sample_duration=16):
    '''make subset of datasets contains video sample list length same as indice
       numbers which sampling from each video in subset split

    Args:
    -----
    datasets: str, ['hmdb51', 'ucf101']
    subset: str, ['train', 'test', 'val']
    split: str, ['s1', 's2', 's3']
    n_samples_for_each_video: int, sample number of a video
    sample_duration: int, frame number of a sample

    Return:
    -------
    datset: list, video folder, class id, frames indices eg.
    cid_to_class: dict, class id map class name

    '''
    annotation_loader = {'hmdb51': load_annotation,
                         'ucf101': load_annotation,
                         'ucf50': load_annotation}.get(datasets)
    annotation_splits, annotation_class = annotation_loader(datasets)
    cid_to_class = annotation_class.set_index('idx').to_dict()['class']
    annotation_data = get_subset_annotation(
        annotation_splits[split], subset
    )[['sample', 'cid', 'dir', 'num']].reset_index().to_dict('index')
    dataset = []
    IDT_feature_path = os.path.join(
        cfg.IDT_FISHER_ROOT,
        cfg.IDT_FEATURES_DIR[split][datasets],
        cfg.IDT_SUBSET[subset]
    )
    pca_transform = os.path.join(
        cfg.SVM_MODEL_ROOT,
        cfg.SVM_MODEL_SET[split][datasets],
        'RBF_PCA_{}_transformer.pkl'.format(cfg.PCA_DIM[datasets])
    )
    sample = {}
    for vid_idx, vid_annotation in annotation_data.items():
        num_frames = vid_annotation['num']
        annotation_data[vid_idx]['indices'] = create_indices(
            num_frames, n_samples_for_each_video, sample_duration)
        for sample_i in annotation_data[vid_idx]['indices']:
            sample['dir'] = annotation_data[vid_idx]['dir']
            sample['sample'] = annotation_data[vid_idx]['sample']
            sample['cid'] = annotation_data[vid_idx]['cid']
            sample['num'] = annotation_data[vid_idx]['num']
            sample['indices'] = sample_i.reshape(1, -1)
            sample['idt_fisher'] = os.path.join(
                IDT_feature_path,
                annotation_data[vid_idx]['sample'].replace(
                    '.avi', '.fisher.npz')
            )
            sample['pca_transform'] = os.path.join(pca_transform)
            dataset.append(copy.deepcopy(sample))
    return dataset, cid_to_class


def video_sample_generator(annotation_data):
    '''generator video image slices

    Args:
    -----
    annotation_data: dict, subset of datasets

    Return:
    -------
    generator: list, images of indices
    '''
    for video_i in np.arange(len(annotation_data)):
        num_sample = annotation_data[video_i]['indices'].shape[0]
        for sample_i in np.arange(num_sample):
            indices = annotation_data[video_i]['indices'][sample_i].astype(int)
            indices = indices[indices > 0]
            yield hmdb_loader(video_dir_path=annotation_data[video_i]['dir'],
                              frame_indices=indices)


if __name__ == "__main__":
    # This main id for test
    hmdb_annotation, hmdb_class = load_annotation('hmdb51')
    ucf_annotation, ucf_class = load_annotation('ucf101')
    hmdb_loader = get_default_video_loader()
    video_name = hmdb_annotation['s1']['sample'][0]
    video = hmdb_loader(video_dir_path=hmdb_annotation['s1']['dir'][0],
                        frame_indices=np.arange(
                            hmdb_annotation['s1']['num'][0])+1
                        )

    print(hmdb_annotation['s1'].shape)
    print(video_name)
    print(len(video))
