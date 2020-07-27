#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===============================================================
# Copyright (C) 2018 HuangYk.
# Licensed under The MIT Lincese.
#
# Filename     : config.py
# Author       : HuangYK
# Last Modified: 2018-09-18 22:46
# Description  :
# ===============================================================


import os
from easydict import EasyDict as edict

__C = edict()

cfg = __C

# if IS_DEMO set true, then you should use jupyter-notebook run demo
# if IS_DEMO set False, then you can run normal train and test
__C.IS_DEMO = False
#
annotation_root = './annotation' if not cfg.IS_DEMO else './sample_annotation'
idt_fisher_feature_dir = './IDT_Fisher'
svm_model_dir = './SVM_Model'

assert os.path.exists(annotation_root), 'Annotation Folder Not Find'
assert os.path.exists(idt_fisher_feature_dir), 'IDT Fisher Folder Not Find'
assert os.path.exists(svm_model_dir), 'SVM Model Folder Not Find'

__C.ANNOTATION_ROOT = annotation_root

__C.IDT_FISHER_ROOT = idt_fisher_feature_dir

__C.SVM_MODEL_ROOT = svm_model_dir

__C.PCA_DIM = {'hmdb51':10000, 'ucf101':10000, 'ucf50':10000}

__C.DEBUG = False

__C.SAME_TRANSFORM = False

__C.IDT_FEATURES_S1={
    'hmdb51':'HMDB51_S1_Fisher_features', 
    'ucf101':'UCF101_S1_Fisher_features',
    'ucf50':'UCF50_S1_Fisher_features',
}

__C.IDT_FEATURES_S2={
    'hmdb51':'HMDB51_S2_Fisher_features', 
    'ucf101':'UCF101_S2_Fisher_features',
    'ucf50':'UCF50_S2_Fisher_features',
}

__C.IDT_FEATURES_S3={
    'hmdb51':'HMDB51_S3_Fisher_features', 
    'ucf101':'UCF101_S3_Fisher_features',
    'ucf50':'UCF50_S3_Fisher_features',
}

__C.IDT_FEATURES_DIR={
    's1':__C.IDT_FEATURES_S1, 
    's2':__C.IDT_FEATURES_S2,
    's3':__C.IDT_FEATURES_S3
}

__C.WEIGHT={
    'rgb':0.5,
    'idt':0.5,
    'opf':0.4
}

fc_type=('direct', 'fc', 'resfc')

__C.FC_TYPE=fc_type[2]

__C.IDT_SUBSET={'train': 'train_features', 'test': 'test_features'}

__C.SVM_MODEL_S1= {
        'hmdb51': 'model_hmdb51', 
        'ucf101': 'model_ucf101', 
        'ucf50': 'model_ucf50'}

__C.SVM_MODEL_S2 = {
        'hmdb51': 's2_model_hmdb51', 
        'ucf101': 's2_model_ucf101',
        "ucf50": 's2_model_ucf50'}

__C.SVM_MODEL_S3 = {
        'hmdb51': 's2_model_hmdb51', 
        'ucf101': 's2_model_ucf101',
        "ucf50": 's2_model_ucf50'}

__C.SVM_MODEL_SET = {'s1': __C.SVM_MODEL_S1, 's2': __C.SVM_MODEL_S2, 's3':__C.SVM_MODEL_S3}

# ['TVL1', 'WarpTVL1', 'Liteflow']
__C.OPTIC_FLOW_TYPE = 'WarpTVL1'
