#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===============================================================
# Copyright (C) 2019 HuangYk.
# Licensed under The MIT Lincese.
#
# Filename     : optic_flow_loader.py
# Author       : HuangYK
# Last Modified: 2019-07-16 21:41
# Description  :
# ===============================================================


import flow_vis
import numpy as np
from PIL import Image


def construct_opticflow_image(opt_x, opt_y, method='TV-L1'):
    if method == 'TV-L1':
        opf_image = Image.fromarray(
            flow_vis.flow_to_color(np.concatenate(
                [np.array(opt_x)[:, :, np.newaxis],
                 np.array(opt_y)[:, :, np.newaxis]],
                axis=2),  convert_to_bgr=False).astype('uint8')).convert('RGB')
        return opf_image
