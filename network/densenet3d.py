#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===============================================================
# Copyright (C) 2019 HuangYk.
# Licensed under The MIT Lincese.
#
# Filename     : densenet3d.py
# Author       : HuangYK
# Last Modified: 2019-05-30 11:00
# Description  :
# ===============================================================


from __future__ import print_function, division, absolute_import
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


__all__ = ['DenseNet3D', 'densenet3D121', 'densenet3D169', 'densenet3D201', 'densenet3D161']


class ResLinear(nn.Module):
    def __init__(self, inplanes, planes, drop_rate=0):
        super(ResLinear, self).__init__()
        self.drop_rate = drop_rate
        if self.drop_rate > 0:
            self.dropout = nn.Dropout(p=drop_rate)
        self.fc = nn.Linear(inplanes, planes)
        # self.bn = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.fc(x)
        # out = self.bn(out)
        out += residual
        out = self.relu(out)
        if self.drop_rate > 0:
            out = self.dropout(out)

        return out


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0)))


class DenseNet3D(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, res_dr=0, t_dim=1, duration=16,
                 num_classes=51, IS_OPF_NET=False):

        super(DenseNet3D, self).__init__()

        self.t_dim = t_dim
        self.duration = duration
        # First convolution
        if IS_OPF_NET:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv3d(2, num_init_features, kernel_size=(3,7,7),
                                    stride=(1,2,2), padding=(1,3,3), bias=False)),
                ('norm0', nn.BatchNorm3d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv3d(3, num_init_features, kernel_size=(3,7,7),
                                    stride=(1,2,2), padding=(1,3,3), bias=False)),
                ('norm0', nn.BatchNorm3d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
            ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        # residual 
        self.resfc = ResLinear(
            num_features*self.t_dim,
            num_features*self.t_dim, 
            res_dr
        )

        # Linear layer
        self.classifier = nn.Linear(num_features*self.t_dim, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = [x[:, :, d*self.duration:(d+1)*self.duration, :, :, ] for d in range(self.t_dim)]
        out = map(self.extract_feature, out)
        out = torch.cat(tuple(out), 1)
        out = self.resfc(out)
        out = self.classifier(out)
        return out

    def extract_feature(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1)).view(features.size(0), -1)

        return out


def _get_pretrained_model(depth):
    imagenet_pretrained = {
        "densenet121": "./pretrained/resnet_pretrained_imagenet/densenet121.pth",
        "densenet161": "./pretrained/resnet_pretrained_imagenet/densenet161.pth",
        "densenet169": "./pretrained/resnet_pretrained_imagenet/densenet169.pth",
        "densenet201": "./pretrained/resnet_pretrained_imagenet/densenet201.pth",  
    }.get(depth)

    return imagenet_pretrained


def _load_imagenet_pretrained(model, pretrained_dict, use_opf=False):
    #assert len(model.state_dict().keys()[:-4]) == len(pretrained_dict.keys()[:-2])
    #use_opf=False
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    for key in list(pretrained_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            pretrained_dict[new_key] = pretrained_dict[key]
            del pretrained_dict[key]
    del pretrained_dict["classifier.weight"]
    del pretrained_dict["classifier.bias"]
    
    for idx, param_name in enumerate(pretrained_dict.keys()):
        if len(pretrained_dict[param_name].shape) > 1:
            # conv layers
            for dim in range(model.state_dict()[param_name].shape[2]):
                if use_opf and param_name=='features.conv0.weight':
                    for channel in range(model.state_dict()[param_name].shape[1]):
                        model.state_dict()[param_name][:, :, dim, :, :, ][:, channel, :, :,].copy_(
                            torch.mean(pretrained_dict[param_name], 1)
                        )
                else:
                    model.state_dict()[param_name][:,:,dim,:,:,].copy_(
                        pretrained_dict[param_name]
                    )
        else:
            # bn layers
            model.state_dict()[param_name].copy_(
                pretrained_dict[param_name]
            )



def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = DenseNet3D(growth_rate, block_config, num_init_features, **kwargs)
    use_opf = kwargs['IS_OPF_NET']
    if pretrained:
        _load_imagenet_pretrained(
            model, torch.load(_get_pretrained_model(arch)), use_opf
            )
    return model


def densenet3D121(pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)


def densenet3D161(pretrained=False, progress=True, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _densenet('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress,
                     **kwargs)


def densenet3D169(pretrained=False, progress=True, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _densenet('densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress,
                     **kwargs)


def densenet3D201(pretrained=False, progress=True, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress,
                     **kwargs)
