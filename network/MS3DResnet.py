#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===============================================================
# Copyright (C) 2018 HuangYk.
# Licensed under The MIT Lincese.
#
# Filename     : MS3DResnet.py
# Author       : HuangYK
# Last Modified: 2019-03-21 15:45
# Description  :
# ===============================================================


from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
from torch.nn import init
from config.config import cfg
# from dropblock import DropBlock3D


def conv_t(in_planes, out_planes, stride=1):
    """
    """
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 1),
                     stride=(stride, 1, 1),
                     padding=(1, 0, 0),
                     bias=False)


def conv_dx3x3(in_planes, out_planes, d=(3, 1, 1), stride=1, 
               groups=1, dilation=1):
    """
    """
    return nn.Conv3d(in_planes, out_planes, kernel_size=(d[0], 3, 3),
                     stride=(d[1], stride, stride),
                     padding=(d[2], dilation, dilation), #padding=(d[2], 1, 1),
                     groups=groups, bias=False, dilation=dilation
                     )

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
    
    
class FcLinear(nn.Module):
    def __init__(self, inplanes, planes, drop_rate=0):
        super(FcLinear, self).__init__()
        self.drop_rate = drop_rate
        if self.drop_rate > 0:
            self.dropout = nn.Dropout(p=drop_rate)
        self.fc = nn.Linear(inplanes, planes)
        # self.bn = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.fc(x)
        #out += residual
        out = self.relu(out)
        if self.drop_rate > 0:
            out = self.dropout(out)

        return out


class T_BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride):
        super(T_BasicBlock, self).__init__()
        self.t_conv = conv_t(inplanes, planes, stride)
        self.t_bn1 = nn.BatchNorm3d(planes)
        self.t_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.t_conv(x)
        out = self.t_bn1(out)
        out += residual
        out = self.t_relu(out)

        return out


class BasicBlock_3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, d=(3, 1, 1), stride=1,
                 downsample=None, norm_layer=None):
        super(BasicBlock_3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
#        if groups != 1 or base_width != 64:
#            raise ValueError(
#                    'BasicBlock only supports groups=1 and base_width=64'
#                    )
#        if dilation > 1:
#            raise NotImplementedError(
#                    "Dilation > 1 not supported in BasicBlock"
#                    )
        self.conv1 = conv_dx3x3(in_planes, planes, d, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_dx3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.stride = (d[1], stride, stride)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if cfg.DEBUG:
            print(out.shape, residual.shape, x.shape)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck_3D(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, d=(3, 1, 1), stride=1,
                 downsample=None, groups=1, base_width=64, dilation=1, 
                 norm_layer=None):
        super(Bottleneck_3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv3d(in_planes, width, kernel_size=(1, 1, 1),
                               bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv3d(width, width, kernel_size=(d[0], 3, 3),
                               stride=(d[1], stride, stride),
                               padding=(d[2], dilation, dilation),
                               groups=groups, dilation=dilation,
                               bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv3d(width, planes * self.expansion, 
                               kernel_size=(1, 1, 1), bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = (d[1], stride, stride)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if cfg.DEBUG:
            print(out.shape, residual.shape, x.shape)

        out += residual
        out = self.relu(out)

        return out


class MS3DResnet(nn.Module):

    def __init__(self, block, layers, num_classes, block_d=(3, 1, 1), 
                 conv1_d=(3, 1, 1), maxpool_d=(3, 2, 1), t_dim=1, duration=16,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, res_dr=0.2, IS_OPF_NET=False, **kws):
        super(MS3DResnet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer
        
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                     replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group        
        self.t_dim = t_dim
        self.duration = duration
        self.block_d = block_d
        
        if IS_OPF_NET:
            self.conv1 = nn.Conv3d(2, 64, kernel_size=(conv1_d[0], 7, 7),
                                   stride=(conv1_d[1], 2, 2),
                                   padding=(conv1_d[2], 3, 3), bias=False)
        else:
            self.conv1 = nn.Conv3d(3, 64, kernel_size=(conv1_d[0], 7, 7),
                                   stride=(conv1_d[1], 2, 2),
                                   padding=(conv1_d[2], 3, 3), bias=False)

        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=(maxpool_d[0], 3, 3),
                                    stride=(maxpool_d[1], 2, 2),
                                    padding=(maxpool_d[2], 1, 1))

        self.layer_idx = 1
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, 
                                       dilate=replace_stride_with_dilation[2])

        # self.layer5 = self._make_layer(block, 512, 3, stride=2)

        # TODO(huangyukun0817@gmail.com):3d pooling modified D dimension kernel
        # self.avgpool = nn.AvgPool3d((1, avgpool, avgpool))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # modified
#        self.dropout = nn.Dropout(p=0.2)
#        self.fc1 = nn.Linear(512*block.expansion, 512*block.expansion)
        self.fc_type = cfg.FC_TYPE
        if self.fc_type == "resfc":
            self.resfc = ResLinear(
                512*block.expansion*self.t_dim,
                512*block.expansion*self.t_dim, 
                res_dr
            	)
        elif self.fc_type == "fc":
            self.fc_linear = FcLinear(
                512*block.expansion*self.t_dim,
                512*block.expansion*self.t_dim, 
                res_dr
            	)

        self.fc = nn.Linear(512*block.expansion*self.t_dim, num_classes)
        #self.fc_blocks = nn.Linear(512*block.expansion, num_classes)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1:
            block_d = self.block_d
        else:
            block_d = list(self.block_d)
            block_d[1] = 1  # 3d conv d-dimension stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1,
                          stride=(block_d[1], stride, stride),
                          bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, block_d, stride,
                            downsample, self.groups, self.base_width, 
                            previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        self.layer_idx += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        out = [x[:, :, d*self.duration:(d+1)*self.duration, :, :, ] for d in range(self.t_dim)]

        out = map(self.feature_extract, out)
#        residual = torch.mean(
#            torch.cat(out, 0),
#            0,
#            True
#        )
        #out_blocks = map(self.fc_blocks, out)
        out = torch.cat(tuple(out), 1)
        if self.fc_type == "resfc":
            out = self.resfc(out)
            out = self.fc(out)
        elif self.fc_type == "fc":
            out = self.fc_linear(out)
            out = self.fc(out)
        elif self.fc_type == "direct":
            #out = self.fc_linear(out)
            out = self.fc(out)
        # modified


        #out = 0.5*out + 0.5*(sum(out_blocks)/self.t_dim)

        return out

    def feature_extract(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the
    net.
    """
    b = [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3,
         model.layer4]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_5x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the
    net.
    """
    b = [model.conv1]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc, model.resfc]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


def get_pretrained_model(depth):
    imagenet_pretrained = {
        "18": "./pretrained/resnet_pretrained_imagenet/resnet18.pth",
        "34": "./pretrained/resnet_pretrained_imagenet/resnet34.pth",
        "50": "./pretrained/resnet_pretrained_imagenet/resnet50.pth",
        "101": "./pretrained/resnet_pretrained_imagenet/resnet101.pth",
        "152": "./pretrained/resnet_pretrained_imagenet/resnet152.pth",
        "50_32x4": "./pretrained/resnet_pretrained_imagenet/resnext50_32x4d.pth",
        "101_32x4": "./pretrained/resnet_pretrained_imagenet/resnext_101_32x4d.pth",
        "101_32x8": "./pretrained/resnet_pretrained_imagenet/resnext101_32x8d.pth",
        "101_64x4": "./pretrained/resnet_pretrained_imagenet/resnext_101_64x4d.pth",      
    }.get(depth)

    return imagenet_pretrained


def load_imagenet_pretrained(model, pretrained_dict, use_opf=False):
    # use list() to be compatible with python3
    #use_opf=False
    for param_name in list(pretrained_dict.keys())[:-2]:
        if len(pretrained_dict[param_name].shape) > 1:
            # conv layers
            for dim in range(model.state_dict()[param_name].shape[2]):
                if use_opf and param_name=='conv1.weight':
                    for channel in range(model.state_dict()[param_name].shape[1]):
                        model.state_dict()[param_name][:, :, dim, :, :, ][:, channel, :, :,].copy_(
                            torch.mean(pretrained_dict[param_name], 1)
                        )
                else:
                    model.state_dict()[param_name][:, :, dim, :, :, ].copy_(
                            pretrained_dict[param_name]
                        )
        else:
            # bn layers
            model.state_dict()[param_name].copy_(pretrained_dict[param_name])


def _load_imagenet_pretrained(model, pretrained_dict):
    assert len(model.state_dict().keys()[:-4]) == len(
        pretrained_dict.keys()[:-2])
    for idx, param_name in enumerate(pretrained_dict.keys()[:-2]):
        model_param_name = model.state_dict().keys()[idx]
        assert param_name.split('.')[-1] == model_param_name.split('.')[-1]
        if len(pretrained_dict[param_name].shape) > 1:
            # conv layers
            for dim in range(model.state_dict()[model_param_name].shape[2]):
                model.state_dict()[model_param_name][:, :, dim, :, :, ].copy_(
                    pretrained_dict[param_name]
                )
        else:
            # bn layers
            model.state_dict()[model_param_name].copy_(
                pretrained_dict[param_name]
            )


# input: [batch, 3, 16, 224, 224] config
_conv1_d = [3, 1, 1]
_maxpool_d = [3, 2, 1] #[3,2,1], [3,1,1], [2,2,0]
_block_d = [3, 1, 1]  # kernel, stride, padding in block layer


def MS3DResnet18(num_classes, t_dim=1, duration=16, pretrained=True, **kws):
    model = MS3DResnet(
        BasicBlock_3D, [2, 2, 2, 2], num_classes,
        block_d=_block_d, conv1_d=_conv1_d, maxpool_d=_maxpool_d,
        t_dim=t_dim, duration=duration, **kws
    )
    use_opf = kws['IS_OPF_NET']
    if pretrained:
        load_imagenet_pretrained(
            model, torch.load(get_pretrained_model('18')), use_opf
        )
        print("**Pretrained in {:s} loaded".format("ImageNet"))

    return model


def MS3DResnet34(num_classes, t_dim=1, duration=16, pretrained=True, **kws):
    model = MS3DResnet(
        BasicBlock_3D, [3, 4, 6, 3], num_classes,
        block_d=_block_d, conv1_d=_conv1_d, maxpool_d=_maxpool_d,
        t_dim=t_dim, duration=duration, **kws
    )
    use_opf = kws['IS_OPF_NET']
    if pretrained:
        load_imagenet_pretrained(
            model, torch.load(get_pretrained_model('34')), use_opf
        )
        print("**Pretrained in {:s} loaded".format("ImageNet"))

    return model


def MS3DResnet50(num_classes, t_dim=1, duration=16, pretrained=True, **kws):
    model = MS3DResnet(
        Bottleneck_3D, [3, 4, 6, 3], num_classes,
        block_d=_block_d, conv1_d=_conv1_d, maxpool_d=_maxpool_d,
        t_dim=t_dim, duration=duration, **kws
    )
    use_opf = kws['IS_OPF_NET']
    if pretrained:
        load_imagenet_pretrained(
            model, torch.load(get_pretrained_model('50')), use_opf
        )
        print("**Pretrained in {:s} loaded".format("ImageNet"))

    return model


def MS3DResnet101(num_classes, t_dim=1, duration=16, pretrained=True, **kws):
    model = MS3DResnet(
        Bottleneck_3D, [3, 4, 23, 3], num_classes,
        block_d=_block_d, conv1_d=_conv1_d, maxpool_d=_maxpool_d,
        t_dim=t_dim, duration=duration, **kws
    )
    use_opf = kws['IS_OPF_NET']
    if pretrained:
        load_imagenet_pretrained(
            model, torch.load(get_pretrained_model('101')), use_opf
        )
        print("**Pretrained in {:s} loaded".format("ImageNet"))

    return model


def MS3DResnet152(num_classes, t_dim=1, duration=16, pretrained=True, **kws):
    model = MS3DResnet(
        Bottleneck_3D, [3, 8, 36, 3], num_classes,
        block_d=_block_d, conv1_d=_conv1_d, maxpool_d=_maxpool_d,
        t_dim=t_dim, duration=duration, **kws
    )
    use_opf = kws['IS_OPF_NET']
    if pretrained:
        load_imagenet_pretrained(
            model, torch.load(get_pretrained_model('152')), use_opf
        )
        print("**Pretrained in {:s} loaded".format("ImageNet"))

    return model


def MS3DResnext3D50_32x4d(num_classes, t_dim=1, duration=16, pretrained=True, **kws):
    kws['groups'] = 32
    kws['width_per_group'] = 4
    model = MS3DResnet(
        Bottleneck_3D, [3, 4, 6, 3], num_classes,
        block_d=_block_d, conv1_d=_conv1_d, maxpool_d=_maxpool_d,
        t_dim=t_dim, duration=duration, **kws
    )
    if pretrained:
        load_imagenet_pretrained(
            model, torch.load(get_pretrained_model('50_32x4'))
        )
        print("**Pretrained in {:s} loaded".format("ImageNet"))

    return model


def MS3DResnext3D101_32x4d(num_classes, t_dim=1, duration=16, pretrained=True, **kws):
    kws['groups'] = 32
    kws['width_per_group'] = 4
    model = MS3DResnet(
        Bottleneck_3D, [3, 4, 23, 3], num_classes,
        block_d=_block_d, conv1_d=_conv1_d, maxpool_d=_maxpool_d,
        t_dim=t_dim, duration=duration, **kws
    )
    if pretrained:
        _load_imagenet_pretrained(
            model, torch.load(get_pretrained_model('101_32x4'))
        )
        print("**Pretrained in {:s} loaded".format("ImageNet"))

    return model


def MS3DResnext3D101_32x8d(num_classes, t_dim=1, duration=16, pretrained=True, **kws):
    kws['groups'] = 32
    kws['width_per_group'] = 8
    model = MS3DResnet(
        Bottleneck_3D, [3, 4, 23, 3], num_classes,
        block_d=_block_d, conv1_d=_conv1_d, maxpool_d=_maxpool_d,
        t_dim=t_dim, duration=duration, **kws
    )
    if pretrained:
        load_imagenet_pretrained(
            model, torch.load(get_pretrained_model('101_32x8'))
        )
        print("**Pretrained in {:s} loaded".format("ImageNet"))

    return model


def MS3DResnext3D101_64x4d(num_classes, t_dim=1, duration=16, pretrained=True, **kws):
    kws['groups'] = 64
    kws['width_per_group'] = 4
    model = MS3DResnet(
        Bottleneck_3D, [3, 4, 23, 3], num_classes,
        block_d=_block_d, conv1_d=_conv1_d, maxpool_d=_maxpool_d,
        t_dim=t_dim, duration=duration, **kws
    )
    if pretrained:
        _load_imagenet_pretrained(
            model, torch.load(get_pretrained_model('101_64x4'))
        )
        print("**Pretrained in {:s} loaded".format("ImageNet"))

    return model