# -*- coding:utf-8 -*-
"""
@author: ryan
@software: PyCharm
@project name: CMPNet
@file: net.py
@time: 2021/09/08 21:15
@desc:
"""

import math
import paddle
import numpy as np
import paddle.nn as nn
from paddle.nn import Linear

__all__ = ['senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d']


model_urls = {
    'senet154': './Pre_training_parameters/senet154.pdparams',
    'se_resnet50': './Pre_training_parameters/se_resnet50.pdparams',
    'se_resnet101': './Pre_training_parameters/se_resnet101.pdparams',
    'se_resnet152': './Pre_training_parameters/se_resnet152.pdparams',
    'se_resnext50_32x4d': './Pre_training_parameters/se_resnext50_32x4d.pdparams',
    'se_resnext101_32x4d': './Pre_training_parameters/se_resnext101_32x4d.pdparams'
}


class SEModule(nn.Layer):

    def __init__(self, channels, reduction):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Conv2D(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2D(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Layer):

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

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, in_channels, channels, groups, reduction, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2D(in_channels, channels * 2, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(channels * 2)
        self.conv2 = nn.Conv2D(channels * 2, channels * 4, kernel_size=3, stride=stride, padding=1, groups=groups, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(channels * 4)
        self.conv3 = nn.Conv2D(channels * 4, channels * 4, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(channels * 4)
        self.relu = nn.ReLU()
        self.se_module = SEModule(channels * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, in_channels, channels, groups, reduction, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2D(in_channels, channels, kernel_size=1, stride=stride, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(channels)
        self.conv2 = nn.Conv2D(channels, channels, kernel_size=3, padding=1, groups=groups, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(channels)
        self.conv3 = nn.Conv2D(channels, channels * 4, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(channels * 4)
        self.relu = nn.ReLU()
        self.se_module = SEModule(channels * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, in_channels, channels, groups, reduction, stride=1, downsample=None, base_width=4):
        super().__init__()
        width = math.floor(channels * (base_width / 64)) * groups
        self.conv1 = nn.Conv2D(in_channels, width, kernel_size=1, stride=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(width)
        self.conv2 = nn.Conv2D(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(width)
        self.conv3 = nn.Conv2D(width, channels * 4, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(channels * 4)
        self.relu = nn.ReLU()
        self.se_module = SEModule(channels * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Layer):

    def __init__(self, block, layers, input_dim, groups, reduction, dropout_p=0.2,
                 in_channels=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the network (layer1...layer4).
        input_dim (int): Number of channels for the input image.
        groups (int): Number of groups for the 3x3 convolution in each bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super().__init__()
        self.in_channels = in_channels
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2D(input_dim, 64, 3, stride=2, padding=1, bias_attr=False)),
                ('bn1', nn.BatchNorm2D(64)),
                ('relu1', nn.ReLU()),
                ('conv2', nn.Conv2D(64, 64, 3, stride=1, padding=1, bias_attr=False)),
                ('bn2', nn.BatchNorm2D(64)),
                ('relu2', nn.ReLU()),
                ('conv3', nn.Conv2D(64, in_channels, 3, stride=1, padding=1, bias_attr=False)),
                ('bn3', nn.BatchNorm2D(in_channels)),
                ('relu3', nn.ReLU()),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2D(input_dim, in_channels, kernel_size=7, stride=2, padding=3, bias_attr=False)),
                ('bn1', nn.BatchNorm2D(in_channels)),
                ('relu1', nn.ReLU()),
            ]

        layer0_modules.append(('pool', nn.MaxPool2D(3, stride=2, ceil_mode=True)))
        self.layer0 = nn.Sequential(*layer0_modules)
        self.layer1 = self._make_layer(
            block,
            channels=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            channels=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            channels=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            channels=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2D(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, channels, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.in_channels, channels * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias_attr=False),
                nn.BatchNorm2D(channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, channels, groups, reduction, stride, downsample))
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = paddle.flatten(x, 1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def initialize_pretrained_model(model, num_classes, weights_url, input_dim=3):
    state_dict = paddle.load(weights_url)

    load_fc = num_classes == 1000

    if load_fc:
        if input_dim == 1:
            conv1_weight = state_dict['layer0.conv1.weight']
            state_dict['layer0.conv1.weight'] = conv1_weight.sum(axis=1, keepdims=True)
        model.set_state_dict(state_dict)
    else:
        state_dict.pop('last_linear.weight')
        state_dict.pop('last_linear.bias')
        if input_dim == 1:
            conv1_weight = state_dict['layer0.conv1.weight']
            state_dict['layer0.conv1.weight'] = conv1_weight.sum(axis=1, keepdims=True)
        model.set_state_dict(state_dict)


def senet154(pretrained=False, num_classes=1000, input_dim=3):
    model = SENet(SEBottleneck, [3, 8, 36, 3], input_dim=input_dim, groups=64, reduction=16,
                  dropout_p=0.2, num_classes=num_classes)
    if pretrained:
        initialize_pretrained_model(model, num_classes, model_urls['senet154'], input_dim)
    return model


def se_resnet50(pretrained=False, num_classes=1000, input_dim=3):
    model = SENet(SEResNetBottleneck, [3, 4, 6, 3], input_dim=input_dim, groups=1, reduction=16,
                  dropout_p=None, in_channels=64, input_3x3=False, downsample_kernel_size=1, 
                  downsample_padding=0, num_classes=num_classes)
    if pretrained:
        initialize_pretrained_model(model, num_classes, model_urls['se_resnet50'], input_dim)
    return model


def se_resnet101(pretrained=False, num_classes=1000, input_dim=3):
    model = SENet(SEResNetBottleneck, [3, 4, 23, 3], input_dim=input_dim, groups=1, reduction=16,
                  dropout_p=None, in_channels=64, input_3x3=False, downsample_kernel_size=1,
                  downsample_padding=0, num_classes=num_classes)
    if pretrained:
        initialize_pretrained_model(model, num_classes, model_urls['se_resnet101'], input_dim)
    return model


def se_resnet152(pretrained=False, num_classes=1000, input_dim=3):
    model = SENet(SEResNetBottleneck, [3, 8, 36, 3], input_dim=input_dim, groups=1,
                  reduction=16, dropout_p=None, in_channels=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0, num_classes=num_classes)
    if pretrained:
        initialize_pretrained_model(model, num_classes, model_urls['se_resnet152'], input_dim)
    return model


def se_resnext50_32x4d(pretrained=False, num_classes=1000, input_dim=3):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], input_dim=input_dim, groups=32, reduction=16,
                  dropout_p=None, in_channels=64, input_3x3=False, downsample_kernel_size=1, 
                  downsample_padding=0, num_classes=num_classes)
    if pretrained:
        initialize_pretrained_model(model, num_classes, model_urls['se_resnext50_32x4d'], input_dim)
    return model


def se_resnext101_32x4d(pretrained=False, num_classes=1000, input_dim=3):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], input_dim=input_dim, groups=32, reduction=16,
                  dropout_p=None, in_channels=64, input_3x3=False, downsample_kernel_size=1,
                  downsample_padding=0, num_classes=num_classes)
    if pretrained:
        initialize_pretrained_model(model, num_classes, model_urls['se_resnext101_32x4d'], input_dim)
    return model


class MixedNN(paddle.nn.Layer):
    def __init__(self):
        super(MixedNN, self).__init__()

        # Connect the output layer
        self.CNN_1 = se_resnext50_32x4d(pretrained=True, num_classes=30, input_dim=3)
        self.CNN_2 = paddle.vision.resnet50(pretrained=True, num_classes=30)
        self.CNN_3 = se_resnet50(pretrained=True, num_classes=30, input_dim=3)
        self.fc5 = Linear(in_features=90, out_features=30)

    def forward(self, inputs_seed, inputs_flower, inputs_wheat_seed):
        outputs1 = self.CNN_1(inputs_seed)
        outputs2 = self.CNN_2(inputs_flower)
        outputs3 = self.CNN_3(inputs_wheat_seed)
        outputs_concat = paddle.concat(x=[paddle.concat(x=[outputs1, outputs2], axis=1), outputs3], axis=1)
        outputs_final = self.fc5(outputs_concat)
        return outputs_final