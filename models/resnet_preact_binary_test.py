# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 22:52:40 2019

@author: Chen
"""

import torch.nn as nn
import math
from .binarized_modules import  BinarizeLinear,BinarizeConv2d

__all__ = ['resnet_preact_binary_test']

def Binaryconv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return BinarizeConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                          padding=1, bias=False)


def init_model(model):
    for m in model.modules():
        if isinstance(m, BinarizeConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,do_bnact=True):
        super(BasicBlock, self).__init__()

        self.downsample = downsample
        self.do_bnact = do_bnact
        self.stride = stride
        
        self.conv1 = Binaryconv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.Hardtanh(inplace=True)
        self.conv2 = Binaryconv3x3(planes, planes)
        self.act2 = nn.Hardtanh(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)
        
    def forward(self, x):

        residual = x.clone()
        
        out = self.act1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        
        out = self.act2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.act1(residual)
            residual = self.downsample(residual)
            
        out += residual
        
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, do_bnact=True):
        super(Bottleneck, self).__init__()
        self.conv1 = BinarizeConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = BinarizeConv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.act = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        out = self.act1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        
        out = self.act2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.act3(out)
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.act1(residual)
            residual = self.downsample(residual)

        out += residual

        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1, do_bnact=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                BinarizeConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))
        
        layers.append(block(self.inplanes, planes, do_bnact=do_bnact))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
#        x = self.bn2(x)
        x = self.act(x)
        x = self.fc(x)
        x = self.bn3(x)
        x = self.logsoftmax(x)

        return x


class ResNet_imagenet(ResNet):

    def __init__(self, num_classes=1000,
                 block=Bottleneck, layers=[3, 4, 23, 3]):
        super(ResNet_imagenet, self).__init__()
        self.inplanes = 64
        self.conv1 = BinarizeConv2d(in_planes=3, out_planes=64, kernel_size=7, 
                                    stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = nn.Hardtanh(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block=block, planes=64, blocks=layers[0])
        self.layer2 = self._make_layer(block=block, planes=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block=block, planes=256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block=block, planes=512, blocks=layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = BinarizeLinear(in_features=512 * block.expansion, out_features=num_classes)

        init_model(self)
#        self.regime = {
#            0: {'optimizer': 'SGD', 'lr': 1e-1,
#                'weight_decay': 1e-4, 'momentum': 0.9},
#            30: {'lr': 1e-2},
#            60: {'lr': 1e-3, 'weight_decay': 0},
#            90: {'lr': 1e-4}
#        }
        self.regime = { 0: {'optimizer': 'Adam'} }


class ResNet_cifar10(ResNet):

    def __init__(self, num_classes=10, block=BasicBlock, depth=18, 
                 inflate=1):
        super(ResNet_cifar10, self).__init__()
        self.inflate = inflate
        self.inplanes = 16
        n = int(depth / 6)
        self.conv1 =BinarizeConv2d(in_channels=3, out_channels=16, kernel_size=3, 
                                   stride=1, padding=1, bias=False)
        self.maxpool = lambda x: x
        self.bn1 = nn.BatchNorm2d(16)
        self.act = nn.Hardtanh(inplace=True)
        self.layer1 = self._make_layer(block=block, planes=16*self.inflate, blocks=n)
        self.layer2 = self._make_layer(block=block, planes=32*self.inflate, blocks=n, stride=2)
        self.layer3 = self._make_layer(block=block, planes=64*self.inflate, blocks=n, stride=2,do_bnact=False)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.bn2 = nn.BatchNorm1d(64*self.inflate)
        self.bn3 = nn.BatchNorm1d(num_classes)
        self.logsoftmax = nn.LogSoftmax()
        self.fc = BinarizeLinear(in_features=64*self.inflate, out_features=num_classes)

        init_model(self)
        #self.regime = {
        #    0: {'optimizer': 'SGD', 'lr': 1e-1,
        #        'weight_decay': 1e-4, 'momentum': 0.9},
        #    81: {'lr': 1e-4},
        #    122: {'lr': 1e-5, 'weight_decay': 0},
        #    164: {'lr': 1e-6}
        #}
#        self.regime = {
#            0: {'optimizer': 'Adam', 'lr': 5e-3},
#            101: {'lr': 1e-3},
#            142: {'lr': 5e-4},
#            184: {'lr': 1e-4},
#            220: {'lr': 1e-5}
#        }
        self.regime = { 0: {'optimizer': 'Adam'} }
        

class ResNet_cifar100(ResNet):

    def __init__(self, num_classes=10, block=BasicBlock, depth=18, 
                 inflate=4):
        super(ResNet_cifar100, self).__init__()
        self.inflate = inflate
        self.inplanes = 16
        n = int(depth / 6)
        self.conv1 =BinarizeConv2d(in_channels=3, out_channels=16, kernel_size=3, 
                                   stride=1, padding=1, bias=False)
        self.maxpool = lambda x: x
        self.bn1 = nn.BatchNorm2d(16)
        self.act = nn.Hardtanh(inplace=True)
        self.layer1 = self._make_layer(block=block, planes=16*self.inflate, blocks=n)
        self.layer2 = self._make_layer(block=block, planes=32*self.inflate, blocks=n, stride=2)
        self.layer3 = self._make_layer(block=block, planes=64*self.inflate, blocks=n, stride=2,do_bnact=False)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.bn2 = nn.BatchNorm1d(64*self.inflate)
        self.bn3 = nn.BatchNorm1d(num_classes)
        self.logsoftmax = nn.LogSoftmax()
        self.fc = BinarizeLinear(in_features=64*self.inflate, out_features=num_classes)

        init_model(self)
        #self.regime = {
        #    0: {'optimizer': 'SGD', 'lr': 1e-1,
        #        'weight_decay': 1e-4, 'momentum': 0.9},
        #    81: {'lr': 1e-4},
        #    122: {'lr': 1e-5, 'weight_decay': 0},
        #    164: {'lr': 1e-6}
        #}
#        self.regime = {
#            0: {'optimizer': 'Adam', 'lr': 5e-3},
#            101: {'lr': 1e-3},
#            142: {'lr': 5e-4},
#            184: {'lr': 1e-4},
#            220: {'lr': 1e-5}
#        }
        self.regime = { 0: {'optimizer': 'Adam'} }


def resnet_preact_binary_test(**kwargs):
    num_classes, depth, dataset, inflate = map(
        kwargs.get, ['num_classes', 'depth', 'dataset', 'inflate'])
    if dataset == 'imagenet':
        num_classes = num_classes or 1000
        depth = depth or 50
        if depth == 18:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=BasicBlock, layers=[2, 2, 2, 2])
        if depth == 34:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=BasicBlock, layers=[3, 4, 6, 3])
        if depth == 50:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 4, 6, 3])
        if depth == 101:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 4, 23, 3])
        if depth == 152:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 8, 36, 3])

    elif dataset == 'cifar10':
        num_classes = num_classes or 10
        depth = depth or 18
        return ResNet_cifar10(num_classes=num_classes, block=BasicBlock, 
                              depth=depth, inflate=inflate)
        
    elif dataset == 'cifar100':
        num_classes = num_classes or 100
        depth = depth or 24
        return ResNet_cifar100(num_classes=num_classes, block=BasicBlock, 
                               depth=depth, inflate=inflate)
