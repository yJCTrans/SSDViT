from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable
from torchvision.models import resnet50, resnet34
import math
import os
import torch.utils.model_zoo as model_zoo


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """
    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(
            x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.p) + ', ' \
            + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """
    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)



class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)    # 按照channel分半，一半用IN，一半用BN，IBN不是这么实现的吧
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)        # 然后把两个输出级联起来
        return out


class Bottleneck_IBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=False, stride=1, downsample=None):
        super(Bottleneck_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn:
            self.bn1 = IBN(planes)          # IBN 加的这么靠前？
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        out += residual
        out = self.relu(out)

        return out

class ResNet_IBN_a_base(nn.Module):

    def __init__(self, layers, block=Bottleneck_IBN):
        scale = 64
        self.inplanes = scale
        super(ResNet_IBN_a_base, self).__init__()
        self.conv1 = nn.Conv2d(3, scale, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(scale)
        # self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, scale, layers[0])
        self.layer2 = self._make_layer(block, scale * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, scale * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, scale * 8, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        ibn = True
        if planes == 512:
            ibn = False
        layers.append(block(self.inplanes, planes, ibn, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, ibn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.relu(x)                  # why remove this relu?
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
        """

    pretrain_dict = {
            k: v
            for k, v in pretrain_dict.items()
            if k in model_dict and model_dict[k].size() == v.size()
        }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

class IBNNet(nn.Module):
    def __init_with_imagenet(self, baseModel):
        # pretrain_dict = resnet50(pretrained=True)
        pretrain_dict = model_zoo.load_url(model_urls['resnet50'])

        model_dict = baseModel.state_dict()
        pretrain_dict = {
            k: v
            for k, v in pretrain_dict.items()
            if k in model_dict and model_dict[k].size() == v.size()
        }
        model_dict.update(pretrain_dict)
        baseModel.load_state_dict(model_dict)

    def getBase(self):
        baseModel = ResNet_IBN_a_base([3, 4, 6, 3])

        self.__init_with_imagenet(baseModel)
        return baseModel

    def __init__(self, num_features=0, dropout=0, cut_at_pooling=False, norm=True, num_classes=[0,0,0], BNNeck=True, GEM=True):
        super(IBNNet, self).__init__()
        self.num_features = num_features
        self.dropout = dropout
        self.cut_at_pooling = cut_at_pooling
        self.num_classes1 = num_classes[0]
        self.num_classes2 = num_classes[1]
        self.num_classes3 = num_classes[2]
        self.has_embedding = num_features > 0
        self.norm = norm
        self.BNNeck = BNNeck
        self.gem_pool = GEM
        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
        # Construct base (pretrained) resnet
        self.base = self.getBase()
        self.base.layer4[0].conv2.stride = (1, 1)           # little trick
        self.base.layer4[0].downsample[0].stride = (1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        if self.gem_pool:
            print("Generalized Mean Pooling")
            self.global_pool = GeneralizedMeanPoolingP()
        else:
            print("Global Adaptive Pooling")
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        out_planes = 2048
        if self.has_embedding:
            self.feat = nn.Linear(out_planes, self.num_features)
            init.kaiming_normal_(self.feat.weight, mode='fan_out')
            init.constant_(self.feat.bias, 0)
        else:
            # Change the num_features to CNN output channels
            self.num_features = out_planes

        self.feat_bn = nn.BatchNorm1d(self.num_features)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        self.reset_IN()

    def reset_IN(self):
        for m in self.modules():
            if isinstance(m, nn.InstanceNorm2d):
                if m.affine:
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x= self.base(x)                     # base is IBN
        # x = self.gap(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x= self.feat_bn(x)
        tri_features = x

        if self.training is False:
            bn_x = F.normalize(bn_x)
            return bn_x

        if isinstance(bn_x, list):
            output = []
            for bnfeature in bn_x:
                if self.norm:
                    bnfeature = F.normalize(bnfeature)
                output.append(bnfeature)
            if self.BNNeck:
                return output, tri_features
            else:
                return output

        if self.norm:
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.BNNeck:
            return bn_x, tri_features
        else:
            return bn_x


