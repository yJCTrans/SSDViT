import torch
import torch.nn as nn
import copy
from .vision_transformer import TransReID
from .swin_transformer import SwinTransformer
from .coatnet import CoAtNet
from .ConViT import VisionTransformer
from .botnet import botnet50
from torch.nn import functional as F

from functools import partial
from timm.models.vision_transformer import _cfg
import pdb

def shuffle_unit(features, shift, group, begin=1):
    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class build_transformer(nn.Module):
    def __init__(self, num_features=0, norm=True, BNNeck=True, SD=False):
        super(build_transformer, self).__init__()
        model_path = './vit_base_ics_cfs_lup.pth'
        self.num_features = num_features
        self.in_planes = 768

        self.base = TransReID(
            img_size=(256, 128), stride_size=16, drop_path_rate=0.1, qkv_bias=True,
            drop_rate=0.0, attn_drop_rate=0.0, conv_stem=True)

        self.norm = norm
        self.BNNeck = BNNeck
        self.SD = SD

        self.base.load_param(model_path, hw_ratio=2)
        print('Loading pretrained ImageNet model from {}'.format(model_path))

        self.feat_bn = nn.BatchNorm1d(self.in_planes)
        self.feat_bn.bias.requires_grad_(False)
        self.feat_bn.apply(weights_init_kaiming)

    def forward(self, x, style=False):
        x, x_rb = self.base(x, style)  # base is ViT_base

        bn_x = self.feat_bn(x)
        bn_x_rb = self.feat_bn(x_rb)

        tri_features = x
        rb_features = x_rb

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
                return output, tri_features, rb_features
            else:
                return output

        if self.norm:
            bn_x = F.normalize(bn_x)
            bn_x_rb = F.normalize(bn_x_rb)

        if self.SD:
            if self.BNNeck:
                return bn_x, bn_x_rb, tri_features, rb_features
            else:
                return bn_x, bn_x_rb
        else:
            if self.BNNeck:
                return bn_x, tri_features
            else:
                return bn_x,
