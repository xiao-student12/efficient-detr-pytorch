# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

# 在 backbone.py 顶部引入需要的模块
import torch.nn as nn
from torchvision.models.efficientnet import efficientnet_b2
from torchvision.ops.misc import SqueezeExcitation # 视 torch 版本而定，可能在 torchvision.models.efficientnet 中


# ----------------- 新增 SimAM 注意力模块 -----------------
class SimAM(torch.nn.Module):
    """
    SimAM: A Simple, Parameter-Free Attention Module
    """
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)

def replace_se_with_simam(module):
    """递归地将网络中的 SqueezeExcitation 模块替换为 SimAM"""
    for name, child in module.named_children():
        if isinstance(child, SqueezeExcitation):
            setattr(module, name, SimAM())
        else:
            replace_se_with_simam(child)
# ---------------------------------------------------------
class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


# 修改原有的 Backbone 类
class Backbone(BackboneBase):
    """EfficientNet-B2 backbone with SimAM."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):

        # 1. 实例化 EfficientNet-B2
        # 注意: DETR 通常冻结 BatchNorm，这里为了简便直接调用。
        # 如果需要冻结 BN，可以遍历替换为 FrozenBatchNorm2d。
        backbone = efficientnet_b2(pretrained=is_main_process())

        # 2. 将 SE 模块替换为 SimAM
        replace_se_with_simam(backbone)

        # 3. 配置输出层提取 (EfficientNet-B2 的 features 最后层为索引 8)
        # 如果 return_interm_layers 为 True，可以提取多尺度特征
        if return_interm_layers:
            return_layers = {"3": "0", "5": "1", "7": "2", "8": "3"}
        else:
            return_layers = {'8': "0"}

        # EfficientNet 的主体特征提取部分被存放在 backbone.features 中
        self.body = IntermediateLayerGetter(backbone.features, return_layers=return_layers)

        # EfficientNet-B2 最后一层的输出通道数为 1408
        self.num_channels = 1408

        # 继承和冻结不需要训练的层
        super(BackboneBase, self).__init__()
        for name, parameter in self.body.named_parameters():
            if not train_backbone:
                parameter.requires_grad_(False)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
