# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Experimental modules
"""
import math

import numpy as np
import torch
import torch.nn as nn

from utils.downloads import attempt_download


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # ch_in, ch_out, kernel, stride, ch_strategy
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([
            nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models（模型集成）ModuleList是一个存储模块的列表，它可以自动注册模块的参数
    def __init__(self):
        super().__init__()

    # 前向传播，接受一个输入张量x，以及几个可选的参数，然后对每个模型进行前向传播。每个模型的输出被收集到一个列表y中
    def forward(self, x, augment=False, profile=False, visualize=False):
        # 遍历Ensemble类中的每个模型，将输入张量x以及几个可选的参数传递给模型，然后获取模型的输出
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        # 将所有模型的输出沿着维度1拼接起来
        y = torch.cat(y, 1)  # nms ensemble
        # 包含所有模型的输出和一个None值
        return y, None  # inference, train output


# 加载一个模型或一组模型。这可以是一个单一的模型，或者是一个模型的集合（即模型集成）
def attempt_load(weights, device=None, inplace=True, fuse=True):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    from models.yolo import Detect, Model

    # 创建一个空的Ensemble对象，用于存储和管理一组模型
    model = Ensemble()
    # 如果weights是个列表则直接遍历，else则将weights放到空列表中进行遍历
    for w in weights if isinstance(weights, list) else [weights]:
        # 下载权重文件（如果不在本地），然后加载权重到模型中
        ckpt = torch.load(attempt_download(w), map_location='cpu')  # load
        # 加载模型权重，将模型移动到指定的设备上，然后将模型的数据类型转换为单精度浮点数
        ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

        # Model compatibility updates
        # 如果没有stride，就为模型添加一个stride属性，并设置其值为一个包含单个元素32的张量
        if not hasattr(ckpt, 'stride'):
            ckpt.stride = torch.tensor([32.])
        # 如果模型有names属性，且这个属性的值是列表或元组
        if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
            # 将names属性的值转换为一个字典，字典的键是索引，值是原来的元素
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict
        # 如果模型有fuse()方法就调用ckpt.fuse().eval()，否则调用ckpt.eval()，然后将操作添加到Ensemble对象中
        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval())  # model in eval mode

    # Module compatibility updates
    # 对模型中的各个模块进行兼容性更新，以确保模型可以在不同版本的PyTorch上正常运行
    for m in model.modules():  # 遍历模型中的所有模块
        t = type(m)  # 获取当前模块的类型
        # 如果模块的类型在给定的类型列表中
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            # 设置模块的inplace属性，为了保证在PyTorch1.7.0版本中的兼容性
            m.inplace = inplace  # torch 1.7.0 compatibility
            # 如果模块的类型是Detect，且其anchor_grid属性不是列表类型
            if t is Detect and not isinstance(m.anchor_grid, list):
                # 删除模块的anchor_grid属性
                delattr(m, 'anchor_grid')
                # 设置模块的anchor_grid属性为一个全零张量的列表
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        # 如果模块的类型是nn.Upsample，且模块没有recompute_scale_factor属性
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            # 设置模块的recompute_scale_factor属性为None，为了保证在PyTorch1.11.0版本中的兼容性
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:  # 如果就一个模型，直接返回
        return model[-1]

    # Return detection ensemble
    print(f'Ensemble created with {weights}\n')
    # 遍历这三个属性，从集成中的第一个模型获取这个属性的值，然后设置到集成对象上
    for k in 'names', 'nc', 'yaml':
        setattr(model, k, getattr(model[0], k))
    # 计算集成中所有模型的最大步长，然后设置到集成对象上
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    # 检查集成中所有模型的类别数是否相同。如果不同，就抛出一个异常
    assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
    # 返回集成对象
    return model
