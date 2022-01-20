# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Date    : 2022/1/20
# @Contact : liaozhi_edo@163.com


"""
    特征交叉
"""

# packages
import torch
from torch import nn


class LogTransformLayer(nn.Module):
    def __init__(self, field_size, embedding_dim, ltl_hidden_size=6):
        super(LogTransformLayer, self).__init__()
        self.ltl_weights = nn.Parameter(torch.Tensor(field_size, ltl_hidden_size))
        self.ltl_biases = nn.Parameter(torch.Tensor(1, 1, ltl_hidden_size))
        self.bn = nn.ModuleList([nn.BatchNorm1d(embedding_dim) for i in range(2)])

    def forward(self, x):
        x = torch.clamp(torch.abs(x), min=1e-4, max=float("Inf"))
        x = torch.transpose(x, 1, 2)
        # Logarithmic transformation layer
        x = torch.log(x)

        x = self.bn[0](x)
        x = torch.matmul(x, self.ltl_weights) + self.ltl_biases
        x = torch.exp(x)
        x = self.bn[1](x)
        x = torch.flatten(x, start_dim=1)

        return x