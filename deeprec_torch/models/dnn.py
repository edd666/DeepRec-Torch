# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Date    : 2022/4/24
# @Contact : liaozhi_edo@163.com


"""
    DNN
"""

# packages
import torch
from torch import nn
from ..feature_column import DenseFeat
from ..inputs import build_embedding_dict, input_from_feature_columns


class DNN(nn.Module):
    def __init__(self, feature_columns):
        super(DNN, self).__init__()
        self.feature_columns = feature_columns
        self.embedding_dict = build_embedding_dict(self.feature_columns)
        in_dim = 0
        for fc in self.feature_columns:
            if isinstance(fc, DenseFeat):
                in_dim += fc.dimension
            else:
                in_dim += fc.embedding_dim
        self.dnn = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        sparse_embedding_list, dense_value_list = input_from_feature_columns(
            x, self.feature_columns, self.embedding_dict)
        dnn_in = torch.cat(sparse_embedding_list + dense_value_list, dim=-1)
        logit = self.dnn(dnn_in)

        return logit
