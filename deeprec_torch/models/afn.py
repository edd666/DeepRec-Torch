# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Date    : 2022/1/20
# @Contact : liaozhi_edo@163.com


"""
    AFN模型

    Reference:
    [1] Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions.
"""

# packages
import torch
from torch import nn
from ..feature_column import DenseFeat
from ..layers.interaction import LogTransformLayer
from ..inputs import build_embedding_dict, input_from_feature_columns


class AFN(nn.Module):
    def __init__(self, linear_feature_columns, dnn_feature_columns, ltl_hidden_size=600):
        super(AFN, self).__init__()
        self.linear_feature_columns = linear_feature_columns
        self.dnn_feature_columns = dnn_feature_columns
        self.ltl_hidden_size = ltl_hidden_size

        # afn-cross feature
        self.embedding_dict = build_embedding_dict(self.linear_feature_columns)
        sparse_feature_columns = [fc for fc in self.linear_feature_columns if not isinstance(fc, DenseFeat)]
        embedding_dim_set = set([fc.embedding_dim for fc in sparse_feature_columns])
        if len(embedding_dim_set) > 1:
            raise ValueError('embedding_dim of SparseFeat and VarlenSparseFeat must be same in this model!')
        else:
            self.embedding_dim = list(embedding_dim_set)[0]

        self.ltl = LogTransformLayer(field_size=len(sparse_feature_columns), embedding_dim=self.embedding_dim,
                                     ltl_hidden_size=self.ltl_hidden_size)
        self.afn_dnn = nn.Sequential(
            nn.Linear(self.embedding_dim * self.ltl_hidden_size, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),

            nn.Linear(400, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),

            nn.Linear(400, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),

            nn.Linear(400, 1),
            nn.Sigmoid()
        )

        # dnn
        self.dnn_embedding_dict = build_embedding_dict(self.dnn_feature_columns)
        dnn_in_dim = 0
        for fc in self.dnn_feature_columns:
            if isinstance(fc, DenseFeat):
                dnn_in_dim += fc.dimension
            else:
                dnn_in_dim += fc.embedding_dim
        self.dnn = nn.Sequential(
            nn.Linear(dnn_in_dim, 512),
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

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # ensemble
        self.w_afn = nn.Parameter(torch.Tensor([0.5]))
        self.w_dnn = nn.Parameter(torch.Tensor([0.5]))
        self.b = nn.Parameter(torch.Tensor([0.]))

    def forward(self, x):
        # afn
        sparse_embedding_list, _ = input_from_feature_columns(x, self.linear_feature_columns, self.embedding_dict)
        sparse_embedding_list = [torch.unsqueeze(embed, dim=1) for embed in sparse_embedding_list]
        ltl_in = torch.cat(sparse_embedding_list, dim=1)
        ltl_out = self.ltl(ltl_in)
        afn_logit = self.afn_dnn(ltl_out)

        # dnn
        dnn_sparse_embedding_list, dnn_dense_value_list = input_from_feature_columns(x, self.dnn_feature_columns,
                                                                                     self.dnn_embedding_dict)
        dnn_in = torch.cat(dnn_sparse_embedding_list + dnn_dense_value_list, dim=-1)
        dnn_logit = self.dnn(dnn_in)

        logit = torch.sigmoid(self.w_afn * afn_logit.detach() + self.w_dnn * dnn_logit.detach() + self.b)

        return logit, afn_logit, dnn_logit


