# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021/7/29
# @Contact : liaozhi_edo@163.com


"""
    DSSM模型

    Reference:
        《Learning deep structured semantic models for web search using clickthrough data》
"""

# packages
import torch
import torch.nn as nn
from deeprec_torch.utils.utils import compute_input_dim
from deeprec_torch.feature_column import SparseFeat, VarLenSparseFeat
from deeprec_torch.inputs import build_embedding_dict, get_dense_value, embedding_lookup, get_varlen_pooling_list


class DSSM(nn.Module):
    """
    Deep Structured Semantic Model
    """
    def __init__(self, user_feature_columns, item_feature_columns, input_dict, *args, **kwargs):
        """

        :param user_feature_columns: list 用户特征列
        :param item_feature_columns: list 物品特征列
        :param input_dict: dict 特征在输入x中的位置
        :param args:
        :param kwargs:
        """
        super(DSSM, self).__init__(*args, **kwargs)
        self.user_feature_columns = user_feature_columns
        self.item_feature_columns = item_feature_columns
        self.input_dict = input_dict

        # 参数
        # feature columns, varlen sparse = seq
        self.feature_columns = self.user_feature_columns + self.item_feature_columns

        self.user_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), self.user_feature_columns))
        self.user_seq_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), self.user_feature_columns))

        self.item_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), self.item_feature_columns))
        self.item_seq_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), self.item_feature_columns))

        # embedding dict
        self.embedding_dict = build_embedding_dict(self.feature_columns)

        # dnn
        self.user_dnn = nn.Sequential(
            nn.Linear(compute_input_dim(self.user_feature_columns, None), 128),
            nn.Linear(128, 64),
        )
        self.item_dnn = nn.Sequential(
            nn.Linear(compute_input_dim(self.item_feature_columns, None), 128),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        # user
        user_dense_value_list = get_dense_value(x, self.input_dict, self.user_feature_columns)
        user_sparse_embedding_list = embedding_lookup(x, self.input_dict, self.embedding_dict,
                                                      self.user_sparse_feature_columns, to_list=True)
        user_seq_pooling_embedding_list = get_varlen_pooling_list(x, self.input_dict, self.embedding_dict,
                                                                  self.user_seq_feature_columns)

        # concat
        user_dnn_embedding_input = torch.cat(user_sparse_embedding_list + user_seq_pooling_embedding_list, dim=-1)
        user_dnn_embedding_input = torch.squeeze(user_dnn_embedding_input, dim=1)
        user_dnn_input = torch.cat(user_dense_value_list + [user_dnn_embedding_input], dim=-1)

        # item
        item_dense_value_list = get_dense_value(x, self.input_dict, self.item_feature_columns)
        item_sparse_embedding_list = embedding_lookup(x, self.input_dict, self.embedding_dict,
                                                      self.item_sparse_feature_columns, to_list=True)
        item_seq_pooling_embedding_list = get_varlen_pooling_list(x, self.input_dict, self.embedding_dict,
                                                                  self.item_seq_feature_columns)

        # concat
        item_dnn_embedding_input = torch.cat(item_sparse_embedding_list + item_seq_pooling_embedding_list, dim=-1)
        item_dnn_embedding_input = torch.squeeze(item_dnn_embedding_input, dim=1)
        item_dnn_input = torch.cat(item_dense_value_list + [item_dnn_embedding_input], dim=-1)

        # dnn output
        user_dnn_output = self.user_dnn(user_dnn_input)
        item_dnn_output = self.item_dnn(item_dnn_input)

        user_dnn_output = user_dnn_output / torch.sum(user_dnn_output * user_dnn_output, dim=1).view(-1, 1)
        item_dnn_output = item_dnn_output / torch.sum(item_dnn_output * item_dnn_output, dim=1).view(-1, 1)

        return user_dnn_output, item_dnn_output
