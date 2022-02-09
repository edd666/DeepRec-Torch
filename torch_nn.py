# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Date    : 2022/1/24
# @Contact : liaozhi_edo@163.com


"""
    Pytorch模型常用代码
"""

# packages
import torch
from torch import nn
from collections import namedtuple, OrderedDict


class DenseFeat(namedtuple('DenseFeat',
                           ['name', 'dimension', 'dtype'])):
    """
    数值特征
    """
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype='float32'):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()


class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash',
                             'dtype', 'embedding_name', 'weight', 'trainable'])):
    """
    类别特征
    """
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim,
                use_hash=False, dtype='int64', embedding_name=None,
                weight=None, trainable=True):

        if embedding_name is None:
            embedding_name = name

        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim,
                                              use_hash, dtype, embedding_name, weight,
                                              trainable)

    def __hash__(self):
        return self.name.__hash__()


class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
                                  ['sparsefeat', 'length_name', 'combiner',
                                   'weight_name', 'weight_norm'])):
    __slots__ = ()

    def __new__(cls, sparsefeat, length_name, combiner='mean',
                weight_name=None, weight_norm=False):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, length_name,
                                                    combiner, weight_name, weight_norm)

    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def weight(self):
        return self.sparsefeat.weight

    @property
    def trainable(self):
        return self.sparsefeat.trainable

    def __hash__(self):
        return self.name.__hash__()


def build_embedding_dict(feature_columns):
    """
    基于特征列(feature_columns)构建Embedding字典

    注意:
        1,fc.weight(预训练Embedding)若不为None,必须为Tensor.

    :param feature_columns: list 特征列
    :return:
        embedding_dict: ModuleDict,形如{embedding_name: embedding_table}
    """
    # 1,获取SparseFeat和VarLenSparseFeat
    sparse_feature_columns = list(filter(
        lambda x: isinstance(x, SparseFeat), feature_columns))
    varlen_sparse_feature_columns = list(filter(
        lambda x: isinstance(x, VarLenSparseFeat), feature_columns))

    # 2,构建Embedding字典
    embedding_dict = nn.ModuleDict()
    for fc in sparse_feature_columns + varlen_sparse_feature_columns:
        embedding_dict[fc.embedding_name] = nn.Embedding(num_embeddings=fc.vocabulary_size,
                                                         embedding_dim=fc.embedding_dim,
                                                         _weight=fc.weight)
        if not fc.trainable:
            embedding_dict[fc.embedding_name].weight.requires_grad = fc.trainable

    return embedding_dict


def get_dense_value(x, feature_columns):
    """
    获取数值输入

    :param x: dict 输入
    :param feature_columns: list 特征列
    :return:
        dense_value_list: list 数值输入
    """
    # 1,获取DenseFeat
    dense_value_list = list()
    dense_feature_columns = list(filter(
        lambda f: isinstance(f, DenseFeat), feature_columns))
    for fc in dense_feature_columns:
        dense_value_list.append(x[fc.name].reshape((-1, 1)).float())  # torch.float32

    return dense_value_list


def embedding_lookup(x, embedding_dict, query_feature_columns, to_list=False):
    """
    embedding查询

    :param x: dict 输入
    :param embedding_dict: embedding字典,形如{embedding_name: embedding_table}
    :param query_feature_columns: list 待查询的特征列
    :param to_list: bool 是否转成list
    :return:
    """
    query_embedding_dict = OrderedDict()
    for fc in query_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            raise ValueError('hash embedding lookup has not yet been implemented.')
        else:
            lookup_idx = x[feature_name].long()  # torch.int64

        query_embedding_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)

    if to_list:
        return list(query_embedding_dict.values())

    return query_embedding_dict


class SequencePoolingLayer(nn.Module):
    def __init__(self, mode):
        super(SequencePoolingLayer, self).__init__()
        if mode not in ['sum', 'mean', 'max']:
            raise ValueError('parameter mode should in [sum, mean, max]')
        self.mode = mode

    @staticmethod
    def _sequence_mask(seq_len, maxlen, dtype=torch.bool):
        row_vector = torch.arange(0, maxlen, 1, device=seq_len.device)
        mask = row_vector < seq_len

        return mask.type(dtype)

    def forward(self, x):
        seq_value, seq_len = x  # [B, T, E], [B, 1]
        mask = self._sequence_mask(seq_len, seq_value.shape[1], torch.float32)
        mask = mask.unsqueeze(dim=2)
        mask = torch.repeat_interleave(mask, seq_value.shape[-1], dim=2)  # [B, T, E]

        if self.mode == 'max':
            seq_value = seq_value - (1 - mask) * 1e9
            seq_value = torch.max(seq_value, dim=1, keepdim=False)[0]

            return seq_value

        seq_value = seq_value * mask
        seq_value = torch.sum(seq_value, dim=1, keepdim=False)

        if self.mode == 'mean':
            seq_value = torch.div(seq_value, seq_len + 1e-8)

        return seq_value


def get_varlen_pooling_list(x, embedding_dict, varlen_sparse_feature_columns):
    """
    对序列特征(VarLenSparseFeat)进行Pooling操作

    :param x: dict 输入
    :param embedding_dict: embedding字典,形如{embedding_name: embedding_table}
    :param varlen_sparse_feature_columns: list 序列特征
    :return:
    """
    # 1,对VarLenSparseFeat的embedding进行Pooling操作
    pooling_value_list = list()
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.weight_name is not None:
            # weighted pooling
            raise ValueError('pooling with weight has not yet been implemented.')
        else:
            seq_value = embedding_dict[embedding_name](x[feature_name].long())
            seq_len = x[fc.length_name].reshape((-1, 1)).float()
            pooling_value = SequencePoolingLayer(mode=fc.combiner)([seq_value, seq_len])

        pooling_value_list.append(pooling_value)

    return pooling_value_list


def input_from_feature_columns(x, feature_columns, embedding_dict):
    """
    输入层

    :param x: dict 输入
    :param feature_columns: list 特征列
    :param embedding_dict: embedding字典,形如{embedding_name: embedding_table}
    :return:
    """
    # 1,划分特征类型
    sparse_feature_columns = list(filter(
        lambda f: isinstance(f, SparseFeat), feature_columns))
    varlen_sparse_feature_columns = list(filter(
        lambda f: isinstance(f, VarLenSparseFeat), feature_columns))

    # 2,输入层转换
    dense_value_list = get_dense_value(x, feature_columns)
    sparse_embedding_list = embedding_lookup(x, embedding_dict, sparse_feature_columns, True)
    varlen_sparse_embedding_list = get_varlen_pooling_list(x, embedding_dict, varlen_sparse_feature_columns)

    return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list
