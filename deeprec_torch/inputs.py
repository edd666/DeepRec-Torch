# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Date    : 2022/1/20
# @Contact : liaozhi_edo@163.com


"""
    输入层
"""

# packages
from torch import nn
from collections import OrderedDict
from .layers.sequence import SequencePoolingLayer
from .feature_column import DenseFeat, SparseFeat, VarLenSparseFeat


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
    dense_feature_columns = list(filter(
        lambda f: isinstance(f, DenseFeat), feature_columns))
    sparse_feature_columns = list(filter(
        lambda f: isinstance(f, SparseFeat), feature_columns))
    varlen_sparse_feature_columns = list(filter(
        lambda f: isinstance(f, VarLenSparseFeat), feature_columns))

    # 2,输入层转换
    dense_value_list = get_dense_value(x, dense_feature_columns)
    sparse_embedding_list = embedding_lookup(x, embedding_dict, sparse_feature_columns, True)
    varlen_sparse_embedding_list = get_varlen_pooling_list(x, embedding_dict, varlen_sparse_feature_columns)

    return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list


