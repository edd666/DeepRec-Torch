# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-07-15
# @Contact : liaozhi_edo@163.com


"""
    模型输入层相关函数,如embedding_lookup
"""

# packages
import torch.nn as nn
from collections import OrderedDict
from deeprec_torch.feature_column import DenseFeat, SparseFeat, VarLenSparseFeat
from deeprec_torch.layers.sequence import WeightedSequenceLayer, SequencePoolingLayer


def build_embedding_dict(feature_columns, device='cpu'):
    """
    基于特征列(feature_columns)构建Embedding字典

    注意:
        1,fc.weight(预训练Embedding)若不为None,必须为Tensor.

    :param feature_columns: list 特征列
    :param device: str device
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
                                                         _weight=fc.weight,)
        if not fc.trainable:
            embedding_dict[fc.embedding_name].weight.requires_grad = fc.trainable

    return embedding_dict.to(device)


def get_dense_value(x, input_dict, feature_columns):
    """
    获取数值输入

    :param x: Tensor 模型输入
    :param input_dict: dict 特征在模型输入x中的位置
    :param feature_columns: list 特征列
    :return:
        dense_value_list: list of tensor 数值输入(torch.float32)
    """
    # 1,获取数值输入
    dense_value_list = []
    dense_feature_columns = list(filter(
        lambda f: isinstance(f, DenseFeat), feature_columns))
    for fc in dense_feature_columns:
        lookup_idx = input_dict[fc.name]
        dense_value_list.append(x[:, lookup_idx[0]:lookup_idx[1]].float())  # tensor.float() == tensor.to(torch.float32)

    return dense_value_list


def embedding_lookup(x, input_dict, embedding_dict, query_feature_columns, to_list=False):
    """
    embedding查询

    注意:
        1,查询结果的维度为(batch_size, 1/maxlen, embedding_dim).

    :param x: Tensor 模型输入
    :param input_dict: dict 特征在模型输入x中的位置
    :param embedding_dict: ModuleDict embedding字典,形如{embedding_name: embedding_table}
    :param query_feature_columns: list 待查询的特征列
    :param to_list: bool 是否转成list
    :return:
    """
    # 1,查询
    query_embedding_dict = OrderedDict()
    for fc in query_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            raise ValueError('hash embedding lookup has not yet been implemented.')
        else:
            lookup_idx = input_dict[feature_name]
            lookup_idx = x[:, lookup_idx[0]:lookup_idx[1]].long()  # tensor.long() == tensor.to(torch.int64)

        query_embedding_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)

    if to_list:
        return list(query_embedding_dict.values())

    return query_embedding_dict


def get_varlen_pooling_list(x, input_dict, embedding_dict, varlen_sparse_feature_columns):
    """
    对序列特征(VarLenSparseFeat)进行Pooling操作

    :param x: Tensor 模型输入
    :param input_dict: dict 特征在模型输入x中的位置
    :param embedding_dict: ModuleDict embedding字典,形如{embedding_name: embedding_table}
    :param varlen_sparse_feature_columns: list 序列特征
    :return:
    """
    # 1,对序列特征进行Pooling操作
    pooling_value_list = []
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name

        # seq_value, mask
        lookup_idx = input_dict[feature_name]
        sequence = x[:, lookup_idx[0]:lookup_idx[1]].long()  # torch.int64
        seq_value = embedding_dict[embedding_name](sequence)
        mask = sequence != 0
        if fc.weight_name is not None:
            # weighted sequence
            weight_idx = input_dict[fc.weight_name]
            weight = x[:, weight_idx[0]:weight_idx[1]]
            seq_value = WeightedSequenceLayer(
                mask_zero=True, weight_normalization=fc.weight_norm)((seq_value, mask, weight))

        pooling_value = SequencePoolingLayer(mode=fc.combiner, mask_zero=True)((seq_value, mask))
        pooling_value_list.append(pooling_value)

    return pooling_value_list
