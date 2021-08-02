# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021/7/30
# @Contact : liaozhi_edo@163.com


"""
    Utils
"""

# packages
import torch
import random
import numpy as np
from deeprec_torch.feature_column import DenseFeat, SparseFeat, VarLenSparseFeat


def compute_input_dim(feature_columns, behavior_columns=None):
    """
    基于特征列计算特征concat后的维度-DNN的输入维度

    :param feature_columns: list 特征列
    :param behavior_columns: list 行为列
    :return:
        input_dim: int 特征concat后的维度,即DNN的输入维度
    """
    # 1,特征划分
    dense_feature_columns = list(filter(
        lambda f: isinstance(f, DenseFeat), feature_columns))
    sparse_feature_columns = list(filter(
        lambda x: isinstance(x, SparseFeat), feature_columns))
    varlen_sparse_feature_columns = list(filter(
        lambda x: isinstance(x, VarLenSparseFeat), feature_columns))
    behavior_columns = behavior_columns if behavior_columns is not None else []
    hist_behavior_columns = ['hist_' + str(col) for col in behavior_columns]
    query_feature_columns = [fc for fc in sparse_feature_columns if fc.name in behavior_columns]
    keys_feature_columns = [fc for fc in varlen_sparse_feature_columns if fc.name in hist_behavior_columns]

    # 2,维度计算
    input_dim = 0
    for fc in dense_feature_columns:
        input_dim += fc.dimension

    for fc in sparse_feature_columns + varlen_sparse_feature_columns:
        if fc not in query_feature_columns + keys_feature_columns:
            input_dim += fc.embedding_dim

    for idx in range(len(behavior_columns)):
        q_fc = query_feature_columns[idx]
        k_fc = keys_feature_columns[idx]
        if 'hist_' + q_fc.name == k_fc.name and q_fc.embedding_dim == k_fc.embedding_dim:
            input_dim += q_fc.embedding_dim
        else:
            raise ValueError('error in compute_input_dim')

    return input_dim


def setup_seed(seed):
    """
    设置随机种子

    :param seed: int 随机种子
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)

    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return
