# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-07-14
# @Contact : liaozhi_edo@163.com


"""
    特征列
"""

# packages
from collections import namedtuple, OrderedDict


# General Setting
EMBEDDING_DIM = 12
GROUP_NAME = 'default'


class DenseFeat(namedtuple('DenseFeat',
                           ['name', 'dimension', 'dtype'])):
    """
    数值特征
    """
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype='float32', *args, **kwargs):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()


class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash',
                             'dtype', 'embedding_name', 'group_name', 'weight', 'trainable'])):
    """
    类别特征
    """
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=EMBEDDING_DIM,
                use_hash=False, dtype='int32', embedding_name=None,
                group_name=GROUP_NAME, weight=None, trainable=True, *args, **kwargs):

        if embedding_dim == 'auto':
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))

        if embedding_name is None:
            embedding_name = name

        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim,
                                              use_hash, dtype, embedding_name, group_name,
                                              weight, trainable)

    def __hash__(self):
        return self.name.__hash__()


class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
                                  ['sparsefeat', 'maxlen', 'combiner',
                                   'weight_name', 'weight_norm'])):
    """
    序列特征
    """
    __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, combiner='mean',
                weight_name=None, weight_norm=True, *args, **kwargs):

        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, combiner,
                                                    weight_name, weight_norm)

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
    def group_name(self):
        return self.sparsefeat.group_name

    @property
    def weight(self):
        return self.sparsefeat.weight

    @property
    def trainable(self):
        return self.sparsefeat.trainable

    def __hash__(self):
        return self.name.__hash__()


def build_input_dict(feature_columns):
    """
    基于feature_columns构建特征与模型输入(tensor)之间的索引关系

    :param feature_columns: list 特征列
    :return:
        input_dict: OrderedDict 形如{feature_name: (idx: idx+dim)}, (idx: idx+dim)表示特征在tensor中的位置
    """
    start = 0  # tensor起始维度
    input_dict = OrderedDict()
    for fc in feature_columns:
        if isinstance(fc, DenseFeat):
            input_dict[fc.name] = (start, start + fc.dimension)
            start += fc.dimension
        elif isinstance(fc, SparseFeat):
            input_dict[fc.name] = (start, start + 1)
            start += 1
        elif isinstance(fc, VarLenSparseFeat):
            # 序列
            input_dict[fc.name] = (start, start + fc.maxlen)
            start += fc.maxlen

            # 序列权重
            if fc.weight_name is not None:
                input_dict[fc.weight_name] = (start, start + fc.maxlen)
                start += fc.maxlen
        else:
            raise ValueError('Invalid type in feature columns.')

    return input_dict


if __name__ == '__main__':
    feat = DenseFeat('price', dimension=2)
    print(feat.name, feat.dtype, feat.dimension)
    sparse_feat = SparseFeat('age', 100)
    print(sparse_feat)
    varlen_sparse_feat = VarLenSparseFeat(SparseFeat('item_id', 10), maxlen=5)
    print(varlen_sparse_feat)
    pass