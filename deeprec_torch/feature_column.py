# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Date    : 2022/1/19
# @Contact : liaozhi_edo@163.com


"""
    特征列
"""


# packages
from collections import namedtuple


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
                                  ['sparsefeat', 'maxlen', 'length_name',
                                   'combiner', 'weight_name', 'weight_norm'])):
    __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, length_name, combiner='mean',
                weight_name=None, weight_norm=False):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, length_name,
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


if __name__ == '__main__':
    pass
