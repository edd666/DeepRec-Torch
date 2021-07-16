# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-07-15
# @Contact : liaozhi_edo@163.com


"""
    Dataset

    注意:
        1,将样本的所有特征(单值、序列)拼成一个长的tensor,在模型内部依据index去找相应的特征.
          如[1, 2, [4, 5, 6]] -> tensor([1, 2, 4, 5, 6])
"""

# packages
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """
    训练/验证数据集的Dataset

    """
    def __init__(self, data):
        """
        初始化构造函数

        :param data: tuple 特征(DataFrame)和标签(Series),形如(x,y)
        """
        self.x = data[0].values  # 2D array
        self.y = data[1].values  # 1D array

    def __len__(self):
        """
        Dataset大小

        :return:
        """
        return len(self.x)

    def __getitem__(self, idx):
        """
        获取特征和样本

        :param idx: int 索引
        :return:
        """
        x = []
        for v in self.x[idx]:
            if isinstance(v, list):
                x.extend(v)
            else:
                x.append(v)
        y = self.y[idx]

        data = (np.array(x), y)

        return data


class TestDataset(Dataset):
    """
    测试数据集Dataset

    """
    def __init__(self, data):
        """
        初始化构造函数

        :param data: tuple 特征,形如(x,)
        """
        self.x = data[0].values  # 2D array

    def __len__(self):
        """
        Dataset大小

        :return:
        """
        return len(self.x)

    def __getitem__(self, idx):
        """
        获取特征

        :param idx: int 索引
        :return:
        """
        x = []
        for v in self.x[idx]:
            if isinstance(v, list):
                x.extend(v)
            else:
                x.append(v)

        return np.array(x)


def build_input_index_dict(data):
    """"""
    pass
