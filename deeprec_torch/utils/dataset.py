# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-07-15
# @Contact : liaozhi_edo@163.com


"""
    Dataset

    注意:
        1,将样本的所有特征(单值、序列)依据顺序(input_dict)拼成一个长的tensor,在模型内部依据index去找相应的特征.
          如[1, 2, [4, 5, 6]] -> tensor([1, 2, 4, 5, 6])
"""

# packages
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """
    训练/验证数据集的Dataset

    """
    def __init__(self, data, input_dict):
        """
        初始化构造函数

        :param data: tuple 特征(DataFrame)和标签(Series),形如(x,y)
        :param input_dict: dict 特征与模型输入tensor维度之间的对应关系
        """
        x = data[0]  # DataFrame
        x = [np.array(x[feature_name].tolist()) for feature_name in input_dict]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        x = np.concatenate(x, axis=-1)
        self.x = x  # 2D array
        self.y = data[1].values  # 1D array
        self.input_dict = input_dict

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
        x = self.x[idx]
        y = self.y[idx]
        data = (x, y)

        return data


class TestDataset(Dataset):
    """
    测试数据集Dataset

    """
    def __init__(self, data, input_dict):
        """
        初始化构造函数

        :param data: tuple 特征(DataFrame),形如(x,)
        :param input_dict: dict 特征与模型输入tensor维度之间的对应关系
        """
        x = data[0]  # DataFrame
        x = [np.array(x[feature_name].tolist()) for feature_name in input_dict]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        x = np.concatenate(x, axis=-1)
        self.x = x  # 2D array
        self.input_dict = input_dict

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
        x = self.x[idx]

        return x
