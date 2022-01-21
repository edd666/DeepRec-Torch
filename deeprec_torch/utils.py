# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Date    : 2022/1/20
# @Contact : liaozhi_edo@163.com


"""
    Utils
"""

# packages
import torch
import random
import numpy as np
from torch.utils.data import Dataset


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


class CustomDataset(Dataset):
    def __init__(self, df):
        self.x = {name: np.array(values.tolist(), dtype=values.dtype if values.dtype != 'object' else None)
                  for name, values in df.items() if name != 'label'}
        self.y = np.array(df['label'].tolist(), dtype=df['label'].dtype)
        self.size = len(df)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = {name: values[idx] for name, values in self.x.items()}
        y = self.y[idx]

        return (x, y)


