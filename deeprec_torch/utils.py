# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Date    : 2022/1/20
# @Contact : liaozhi_edo@163.com


"""
    Utils
"""

# packages
import numpy as np
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, df):
        self.x = {name: np.array(values.tolist(), dtype=values.dtype if values.dtype != 'object' else None) for
                  name, values in df.items() if name != 'label'}
        self.y = np.array(df['label'].tolist(), dtype=df['label'].dtype)
        self.size = len(df)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = {name: values[idx] for name, values in self.x.items()}
        y = self.y[idx]

        return (x, y)


