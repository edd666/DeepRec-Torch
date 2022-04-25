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
