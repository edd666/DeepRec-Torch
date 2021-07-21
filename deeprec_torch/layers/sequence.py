# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-07-20
# @Contact : liaozhi_edo@163.com


"""
    序列特征处理
"""

# packages
import torch
import torch.nn as nn
from torch.nn import functional as F


class SequencePoolingLayer(nn.Module):
    """
    The SequencePoolingLayer is used to apply pooling operation(sum,mean,max) on variable-length
    sequence feature/multi-value feature.

    Input shape:
        seq_value is a 3D tensor with shape (batch_size, T, embedding_size)
        seq_len is a 2D tensor with shape (batch_size, 1)
        mask is a 2D tensor with shape (batch_size, T)

    Output shape:
        3D tensor with shape (batch_size, 1, embedding_size)
    """
    def __init__(self, mode, mask_zero, **kwargs):
        """

        :param mode: str Pooling的方式
        :param mask_zero: bool 是否支持mask
        :param kwargs:
        """
        super(SequencePoolingLayer, self).__init__(**kwargs)
        if mode not in ('sum', 'mean', 'max'):
            raise ValueError("mode must be sum, mean or max")
        self.mode = mode
        self.mask_zero = mask_zero
        self.eps = torch.FloatTensor([1e-8])

    @staticmethod
    def _sequence_mask(seq_len, maxlen, dtype=torch.bool):
        row_vector = torch.arange(0, maxlen, 1, device=seq_len.device)
        mask = row_vector < seq_len

        return mask.type(dtype)

    def forward(self, x):
        if self.mask_zero:
            # mask
            seq_value, mask = x  # [B, T, E], [B, T]
            mask = mask.float()  # torch.float32
            seq_len = torch.sum(mask, dim=-1, keepdim=True)
            mask = mask.unsqueeze(dim=2)
        else:
            seq_value, seq_len = x  # [B, T, E], [B, 1]
            seq_len = seq_len.float()  # torch.float32
            mask = self._sequence_mask(seq_len, seq_value.shape[1], torch.float32)
            mask = mask.unsqueeze(dim=2)

        embedding_size = seq_value.shape[-1]
        mask = torch.repeat_interleave(mask, embedding_size, dim=2)  # [B, T, E]

        # max
        if self.mode == 'max':
            seq_value = seq_value - (1 - mask) * 1e9

            return torch.max(seq_value, dim=1, keepdim=True)[0]

        # sum
        seq_value = torch.sum(seq_value * mask, dim=1, keepdim=False)

        # mean
        if self.mode == 'mean':
            self.eps = self.eps.to(seq_value.device)
            seq_value = torch.div(seq_value, seq_len + self.eps)

        seq_value = seq_value.unsqueeze(dim=1)

        return seq_value


class WeightedSequenceLayer(nn.Module):
    """
    WeightedSequenceLayer is used to apply weight score on variable-length sequence feature.

    Input shape:
        seq_value is a 3D tensor with shape (batch_size, T, embedding_size)
        seq_len is a 2D tensor with shape (batch_size, 1)
        mask is a 2D tensor with shape (batch_size, T)
        weight is a 2D tensor with shape (batch_size, T)

    Output shape:
        3D tensor with shape (batch_size, T, embedding_size)
    """
    def __init__(self, mask_zero, weight_normalization=True, **kwargs):
        """

        :param mask_zero: bool 是否支持mask zero
        :param weight_normalization: bool 是否权重归一化
        :param kwargs:
        """
        super(WeightedSequenceLayer, self).__init__(**kwargs)
        self.mask_zero = mask_zero
        self.weight_normalization = weight_normalization

    @staticmethod
    def _sequence_mask(seq_len, maxlen, dtype=torch.bool):
        row_vector = torch.arange(0, maxlen, 1, device=seq_len.device)
        mask = row_vector < seq_len

        return mask.type(dtype)

    def forward(self, x):
        if self.mask_zero:
            # mask
            seq_value, mask, weight = x
        else:
            # seq_len
            seq_value, seq_len, weight = x
            mask = self._sequence_mask(seq_len, seq_value.shape[1], torch.bool)

        weight = weight.float()  # torch.float32

        if self.weight_normalization:
            padding = torch.ones_like(weight) * (-2 ** 31 + 1)
        else:
            padding = torch.zeros_like(weight)

        weight = torch.where(mask, weight, padding)

        if self.weight_normalization:
            weight = F.softmax(weight, dim=1)

        embedding_size = seq_value.shape[-1]
        weight = weight.unsqueeze(2)
        weight = torch.repeat_interleave(weight, embedding_size, dim=2)

        return seq_value * weight
