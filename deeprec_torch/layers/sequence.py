# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Date    : 2022/1/20
# @Contact : liaozhi_edo@163.com


"""
    序列特征处理
"""

# packages
import torch
import torch.nn as nn


class SequencePoolingLayer(nn.Module):
    def __init__(self, mode):
        super(SequencePoolingLayer, self).__init__()
        if mode not in ['sum', 'mean', 'max']:
            raise ValueError('parameter mode should in [sum, mean, max]')
        self.mode = mode

    @staticmethod
    def _sequence_mask(seq_len, maxlen, dtype=torch.bool):
        row_vector = torch.arange(0, maxlen, 1, device=seq_len.device)
        mask = row_vector < seq_len

        return mask.type(dtype)

    def forward(self, x):
        seq_value, seq_len = x  # [B, T, E], [B, 1]
        mask = self._sequence_mask(seq_len, seq_value.shape[1], torch.float32)
        mask = mask.unsqueeze(dim=2)
        mask = torch.repeat_interleave(mask, seq_value.shape[-1], dim=2)  # [B, T, E]

        if self.mode == 'max':
            seq_value = seq_value - (1 - mask) * 1e9
            seq_value = torch.max(seq_value, dim=1, keepdim=False)[0]

            return seq_value

        seq_value = seq_value * mask
        seq_value = torch.sum(seq_value, dim=1, keepdim=False)

        if self.mode == 'mean':
            seq_value = torch.div(seq_value, seq_len + 1e-8)

        return seq_value


