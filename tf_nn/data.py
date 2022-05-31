# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Date    : 2022/5/30
# @Contact : liaozhi_edo@163.com


"""
    TF-Data
"""

# packages
import numpy as np
import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64list_feature(value_list):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))


def build_data_dict(df, varlen_sparse_feature_columns=None):
    """
    DataFrame转换成tf模型输入

    :param df:
    :param varlen_sparse_feature_columns:
    :return:
    """
    # 1,构建模型输入
    data_dict = dict()
    for name, value in df.items():
        value = value.values
        if varlen_sparse_feature_columns and name in varlen_sparse_feature_columns:
            data_dict[name] = np.vstack(value)
        else:
            data_dict[name] = value

    return data_dict


def read_tfrecord_dataset():
    pass
