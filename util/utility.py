# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: utility.py
@time: 2019/2/12 0:55
@description:
"""

import numpy as np
import torch


def hierarchy_rows_padding(padding_rows, padding_cols, ndarray_list):
    vec_paddings = np.zeros((padding_rows, padding_cols))
    path_len = len(ndarray_list)
    if path_len < padding_rows:
        vec_paddings[0:path_len - 2, :] = ndarray_list[0:path_len - 2, :]
        vec_paddings[padding_rows - 2:, :] = ndarray_list[path_len - 2:, :]

    return vec_paddings


def hierarchy_cols_padding(x, vec_dim):
    """description_len长度不一，需要进行对齐"""
    # x.size(): (path_len, description_len, word2vec_len)
    # max_description_len to pad
    x_len = []

    for x_description in x:
        x_len.append(len(x_description))

    max_description_len = max(x_len)
    new_x = []
    for i in range(len(x)):
        new_x_description = np.zeros((max_description_len, vec_dim))
        new_x_description[0: x_len[i]] = x[i]
        new_x.append(new_x_description)

    return torch.DoubleTensor(new_x), np.array(x_len)


def hierarchy_path_align(batch_tensors, x_len):
    batch_size = batch_tensors.size(0)
    path_len_pad = batch_tensors.size(1)
    for i in range(batch_size):
        tensor = batch_tensors[i]
        path_len = x_len[i]
        vec_dim = tensor.size(1)
        tmp_tensor = tensor[path_len-2:path_len]
        # print("tmp_tenor: ", tmp_tensor.size())
        # print("tenor: ", tensor.size())
        # print("path_len: ", path_len)
        tensor[path_len-2:path_len] = torch.zeros(2, vec_dim)
        tensor[path_len_pad-2:] = tmp_tensor

    return batch_tensors

