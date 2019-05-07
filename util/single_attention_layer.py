# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: single_attention_layer.py
@time: 2019/2/26 16:57
@description: 各种attention方法
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def max_pooling_attention(mat1, mat2, W, device=torch.device("cpu")):
    mat2_T = mat2.permute(0, 2, 1)
    G = torch.bmm(torch.matmul(mat1, W), mat2_T).to(device)  # torch.size([batch_size, mat1_feature_dim, mat2_feature_dim])
    att1 = F.softmax(torch.tanh(G.max(2)[0]), 1)  # torch.size([batch_size, 10])
    att2 = F.softmax(torch.tanh(G.max(1)[0]), 1)
    att1 = torch.unsqueeze(att1, 2)  # torch.size([batch_size, 10]) -> torch.size([batch_size, 10, 1])
    att2 = torch.unsqueeze(att2, 2)

    return torch.mul(att1, mat1), torch.mul(att2, mat2), att1, att2


def pharmacology_attention(mat, att_mat, W, device=torch.device("cpu")):
    att_mat_extend = torch.unsqueeze(att_mat, 2)  # torch.size([batch_size, pharmacology_feature_dim]) -> torch.size([batch_size, pharmacology_feature_dim, 1])
    att = F.softmax(torch.bmm(torch.matmul(mat, W), att_mat_extend).to(device), 2)
    return torch.mul(att, mat), att

def mutual_attention(mat1, mat2, V, W, device=torch.device("cpu")):
    att = F.softmax(torch.matmul(torch.tanh(torch.matmul(mat2, W)), V).to(device), 2)
    return torch.mul(att, mat1), att


def test_max_pooling_attention():
    mat1 = torch.randn((6, 10, 28))
    mat2 = torch.randn((6, 10, 28))
    W = Variable(torch.randn(28, 28))
    new_mat1, new_mat2, att1, att2 = max_pooling_attention(mat1, mat2, W)
    # print("new_mat1: ", new_mat1.size())
    # print("new_mat2: ", new_mat2.size())
    # print("att1: ", att1.size())
    # print("att2: ", att2.size())


def test_pharmacology_attention():
    mat =  torch.randn((6, 10, 28))
    att_mat = torch.randn((6, 50))
    W = Variable(torch.randn(28, 50))
    new_mat, att = pharmacology_attention(mat, att_mat, W)
    # print("new_mat:", new_mat.size())
    # print("att:", att.size())


def test_mutual_attention():
    mat1 = torch.randn((6, 10, 28))
    mat2 = torch.randn((6, 10, 28))
    W = Variable(torch.randn(28, 10))
    V = Variable(torch.randn(10, 1))
    new_mat, att = mutual_attention(mat1, mat2, V, W)
    print("new_mat:", new_mat.size())
    print("att:", att.size())


if __name__ == '__main__':
    # apply a test case for max_pooling_attention
    # test_max_pooling_attention()

    # apply a test case for mutual_attention
    # test_mutual_attention()

    pass
