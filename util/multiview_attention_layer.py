# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: multiview_attention_layer.py
@time: 2019/2/28 15:48
@description:  multiview attention的实现
"""
import numpy
import torch
import torch.nn.functional as F
from torch.autograd import Variable


## 勿删，帮助理解single_attention各个待训练的参数
# def pharmacology_attention(phy_mat, mat, W_p, U, V, func):
#     """
#     以药理特征为base, 分别对药物文本特征，药物network_embedding特征作attention
#     :param phy_mat: torch.size([batch_size, 10, dim])
#     :param mat:     torch.size([batch_size, 10, dim1])
#     :param W_p:     torch.size([1, 10])
#     :param U:       torch.size([dim1, 10])
#     :param V:       torch.size([10, 1])
#     :param func:    torch.mean(), torch.max()
#     :return:
#     """
#     O = func(phy_mat, 2)  # torch.size([batch_size, 10, 1])
#     att = F.softmax(torch.matmul(torch.tanh(torch.matmul(O, W_p) + torch.matmul(mat, U)), V), 2)  # torch.size([batch_size, 10, 1])
#     return torch.mul(att, mat), att
#
#
# def mutual_attention(mat_t, mat_d, W_t, U_d, V_td, W_d, U_t, V_dt, func):
#     """
#     :param mat_t:
#     :param mat_d:
#     以药物文本特征为base，对药物network_embedding特征作attention
#     :param W_t:    torch.size([1, 10])
#     :param U_d:    torch.size([dim_d, 10])
#     :param V_td:   torch.size([10, 1])
#     以对药物network_embedding特征作为base，对药物文本特征作attention
#     :param W_d:    torch.size([1, 10])
#     :param U_t:    torch.size([dim_t, 10])
#     :param V_dt:   torch.size([10, 1])
#     :param func:  torch.mean(), torch.max()
#     :return:
#     """
#     O_t = func(mat_t, 2)  # torch.size([batch_size, 10, 1])
#     att_t = F.softmax(torch.matmul(torch.tanh(torch.matmul(O_t, W_t) + torch.matmul(mat_d, U_d)), V_td), 2)
#     O_d = func(mat_d, 2)
#     att_d = F.softmax(torch.matmul(torch.tanh(torch.matmul(O_d, W_d) + torch.matmul(mat_t, U_t)), V_dt), 2)
#     return torch.mul(att_t, mat_t), torch.mul(att_d, mat_d), att_t, att_d


def single_attention(mat, att_mat, W_p, U, V, func):
    # print("att_mat:", att_mat.size())
    if func.__name__ == "max":
        func_mat = func(att_mat, 2)[0]
    else:
        func_mat = func(att_mat, 2)
    # print("func_mat: ", func_mat.size())
    O = torch.unsqueeze(func_mat, 2)  # torch.size([batch_size, 10]) --> torch.size([batch_size, 10, 1])
    att = F.softmax(torch.matmul(torch.tanh(torch.matmul(O, W_p) + torch.matmul(mat, U)), V), 1)  # torch.size([batch_size, 10, 1])
    return att


def co_attention(mat1, mat2, S):
    """
    :param mat1:  torch.size([batch_size, 10, dim1])
    :param mat2:  torch.size([batch_size, 10, dim2])
    :param S:     torch.size([dim1, dim2])
    :return:
    """
    mat2_T = mat2.permute(0, 2, 1)
    G = torch.bmm(torch.matmul(mat1, S), mat2_T)#.to(device)  # torch.size([batch_size, mat1_feature_dim, mat2_feature_dim])
    att1 = F.softmax(torch.tanh(G.max(2)[0]), 1)  # torch.size([batch_size, 10])
    att2 = F.softmax(torch.tanh(G.max(1)[0]), 1)
    att1 = torch.unsqueeze(att1, 2)  # torch.size([batch_size, 10]) -> torch.size([batch_size, 10, 1])
    att2 = torch.unsqueeze(att2, 2)

    return att1, att2

def multiview_attentive_representation(w, ph_mat, textual_mat, deepwalk_mat, W, U, V, S, wf=None):
    # pharmacology attention
    t_att_p_max  = single_attention(textual_mat, ph_mat, W[0], U[0], V[0], torch.max)
    t_att_p_mean = single_attention(textual_mat, ph_mat, W[1], U[1], V[1], torch.mean)
    d_att_p_max  = single_attention(deepwalk_mat, ph_mat,  W[2], U[2], V[2], torch.max)
    d_att_p_mean = single_attention(deepwalk_mat, ph_mat,  W[3], U[3], V[3], torch.mean)

    # textual attention
    d_att_t_max  = single_attention(deepwalk_mat, textual_mat, W[4], U[4], V[5], torch.max)
    d_att_t_mean = single_attention(deepwalk_mat, textual_mat, W[5], U[5], V[5], torch.mean)

    # deepwalk attention
    t_att_d_max  = single_attention(textual_mat, deepwalk_mat, W[6], U[6], V[6], torch.max)
    t_att_d_mean = single_attention(textual_mat, deepwalk_mat, W[7], U[7], V[7], torch.mean)

    # co_attention"0 1 1 0
    t_att_co, d_att_co = co_attention(textual_mat, deepwalk_mat, S)

    t_att_sum = w[0] * t_att_p_max + w[1] * t_att_p_mean + w[2] * t_att_d_max + w[3] * t_att_d_mean + t_att_co
    # t_att_sum = t_att_co
    d_att_sum = w[0] * d_att_p_max + w[1] * d_att_p_mean + w[2] * d_att_t_max + w[3] * d_att_t_mean + d_att_co
    t_att = F.softmax(t_att_sum, 1)
    d_att = F.softmax(d_att_sum, 1)

    if wf is not None:
        write_attention_to_file(wf, t_att_p_mean, t_att_d_max, t_att_co, t_att, d_att_p_mean, d_att_t_max, d_att_co, d_att)

    return t_att, d_att


def write_attention_to_file(wf, t_att_p_mean, t_att_d_max, t_att_co, t_att, d_att_p_mean, d_att_t_max, d_att_co, d_att):
    t_att_p_mean=t_att_p_mean.permute(0, 2, 1)
    t_att_d_max=t_att_d_max.permute(0, 2, 1)
    t_att_co=t_att_co.permute(0, 2, 1)
    t_att=t_att.permute(0, 2, 1)
    d_att_p_mean=d_att_p_mean.permute(0, 2, 1)
    d_att_t_max=d_att_t_max.permute(0, 2, 1)
    d_att_co=d_att_co.permute(0, 2, 1)
    d_att=d_att.permute(0, 2, 1)
    t = torch.squeeze(torch.stack((t_att_p_mean, t_att_d_max, t_att_co, t_att), dim=1))#.permute(0,2,1)
    d =  torch.squeeze(torch.stack((d_att_p_mean, d_att_t_max, d_att_co, d_att), dim=1))#.permute(0,2,1)

    batch_size = t.size(0)
    for i in range(batch_size):
        wf.write(str(t[i].cpu().numpy()))
        wf.write("\n\n")
        wf.write(str(d[i].cpu().numpy()))
        wf.write("\n=================================\n")


def no_pharmacology_mt_attultiview_attention(w, textual_mat, deepwalk_mat, W, U, V, S):
    # textual attention
    d_att_t_max  = single_attention(deepwalk_mat, textual_mat, W[4], U[4], V[5], torch.max)
    d_att_t_mean = single_attention(deepwalk_mat, textual_mat, W[5], U[5], V[5], torch.mean)

    # deepwalk attention
    t_att_d_max  = single_attention(textual_mat, deepwalk_mat, W[6], U[6], V[6], torch.max)
    t_att_d_mean = single_attention(textual_mat, deepwalk_mat, W[7], U[7], V[7], torch.mean)

    # co_attention"0 1 1 0
    t_att_co, d_att_co = co_attention(textual_mat, deepwalk_mat, S)

    t_att_sum = w[2] * t_att_d_max + w[3] * t_att_d_mean + t_att_co
    # t_att_sum = t_att_co
    d_att_sum = w[2] * d_att_t_max + w[3] * d_att_t_mean + d_att_co
    t_att = F.softmax(t_att_sum, 1)
    d_att = F.softmax(d_att_sum, 1)

    return t_att, d_att



if __name__ == '__main__':
    w = torch.Tensor(numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                     [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1],
                     [1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1],
                     [1, 1, 1, 1]
                    ]))

    ph_mat = torch.randn((6, 10, 52))
    textual_mat = torch.randn((6, 10, 50))
    deepwalk_mat = torch.randn((6, 10, 50))
    W = Variable(torch.randn(8, 1, 10))
    U = Variable(torch.randn(8, 50, 10))
    V = Variable(torch.randn(8, 10, 1))
    S = Variable(torch.randn(50, 50))
    wf = open("att.txt", "w")
    t_att, d_att = multiview_attentive_representation(w[0], ph_mat, textual_mat, deepwalk_mat, W, U, V, S, wf)
    wf.close()
