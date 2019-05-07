# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: neural_network.py
@time: 19-1-23 上午9:26
@description:
"""

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from util.utility import hierarchy_cols_padding, hierarchy_rows_padding, hierarchy_path_align


class Lenet5Classifier(nn.Module):

    def __init__(self, feature_dim):
        super(Lenet5Classifier, self).__init__()
        self.device = torch.device('cuda:1')
        self.num_classes = 2
        layer1_out_channels = 20
        layer2_out_channels = 20
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, layer1_out_channels, kernel_size=5, stride=1, padding=2), # Conv1d, https://pytorch.org/docs/stable/nn.html#conv1d
            nn.Dropout(0.3),
            nn.BatchNorm2d(layer1_out_channels),  # BatchNorm1d(channel_out), https://pytorch.org/docs/stable/nn.html#batchnorm1d
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # Maxpool1d, https://pytorch.org/docs/stable/nn.html#maxpool1d
        self.layer2 = nn.Sequential(
            nn.Conv2d(layer1_out_channels, layer2_out_channels, kernel_size=5, stride=1, padding=2),
            nn.Dropout(0.3),
            nn.BatchNorm2d(layer2_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(int(feature_dim/4)*1*layer2_out_channels, 2) # https://pytorch.org/docs/stable/nn.html#linear
        print("fc feature_dim: ", int(feature_dim/4)*1*layer2_out_channels)

    def forward(self, in_tensor):
        # in_tensor = x.type(torch.cuda.DoubleTensor)
        # print(in_tensor.type)
        # print("in_tensor: ", in_tensor.shape)
        # in_tensor: torch.Size([16, 1, feature_dim, 1])
        out = self.layer1(in_tensor)
        # print("layer1: ", out.shape)
        out = self.layer2(out)
        # print("layer2: ", out.shape)
        out = out.reshape((out.size(0), -1))  # [100, 7, 7, 32] -> [100, 1568])
        out = self.fc(out)
        return out


class Conv2dReduceDims(nn.Module):
    def __init__(self, H_in, W_in, fc_out_channels, device):
        super(Conv2dReduceDims, self).__init__()
        conv_out_channels = 20
        self.fc_out_channels = fc_out_channels
        self.W_in = W_in if W_in == 1 else int(W_in/2)
        self.device = device
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, conv_out_channels, kernel_size=5, stride=1, padding=2), # Conv1d, https://pytorch.org/docs/stable/nn.html#conv1d
            nn.BatchNorm2d(conv_out_channels),  # BatchNorm1d(channel_out), https://pytorch.org/docs/stable/nn.html#batchnorm1d
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(int(H_in / 2) * self.W_in * conv_out_channels, fc_out_channels)
        # print("Conv2dReduceDims fc feature_dim: ", int(H_in / 2) * self.W_in * conv_out_channels)

    def forward(self, x):
        # print("input: ", x.size())
        x = x.double().to(self.device)
        out = self.conv_layer(x)
        # print("conv_layer out: ", out.size())
        out = out.reshape((out.size(0), -1))
        # print("Conv2dReduceDims: ", out.shape)
        out = self.fc(out)
        return out

    def get_out_dims(self):
        return self.fc_out_channels


class Conv2dPharmacology(nn.Module):
    def __init__(self, feature_dim):
        super(Conv2dPharmacology, self).__init__()
        self.conv_out_channels = 10
        self.feature_dim = feature_dim
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, self.conv_out_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(self.conv_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=10, stride=5, padding=5))
        self.cnn2 = nn.Sequential(
            nn.Conv2d(self.conv_out_channels, self.conv_out_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(self.conv_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=10, stride=5, padding=5))

    def forward(self, x):
        # TODO: test the size of out
        # x = x.double()
        out = self.cnn1(x)
        # print("cnn1 out:", out.size())
        out = self.cnn2(out)
        # print("cnn2 out:", out.size())
        return out.squeeze()

    def get_flatten_out_dim(self):
        cnn1_out = int(((self.feature_dim-1)/2 + 1) / 5) + 1
        cnn2_out = int(((cnn1_out-1)/2 + 1)/5) + 1
        return cnn2_out * self.conv_out_channels


def test_Conv2dPharmacology():
    layer1_out_channels = 10
    layer2_out_channels = 10
    conv_out_channels = 10
    inputs = torch.randn((6, 1, 5080, 1))
    # layer1 = nn.Sequential(
    #     nn.Conv2d(1, conv_out_channels, kernel_size=5, stride=2, padding=2),
    #     nn.BatchNorm2d(conv_out_channels),
    #     nn.ReLU(),
    #     nn.MaxPool2d(kernel_size=10, stride=5, padding=5))# Maxpool1d, https://pytorch.org/docs/stable/nn.html#maxpool1d
    # layer2 = nn.Sequential(
    #     nn.Conv2d(conv_out_channels, conv_out_channels, kernel_size=5, stride=2, padding=2),
    #     nn.BatchNorm2d(conv_out_channels),
    #     nn.ReLU(),
    #     nn.MaxPool2d(kernel_size=10, stride=5, padding=5))
    # outputs = layer1(inputs)
    # print("layer1 outputs: ", outputs.size())
    # outputs = layer2(outputs)
    cnn = Conv2dPharmacology(5080)
    outputs = cnn(inputs)
    print("layer2 outputs: ", outputs.size(), outputs.squeeze().size(), cnn.get_out_dim()) # torch.Size([6, 10, 52, 1]) torch.Size([6, 10, 52]) 52


class BilstmDescription(nn.Module):
    def __init__(self, input_size, device, out_layers=500):
        super(BilstmDescription, self).__init__()
        self.input_size = input_size
        self.hidden_size = 128
        self.num_layers = 2
        self.device = device
        self.batch_first = True
        self.out_layers = out_layers
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size * 2, out_layers)  # 2 for bidirection

    def forward(self, x):
        x, x_len = hierarchy_cols_padding(x, self.input_size)
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)).double().to(self.device)  # 2 for bidirection
        c0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)).double().to(self.device)

        """sort"""
        x_sort_idx = np.argsort(-x_len)
        x_unsort_idx = torch.LongTensor(np.argsort(x_sort_idx))
        x_len = x_len[x_sort_idx]
        x = x[torch.LongTensor(x_sort_idx)]

        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first).to(self.device)
        """process using RNN"""
        out_pack, _ = self.lstm(x_emb_p, (h0, c0))

        """unpack: out"""
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
        out = self.fc(out[:, -1, :])
        """unsort: out"""
        out = out[x_unsort_idx]

        # """unsort: c"""
        # ct = torch.transpose(ct, 0, 1)[
        #     x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        # ct = torch.transpose(ct, 0, 1)

        # return out, (ht, ct)
        return out

    # def pad_description(self, x):
    #     """description_len长度不一，需要进行对齐"""
    #     # x.size(): (path_len, description_len, word2vec_len)
    #     # max_description_len to pad
    #     x_len = []
    #     for x_description in x:
    #         x_len.append(len(x_description))
    #
    #     max_description_len = max(x_len)
    #     new_x = []
    #     for i in range(len(x)):
    #         new_x_description = np.zeros((max_description_len, self.input_size))
    #         new_x_description[0: x_len[i]] = x[i]
    #         new_x.append(new_x_description)
    #
    #     return torch.DoubleTensor(new_x), np.array(x_len)

    def get_out_dims(self):
        return self.out_layers


class GruHierarchy(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(GruHierarchy, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 2
        self.device = device
        self.batch_first = True
        self.bidirectional = True
        self.directions = 2 if self.bidirectional else 1
        self.out_layers = 100
        self.GRU = nn.GRU(input_size,
                          self.hidden_size,
                          self.num_layers,
                          batch_first=self.batch_first,
                          bidirectional=self.bidirectional)
        # print("GRU param: ",
        #       "\nGRU.hidden_size: ", self.GRU.hidden_size,
        #       "\nGRU.bidirectional: ", self.GRU.bidirectional,
        #       "\nGRU.input_size: ", self.GRU.input_size,
        #       "\nGRU.bidirectional: ",  self.GRU.bidirectional,
        #       "\nGRU.training: ", self.GRU.training,
        #       "\nGRU.batch_first: ", self.GRU.batch_first,
        #       "\nGRU.bias: ", self.GRU.bias,
        #       "\nGRU.num_layers: ", self.num_layers)
        # print("self.device: ", self.device, "input_size: ", self.input_size)

    def forward(self, x):
        # print("GruHierarchy")
        x, x_len = hierarchy_cols_padding(x, self.input_size)
        batch_size = x.size(0)
        # print("x: ", type(x), x.size())
        h0 = Variable(torch.randn(self.num_layers * self.directions, batch_size, self.hidden_size)).double().to(self.device) # 2 for bidirection
        # print("forward: x:", x.size(), "h0:", h0.size())

        """sort"""
        x_sort_idx = np.argsort(-x_len)
        x_unsort_idx = torch.LongTensor(np.argsort(x_sort_idx))
        x_len = x_len[x_sort_idx]
        x = x[torch.LongTensor(x_sort_idx)]

        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first).to(self.device)
        """process using RNN"""
        out_pack, _ = self.GRU(x_emb_p, h0)

        """unpack: out"""
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
        # out = self.fc(out[:, -1, :])
        """unsort: out"""
        out = out[x_unsort_idx]
        return out, x_len #.permute(1, 0, 2)   # (seq_len, batch, input_size) -> (batch, seq_len, input_size)

    def get_out_dims(self):
        return self.hidden_size * self.directions


def process_hierarchy_description(description_lstm, hierarchy_gru_description, hierarchy_descriptions):
    tmp_outs = []
    # print("batch_hierarchy_description: ", len(hierarchy_descriptions))
    for descriptions in hierarchy_descriptions:
        out = description_lstm(descriptions)  # path_len * self.description_lstm.get_out_dims()
        # print("description lstm out: ", out.size(), type(out), torch.is_tensor(out))
        tmp_outs.append(out.cpu().detach().numpy())

    batch_gru_outs, x_len = hierarchy_gru_description(tmp_outs)  # tmp_outs: [batch_size, 10, num_directions*hidder_size]
    # print("x_len: ", type(x_len), x_len)
    # print("batch_gru_outs, x_len = self.hierarchy_gru(tmp_outs)  ", batch_gru_outs.size())
    batch_gru_outs = hierarchy_path_align(batch_gru_outs, x_len)  # batch内对齐（每个batch内以最长的路径为准）
    # print("hierarchy_path_align(batch_gru_outs, x_len", batch_gru_outs.size())

    # 整个数据集内对齐
    batch_outs = []
    batch_outs_numpy = batch_gru_outs.cpu().detach().numpy()
    for outs in batch_outs_numpy:
        batch_outs.append(hierarchy_rows_padding(10, hierarchy_gru_description.get_out_dims(), outs))

    return torch.Tensor(batch_outs)


def process_hierarchy_deepwalk(hierarchy_gru_deepwalk, drug_deepwalks):
    batch_gru_outs, x_len = hierarchy_gru_deepwalk(drug_deepwalks)
    batch_gru_outs = hierarchy_path_align(batch_gru_outs, x_len)  # batch内对齐（每个batch内以最长的路径为准）

    # 整个数据集内对齐
    batch_outs = []
    batch_outs_numpy = batch_gru_outs.cpu().detach().numpy()
    for outs in batch_outs_numpy:
        batch_outs.append(hierarchy_rows_padding(10, hierarchy_gru_deepwalk.get_out_dims(), outs))

    return torch.Tensor(batch_outs)


def pad_description_test(x):
    tt = []
    tt.append(np.arange(4).reshape((1, 4)))
    tt.append(np.arange(8).reshape((2, 4)))
    tt.append(np.arange(12).reshape((3, 4)))
    tt.append(np.arange(16).reshape((4, 4)))
    tt.append(np.arange(20).reshape((5, 4)))

    def func_test(x):
        x_len = []
        input_size = len(x[0][0])
        for x_description in x:
            x_len.append(len(x_description))

        max_description_len = max(x_len)
        new_x = []
        for i in range(len(x)):
            new_x_description = np.zeros((max_description_len, input_size))
            new_x_description[0: x_len[i]] = x[i]
            new_x.append(new_x_description)

        return new_x, x_len

    print(func_test(tt))


def dynamic_lstm_test():
    """pytorch中的动态lstm测试样例"""
    sent_1 = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [0, 0, 0, 0]]
    sent_2 = [[10, 11, 12, 13], [12, 13, 14, 15], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    sent_3 = [[30, 31, 32, 34], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    sent_4 = [[40, 41, 42, 44], [41, 42, 43, 44], [42, 43, 44, 45], [43, 44, 45, 46], [44, 45, 46, 47]]
    sent_5 = [[50, 51, 52, 53], [51, 52, 53, 54], [52, 53, 54, 55], [0, 0, 0, 0], [0, 0, 0, 0]]
    text = np.zeros((5, 5, 4))
    text[0] = sent_1
    text[1] = sent_2
    text[2] = sent_3
    text[3] = sent_4
    text[4] = sent_5
    text = torch.DoubleTensor(text)
    sent_len = np.array([4, 2, 1, 5, 3])
    print("text.shape: ", text.shape)
    def func_test(x, x_len):
        x_sort_idx = np.argsort(-x_len)
        x_unsort_idx = torch.LongTensor(np.argsort(x_sort_idx))
        x_len = x_len[x_sort_idx]
        x = x[torch.LongTensor(x_sort_idx)]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True)

        rnn = nn.GRU(4, 3, 2, batch_first=True).double()
        # rnn = nn.LSTM(4, 3, 2, batch_first=True).double()
        h0 = Variable(torch.randn(2, 5, 3)).double()
        c0 = Variable(torch.randn(2, 5, 3)).double()

        out, _ = rnn(x_emb_p, h0)
        # out, _ = rnn(x_emb_p, (h0, c0))

        # unpack
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        print("pad_packed_sequence: ", unpacked.size())
        print("unpacked[0]: ", unpacked[0].size())
        print(unpacked)

        fc = nn.Linear(3, 5).double()
        unpacked = unpacked[:, -1, :]
        print("out[:, -1, :]: ", unpacked.size())
        fc_out = fc(unpacked)
        print("fc_out: ", fc_out.size())
        print(fc_out)

    func_test(text, sent_len)


if __name__ == '__main__':
    dynamic_lstm_test()
