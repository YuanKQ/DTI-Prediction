# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: data_loader.py
@time: 19-1-1 上午10:32
@description:
"""
import math
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import random
import torch.utils.data as data

from util.utility import hierarchy_rows_padding

SEED = 666
random.seed(SEED)

class PharmacologyDataset(Dataset):
    def __init__(self, file):
        self.transforms = transforms.Compose([transforms.ToTensor()])

        with open(file, "rb") as rf:
            self.label_list = pickle.load(rf)
            self.target_seq_list = pickle.load(rf)
            self.pharmacologicy_list = pickle.load(rf)

        self.data_size = len(self.label_list)
        self.pharmacologicy_feature_dim = len(self.pharmacologicy_list[0])
        self.target_seq_dim = len(self.target_seq_list[0])

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        # 人为地将最后一个batch补满
        if index > self.data_size:
            index = random.randint((0, self.data_size-1))

        drug_pharmacologicy = self.transforms(self.pharmacologicy_list[index].reshape((-1, 1, 1))) #  (pharmacologicy_feature_dim, 1) -reshape-> (pharmacologicy_feature_dim, 1, 1) -transforms->(1, pharmacologicy_feature_dim, 1) (
        target_seq = self.transforms(self.target_seq_list[index].reshape((-1, 1, 1)))
        label = self.label_list[index]

        return drug_pharmacologicy, target_seq, label

    def get_feature_dim(self):
        return self.pharmacologicy_feature_dim+self.target_seq_dim

class PharmacologyDeepwalkPadDataset(Dataset):
    """
    对药物的Networkembedding进行头尾对齐，空缺部分使用０来补全
    """
    PADDING_COL = 10 # 药物的分类层级结构path长度最大为10
    def __init__(self, file):
        self.transforms = transforms.Compose([transforms.ToTensor()])

        with open(file, "rb") as rf:
            self.label_list = pickle.load(rf)
            self.target_seq_list = pickle.load(rf)
            self.pharmacologicy_list = pickle.load(rf)
            self.deepwalk_list = pickle.load(rf)

        # self.target_seq_dim = len(self.target_seq_list[0])
        # self.pharmacologicy_feature_dim = len(self.pharmacologicy_list[0])
        self.path_len = len(self.deepwalk_list)
        self.data_size = len(self.label_list)
        self.deepwalk_feature_dim = len(self.deepwalk_list[0][0])

    def __len__(self):
        return self.data_size

    # def padding_deepwalk(self, index):
    #     deepwalk_paddings = np.zeros((self.PADDING_COL, self.deepwalk_feature_dim))
    #     path_len = len(self.deepwalk_list[index])
    #     if path_len < self.PADDING_COL:
    #         deepwalk_paddings[0:path_len-2, :] = self.deepwalk_list[index][0:path_len-2, :]
    #         deepwalk_paddings[self.PADDING_COL-2:, :] = self.deepwalk_list[index][path_len-2:, :]
    #
    #     return deepwalk_paddings

    def __getitem__(self, index):
        drug_pharmacologicy = self.transforms(self.pharmacologicy_list[index].reshape((-1, 1, 1)))
        target_seq = self.transforms(self.target_seq_list[index].reshape((-1, 1, 1)))
        # drug_deepwalk = self.padding_deepwalk(index).reshape((1, -1)) # (10, 128) -> (1, 1280)
        drug_deepwalk = hierarchy_rows_padding(self.PADDING_COL, self.deepwalk_feature_dim, self.deepwalk_list[index]).reshape((1, -1)) # (10, 128) -> (1, 1280)
        label = self.label_list[index]
        # print("dataloader deepwalk: ", drug_deepwalk.shape)
        return drug_pharmacologicy, drug_deepwalk, target_seq, label

    def get_pharmacologicy_feature_dim(self):
        return len(self.pharmacologicy_list[0])

    def get_target_seq_len(self):
        return len(self.target_seq_list[0])

    def get_hierarchy_feature_dims(self):
        return self.PADDING_COL * self.deepwalk_feature_dim


class PharmacologyDescriptionDataset(Dataset):
    def __init__(self, file):
        self.transforms = transforms.Compose([transforms.ToTensor()])

        with open(file, "rb") as rf:
            self.label_list = pickle.load(rf)
            self.target_seq_list = pickle.load(rf)
            self.pharmacologicy_list = pickle.load(rf)
            pickle.load(rf)  # deepwalk_list
            self.hierarchy_description_list = pickle.load(rf)
            print("hierarchy_description_list: ", type(self.hierarchy_description_list), type(self.hierarchy_description_list[0]))

        self.data_size = len(self.label_list)
        self.pharmacologicy_feature_dim = len(self.pharmacologicy_list[0])
        self.target_seq_dim = len(self.target_seq_list[0])
        self.description_wordembedding_dim = len(self.hierarchy_description_list[0][0][0])
        print("data file name: ", file, "data size: ", self.data_size)

    def __getitem__(self, index):
        drug_pharmacologicy = self.transforms(self.pharmacologicy_list[index].reshape((-1, 1, 1)))
        target_seq = self.transforms(self.target_seq_list[index].reshape((-1, 1, 1)))
        # hierarchy_description_list = list(self.hierarchy_description_list[index])
        label = self.label_list[index]
        # print("dataloader deepwalk: ", drug_deepwalk.shape)
        return drug_pharmacologicy,  target_seq, label

    def __len__(self):
        return self.data_size

    def get_description_wordembedding_dim(self):
        return self.description_wordembedding_dim

    def get_pharmacologicy_feature_dim(self):
        return len(self.pharmacologicy_list[0])

    def get_target_seq_len(self):
        return len(self.target_seq_list[0])

    def feed_batch_data(self, i, batch_size):
        data_size = len(self.hierarchy_description_list)
        start = i * batch_size
        end = min(start + batch_size, data_size)

        # print("start: ", start, "end: ", end)

        return self.hierarchy_description_list[start: end]


class PharmacologyDescriptionDeepwalkDataset(Dataset):
    PADDING_COL = 10

    def __init__(self, file):
        self.transforms = transforms.Compose([transforms.ToTensor()])

        with open(file, "rb") as rf:
            self.label_list = pickle.load(rf)
            self.target_seq_list = one_hot_process_target_seq(pickle.load(rf))
            self.pharmacologicy_list = pickle.load(rf)
            self.deepwalk_list = pickle.load(rf)  # deepwalk_list
            self.hierarchy_description_list = pickle.load(rf)
            # print("hierarchy_description_list: ", type(self.hierarchy_description_list), type(self.hierarchy_description_list[0]))

        self.data_size = len(self.label_list)
        self.pharmacologicy_feature_dim = len(self.pharmacologicy_list[0])
        self.target_seq_len = len(self.target_seq_list[0])
        self.target_seq_width = len(self.target_seq_list[0][0])
        self.description_wordembedding_dim = len(self.hierarchy_description_list[0][0][0])
        self.deepwalk_feature_dim = len(self.deepwalk_list[0][0])
        print("data file name: ", file, "data size: ", self.data_size)

    def __getitem__(self, index):
        drug_pharmacologicy = self.transforms(self.pharmacologicy_list[index].reshape((-1, 1, 1)))
        # target_seq = self.transforms(self.target_seq_list[index].reshape((-1, 1, 1)))
        target_seq = self.transforms(self.target_seq_list[index].reshape((-1, self.target_seq_width, 1)))
        # drug_deepwalk = hierarchy_padding(self.PADDING_COL, self.deepwalk_feature_dim, self.deepwalk_list[index]).reshape((1, -1))  # (10, 128) -> (1, 1280)
        label = self.label_list[index]
        # print("dataloader deepwalk: ", drug_deepwalk.shape)
        return drug_pharmacologicy, target_seq, label

    def __len__(self):
        return self.data_size

    def get_description_wordembedding_dim(self):
        return self.description_wordembedding_dim

    def get_pharmacologicy_feature_dim(self):
        return len(self.pharmacologicy_list[0])

    def get_target_seq_len(self):
        return len(self.target_seq_list[0])

    def feed_batch_data(self, i, batch_size):
        data_size = len(self.hierarchy_description_list)
        start = i * batch_size
        end = min(start + batch_size, data_size)
        # print("start: ", start, "end: ", end)

        return self.hierarchy_description_list[start: end], self.deepwalk_list[start: end]

    def get_deepwalk_feature_dim(self):
        return self.deepwalk_feature_dim

    def get_target_seq_width(self):
        return self.target_seq_width


def one_hot_process_target_seq(target_seqs):
    target_len = len(target_seqs[0])
    one_hot_target_seqs = []
    for seq in target_seqs:
        seq = seq.astype(int)
        one_hots = np.eye(target_len, 5)[seq]
        one_hot_target_seqs.append(one_hots[:, 1:])

    return one_hot_target_seqs


BATCH_SIZE = 8
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    test_dataset = PharmacologyDescriptionDeepwalkDataset("../Data/DTI/e_test.pickle")
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)

    for i, (phy_features, target_seqs, labels) in enumerate(test_loader):
        phy_features = phy_features.to(DEVICE)  # torch.Size([100, 1, 28, 28])
        target_seqs = target_seqs.to(DEVICE)  # torch.Size([100, 1, 28, 28])
        # drug_deepwalk = drug_deepwalk.to(DEVICE)
        hierarchy_description, drug_deepwalk = test_dataset.feed_batch_data(i, BATCH_SIZE)
        labels = labels.to(DEVICE)  # torch.Size([100])
        print(i, labels.size(), target_seqs.size(), phy_features.size(), len(drug_deepwalk),
              len(drug_deepwalk[0]),
              len(drug_deepwalk[0][0]))

        print(type(drug_deepwalk[0]),
              type(drug_deepwalk[0][0]))
