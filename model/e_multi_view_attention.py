# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: multi_view_attention.py
@time: 2019/3/1 20:35
@description: 针对enzyme数据集预测结果进行分析，将预测结果的score从高到低输出出来。
"""
import time

import numpy
import sys


sys.path.extend(['/home/kqyuan/DDISuccess', '/home/kqyuan/DDISuccess/util', '/home/kqyuan/DDISuccess/model',
                 '/home/lei/DDISuccess', '/home/lei/DDISuccess/util', '/home/lei/DDISuccess/model'])
from util.multiview_attention_layer import multiview_attentive_representation
import torch
from sklearn import metrics

from torch import nn
from torch.utils import data

from model.pharmacology_cnn import BaseModel
from util.data_loader import PharmacologyDescriptionDeepwalkDataset
from util.neural_network import Conv2dReduceDims, BilstmDescription, Lenet5Classifier, GruHierarchy, \
    process_hierarchy_description, process_hierarchy_deepwalk, Conv2dPharmacology
from torch.autograd import Variable


class MultiViewAttentionModel(BaseModel):
    def __init__(self, dataset, train_data_file, model_save_path, w, loss_file):
        super().__init__(dataset, train_data_file, model_save_path)
        print("self.BATCH_SIZE: ", self.BATCH_SIZE)
        print("self.NUM_EPOCHS: ", self.NUM_EPOCHS)
        self.device_2 = torch.device('cuda:2')
        self.w = w.double().to(self.device)
        self.loss_file = loss_file

    def train(self):
        pharmacology_feature_dim = self.train_dataset.get_pharmacologicy_feature_dim()
        target_seq_len = self.train_dataset.get_target_seq_len()
        description_wordembedding_dim = self.train_dataset.get_description_wordembedding_dim()
        deepwalk_feature_dim = self.train_dataset.get_deepwalk_feature_dim()
        print("pharmacology_feature_dim: ", pharmacology_feature_dim)
        print("target_seq_len ", target_seq_len)
        print("description_wordembedding_dim: ", description_wordembedding_dim)
        print("hierarchy_feature_dim: ", deepwalk_feature_dim)

        self.pharmacology_cnn = Conv2dPharmacology(pharmacology_feature_dim)
        self.pharmacology_cnn.double()
        self.pharmacology_cnn.to(self.device)

        self.description_lstm = BilstmDescription(description_wordembedding_dim, self.device)
        self.description_lstm.double()
        self.description_lstm.to(self.device)

        gru_hidden_size = 25
        self.hierarchy_gru_description = GruHierarchy(self.description_lstm.get_out_dims(), gru_hidden_size, self.device)
        self.hierarchy_gru_description.double()
        self.hierarchy_gru_description.to(self.device)

        self.hierarchy_gru_deepwalk = GruHierarchy(deepwalk_feature_dim, gru_hidden_size, self.device)
        self.hierarchy_gru_deepwalk.double()
        self.hierarchy_gru_deepwalk.to(self.device)
        # self.hierarchy_gru_deepwalk.to(self.device_2)

        drug_vec_dim = self.pharmacology_cnn.get_flatten_out_dim() + 10 * self.hierarchy_gru_description.get_out_dims() + 10 * self.hierarchy_gru_deepwalk.get_out_dims()
        self.target_cnn = Conv2dReduceDims(target_seq_len, self.train_dataset.get_target_seq_width(), drug_vec_dim, self.device_2)
        self.target_cnn.double()
        self.target_cnn.to(self.device_2)

        classifier_in_channels = drug_vec_dim * 2 + 10 * self.hierarchy_gru_description.get_out_dims() + 10 * self.hierarchy_gru_deepwalk.get_out_dims()
        print("classifier feature dim: ", classifier_in_channels)
        self.classifier = Lenet5Classifier(feature_dim=classifier_in_channels).to(self.device)
        self.classifier.double()
        self.classifier.to(self.device)

        # attention variable
        self.W = Variable(torch.randn(8, 1, 10)).double().to(self.device)
        self.U = Variable(torch.randn(8, self.hierarchy_gru_description.get_out_dims(), 10)).double().to(self.device)
        self.V = Variable(torch.randn(8, 10, 1)).double().to(self.device)
        self.S = Variable(torch.randn(self.hierarchy_gru_description.get_out_dims(), self.hierarchy_gru_description.get_out_dims())).double().to(self.device)


        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        # classifier.parameters(). Returns an iterator over module parameters. This is typically passed to an optimizer.
        optimizer = torch.optim.SGD(self.classifier.parameters(), lr=self.LEARNING_RATE)

        loss_wf = open(self.loss_file, "w")

        # Train the classifier
        total_step = len(self.train_loader)
        start = time.process_time()
        for epoch in range(self.NUM_EPOCHS):
            correct = 0
            total = 0
            for i, (phy_features, target_seqs, labels) in enumerate(self.train_loader):
                hierarchy_description, drug_deepwalk = self.train_dataset.feed_batch_data(i, self.BATCH_SIZE)
                phy_features = phy_features.double().to(self.device)
                phy_features = self.pharmacology_cnn(phy_features)
                this_batch_size = len(labels)  # torch.Size([100, 1, 28, 28])
                drug_deepwalk = process_hierarchy_deepwalk(self.hierarchy_gru_deepwalk, drug_deepwalk).double().to(self.device)
                hierarchy_description = process_hierarchy_description(self.description_lstm,
                                                                      self.hierarchy_gru_description,
                                                                      hierarchy_description).double().to(self.device)
                target_seqs = self.target_cnn(target_seqs).to(self.device)
                labels = labels.to(self.device)  # torch.Size([100])

                # multiview_attention
                # print("phy: ",phy_features.size())
                att_descriptions, att_deepwalks = multiview_attentive_representation(self.w, phy_features,
                                                                                     hierarchy_description,
                                                                                     drug_deepwalk,
                                                                                     self.W, self.U, self.V, self.S)
                # print("att_hierarchy_descriptions: ", att_hierarchy_descriptions.size(), "att_hierarchy_deepwalks: ",
                #       att_hierarchy_deepwalks.size())
                att_hierarchy_deepwalks = torch.mul(att_deepwalks, drug_deepwalk).to(self.device).reshape((this_batch_size, -1))
                att_hierarchy_descriptions = torch.mul(att_descriptions, hierarchy_description).to(self.device).reshape((this_batch_size, -1))
                # print("reshape: att_hierarchy_descriptions: ", att_hierarchy_descriptions.size(), "att_hierarchy_deepwalks: ",
                #       att_hierarchy_deepwalks.size())
                inputs = torch.cat((phy_features.reshape((this_batch_size, -1)),
                                    target_seqs,
                                    hierarchy_description.reshape((this_batch_size, -1)),
                                    drug_deepwalk.reshape((this_batch_size, -1)),
                                    att_hierarchy_descriptions, att_hierarchy_deepwalks), 1).reshape((this_batch_size, 1, -1, 1))
                outputs = self.classifier(inputs)
                loss = criterion(outputs, labels)

                # Backward and optimize
                # 根据pytorch中的backward()函数的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉；
                # 但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad 了
                optimizer.zero_grad()  # Sets gradients of all classifier parameters to zero.

                loss.backward()
                optimizer.step()

                total, correct = self.calculate_accuracy(outputs, labels, total, correct)

                if (i + 1) % 32 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                          .format(epoch + 1, self.NUM_EPOCHS, i + 1, total_step, loss.item(), correct / total * 100))
            loss_wf.write(str(loss.item()) + ", " + str(correct / total * 100) + "\n")

        end = time.process_time()
        loss_wf.close()
        print("%%%total time: ", (end - start)/1000000, "s %%%")
        # # Test the classifier
        # self.classifier.eval()

    def test(self, test_data_file):
        test_dataset = self.dataset_func(test_data_file)
        test_loader = data.DataLoader(dataset=test_dataset,
                                      batch_size=self.BATCH_SIZE,
                                      shuffle=False)

        with torch.no_grad():
            correct = 0
            total = 0
            y_true = None
            y_pred = None
            pred_score = None
            # print("************test**************")
            for i, (phy_features, target_seqs, labels) in enumerate(test_loader):
                hierarchy_description, drug_deepwalk = test_dataset.feed_batch_data(i, self.BATCH_SIZE)
                this_batch_size = len(labels)

                # print("before cnn phy: ", phy_features.size())
                phy_features = phy_features.double().to(self.device)
                phy_features = self.pharmacology_cnn(phy_features)
                if len(phy_features.size()) < 3:
                    dim0 = phy_features.size(0)
                    dim1 = phy_features.size(1)
                    phy_features = phy_features.reshape((this_batch_size, dim0, dim1))

                hierarchy_description = process_hierarchy_description(self.description_lstm,
                                                                      self.hierarchy_gru_description,
                                                                      hierarchy_description)
                hierarchy_description = hierarchy_description.double().to(self.device)
                drug_deepwalk = process_hierarchy_deepwalk(self.hierarchy_gru_deepwalk, drug_deepwalk).double().to(self.device)
                target_seqs = self.target_cnn(target_seqs).to(self.device)
                labels = labels.to(self.device)  # torch.Size([100])

                # multiview_attention
                # print("after cnn phy: ", phy_features.size())
                att_descriptions, att_deepwalks = multiview_attentive_representation(self.w, phy_features,
                                                                                     hierarchy_description,
                                                                                     drug_deepwalk,
                                                                                     self.W, self.U, self.V, self.S)
                # print("att_hierarchy_descriptions: ", att_hierarchy_descriptions.size(), "att_hierarchy_deepwalks: ",
                #       att_hierarchy_deepwalks.size())
                att_hierarchy_deepwalks = torch.mul(att_deepwalks, drug_deepwalk).to(self.device).reshape(
                    (this_batch_size, -1))
                att_hierarchy_descriptions = torch.mul(att_descriptions, hierarchy_description).to(self.device).reshape(
                    (this_batch_size, -1))
                # print("reshape: att_hierarchy_descriptions: ", att_hierarchy_descriptions.size(), "att_hierarchy_deepwalks: ",
                #       att_hierarchy_deepwalks.size())
                inputs = torch.cat((phy_features.reshape((this_batch_size, -1)),
                                    target_seqs,
                                    hierarchy_description.reshape((this_batch_size, -1)),
                                    drug_deepwalk.reshape((this_batch_size, -1)),
                                    att_hierarchy_descriptions, att_hierarchy_deepwalks), 1).reshape((this_batch_size, 1, -1, 1))
                outputs = self.classifier(inputs)
                y_true, total, correct, y_pred, pred_score = self.eval_model(outputs, labels, total, correct, y_true,
                                                                             y_pred, pred_score)

            print("************test**************")
            accuracy = correct / total
            precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred)
            auroc = metrics.roc_auc_score(y_true,
                                          pred_score)  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
            aupr_precision, aupr_recall, _ = metrics.precision_recall_curve(y_true, pred_score, pos_label=1)
            aupr = metrics.auc(aupr_recall, aupr_precision)

            print("test dataset: ", test_data_file)
            print("auroc: ", auroc)
            print("accuracy: ", accuracy)
            print("precision: ", precision)
            print("recall: ", recall)
            print("f1: ", f1)
            print("aupr: ", aupr)
            print("y_score: ", pred_score)

        # # Save the classifier checkpoint
        torch.save(self.classifier.state_dict(), self.model_save_path)

if __name__ == '__main__':
    index = 7
    w = torch.Tensor(numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                                  [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1],
                                  [1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1],
                                  [1, 1, 1, 1]
                                  ]))
    print("index: ", index, "weight:", w[index])
    model = MultiViewAttentionModel(PharmacologyDescriptionDeepwalkDataset, "../Data/DTI/e_train.pickle", "e_multi_view_attention_model.ckpt", w[index], "e_loss.txt")
    model.train()
    model.classifier.eval()
    print("index: ", index, "weight:", w[index])
    model.test("../Data/DTI/e_test.pickle")

    # print("index: ", index, "weight:", w[index])
    # model = MultiViewAttentionModel(PharmacologyDescriptionDeepwalkDataset, "../Data/DTI/ic_train.pickle", "ic_multi_view_attention_model.ckpt", w[index], "ic_loss.txt)
    # model.train()
    # model.classifier.eval()
    # print("index: ", index, "weight:", w[index])
    # model.test("../Data/DTI/ic_test.pickle")

    # print("index: ", index, "weight:", w[index])
    # model = MultiViewAttentionModel(PharmacologyDescriptionDeepwalkDataset, "../Data/DTI/gpcr_train.pickle", "gpcr_multi_view_attention_model.ckpt", w[index], "gpcr_loss.txt)
    # model.train()
    # model.classifier.eval()
    # print("index: ", index, "weight:", w[index])
    # model.test("../Data/DTI/gpcr_test.pickle")
