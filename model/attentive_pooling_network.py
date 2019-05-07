# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: attentive_pooling_network.py
@time: 2019/2/26 15:57
@description: hierarchy_network_embedding与hierarchy_textual_embedding进行矩阵变换，
              取col-max, row-max 作attention
"""
import sys


sys.path.extend(['/home/kqyuan/DDISuccess', '/home/kqyuan/DDISuccess/util',
                 '/home/lei/DDISuccess', '/home/lei/DDISuccess/util'])
from util.single_attention_layer import max_pooling_attention
import torch
from sklearn import metrics

from torch import nn
from torch.utils import data

from model.pharmacology_cnn import BaseModel
from util.data_loader import PharmacologyDescriptionDeepwalkDataset
from util.neural_network import Conv2dReduceDims, BilstmDescription, Lenet5Classifier, GruHierarchy,\
    process_hierarchy_description, process_hierarchy_deepwalk
from torch.autograd import Variable


class AttentivePoolingNetworkModel(BaseModel):
    def __init__(self, dataset, train_data_file, model_save_path='attentive_network_pooling.ckpt'):
        super().__init__(dataset, train_data_file, model_save_path)
        print("self.BATCH_SIZE: ", self.BATCH_SIZE)
        print("self.NUM_EPOCHS: ", self.NUM_EPOCHS)
        self.device_2 = torch.device('cuda:2')

    def train(self):
        pharmacology_feature_dim = self.train_dataset.get_pharmacologicy_feature_dim()
        target_seq_len = self.train_dataset.get_target_seq_len()
        description_wordembedding_dim = self.train_dataset.get_description_wordembedding_dim()
        deepwalk_feature_dim = self.train_dataset.get_deepwalk_feature_dim()
        print("pharmacology_feature_dim: ", pharmacology_feature_dim)
        print("target_seq_len ", target_seq_len)
        print("description_wordembedding_dim: ", description_wordembedding_dim)
        print("hierarchy_feature_dim: ", deepwalk_feature_dim)

        self.pharmacology_cnn = Conv2dReduceDims(pharmacology_feature_dim, 1, 500, self.device)
        self.pharmacology_cnn.double()
        self.pharmacology_cnn.to(self.device)

        self.description_lstm = BilstmDescription(description_wordembedding_dim, self.device)
        self.description_lstm.double()
        self.description_lstm.to(self.device)

        self.hierarchy_gru_description = GruHierarchy(self.description_lstm.get_out_dims(), 25, self.device)
        self.hierarchy_gru_description.double()
        self.hierarchy_gru_description.to(self.device)

        self.hierarchy_gru_deepwalk = GruHierarchy(deepwalk_feature_dim, 25, self.device)
        self.hierarchy_gru_deepwalk.double()
        self.hierarchy_gru_deepwalk.to(self.device)
        # self.hierarchy_gru_deepwalk.to(self.device_2)

        drug_vec_dim = self.pharmacology_cnn.get_out_dims() + 10 * self.hierarchy_gru_description.get_out_dims() + 10 * self.hierarchy_gru_deepwalk.get_out_dims()
        self.target_cnn = Conv2dReduceDims(target_seq_len, self.train_dataset.get_target_seq_width(), drug_vec_dim, self.device_2)
        self.target_cnn.double()
        self.target_cnn.to(self.device_2)

        classifier_in_channels = drug_vec_dim * 2 + 10 * self.hierarchy_gru_description.get_out_dims() + 10 * self.hierarchy_gru_deepwalk.get_out_dims()
        print("classifier feature dim: ", classifier_in_channels)
        self.classifier = Lenet5Classifier(feature_dim=classifier_in_channels).to(self.device)
        self.classifier.double()
        self.classifier.to(self.device)

        # attention variable
        self.weight = Variable(torch.randn(self.hierarchy_gru_description.get_out_dims(), self.hierarchy_gru_deepwalk.get_out_dims())).double().to(self.device)  # 2 for bidirection

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        # classifier.parameters(). Returns an iterator over module parameters. This is typically passed to an optimizer.
        optimizer = torch.optim.SGD(self.classifier.parameters(), lr=self.LEARNING_RATE)

        # Train the classifier
        total_step = len(self.train_loader)
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

                # attentive_network_pooling
                att_hierarchy_descriptions, att_hierarchy_deepwalks, _, __ = max_pooling_attention(
                    hierarchy_description, drug_deepwalk, self.weight, self.device)
                # print("att_hierarchy_descriptions: ", att_hierarchy_descriptions.size(), "att_hierarchy_deepwalks: ",
                #       att_hierarchy_deepwalks.size())
                att_hierarchy_deepwalks = att_hierarchy_deepwalks.to(self.device).reshape((this_batch_size, -1))
                att_hierarchy_descriptions = att_hierarchy_descriptions.to(self.device).reshape((this_batch_size, -1))
                # print("reshape: att_hierarchy_descriptions: ", att_hierarchy_descriptions.size(), "att_hierarchy_deepwalks: ",
                #       att_hierarchy_deepwalks.size())
                inputs = torch.cat((phy_features, target_seqs,
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
            for i, (phy_features, target_seqs, labels) in enumerate(test_loader):
                hierarchy_description, drug_deepwalk = test_dataset.feed_batch_data(i, self.BATCH_SIZE)
                phy_features = phy_features.double().to(self.device)
                phy_features = self.pharmacology_cnn(phy_features)
                this_batch_size = len(labels)
                hierarchy_description = process_hierarchy_description(self.description_lstm,
                                                                      self.hierarchy_gru_description,
                                                                      hierarchy_description)
                hierarchy_description = hierarchy_description.double().to(self.device)
                drug_deepwalk = process_hierarchy_deepwalk(self.hierarchy_gru_deepwalk, drug_deepwalk).double().to(self.device)
                target_seqs = self.target_cnn(target_seqs).to(self.device)
                labels = labels.to(self.device)  # torch.Size([100])

                # attentive_network_pooling
                att_hierarchy_descriptions, att_hierarchy_deepwalks, _, __ = max_pooling_attention(hierarchy_description, drug_deepwalk, self.weight, self.device)
                # print("att_hierarchy_descriptions: ", att_hierarchy_descriptions.size(), "att_hierarchy_deepwalks: ", att_hierarchy_deepwalks.size())
                att_hierarchy_deepwalks = att_hierarchy_deepwalks.to(self.device).reshape((this_batch_size, -1))
                att_hierarchy_descriptions = att_hierarchy_descriptions.to(self.device).reshape((this_batch_size, -1))
                # Forward pass
                # inputs = torch.cat((phy_features, target_seqs, hierarchy_description, drug_deepwalk), 1).reshape(
                #     (this_batch_size, 1, -1, 1))
                inputs = torch.cat((phy_features, target_seqs, hierarchy_description.reshape((this_batch_size, -1)), drug_deepwalk.reshape((this_batch_size, -1)),
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

        # # Save the classifier checkpoint
        torch.save(self.classifier.state_dict(), self.model_save_path)

if __name__ == '__main__':
    model = AttentivePoolingNetworkModel(PharmacologyDescriptionDeepwalkDataset, "../Data/DTI/e_train.pickle", "e_attentive_pooling_network_model.ckpt")
    model.train()
    model.classifier.eval()
    model.test("../Data/DTI/e_test.pickle")

    # model = AttentivePoolingNetworkModel(PharmacologyDescriptionDeepwalkDataset, "../Data/DTI/ic_train.pickle", "ic_attentive_pooling_network_model.ckpt")
    # model.train()
    # model.classifier.eval()
    # model.test("../Data/DTI/ic_test.pickle")

    # model = AttentivePoolingNetworkModel(PharmacologyDescriptionDeepwalkDataset, "../Data/DTI/gpcr_train.pickle", "gpcr_attentive_pooling_network_model.ckpt")
    # model.train()
    # model.classifier.eval()
    # model.test("../Data/DTI/gpcr_test.pickle")
