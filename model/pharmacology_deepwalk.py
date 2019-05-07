# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: pharmacology_deepwalk.py
@time: 19-1-23 上午9:26
@description:
    药物cnn降维至500
    拼接hierarchy deepwalk matrix和target_seq
    放到cnn分类器中处理
"""
import sys
sys.path.extend(['/home/kqyuan/DDISuccess', '/home/kqyuan/DDISuccess/util',
                 '/home/lei/DDISuccess', '/home/lei/DDISuccess/util'])
from util.neural_network import Lenet5Classifier, Conv2dReduceDims
from model.pharmacology_cnn import BaseModel
from util.data_loader import PharmacologyDeepwalkPadDataset
import sklearn.metrics as metrics
import torch.nn as nn
import torch


class PharmacologyDeepwalkModel(BaseModel):
    def __init__(self, dataset, train_data_file, test_data_file):
        super().__init__(dataset, train_data_file, test_data_file)

    def train_and_test(self):
        pharmacology_feature_dim = self.train_dataset.get_pharmacologicy_feature_dim()
        target_seq_len = self.train_dataset.get_target_seq_len()
        hierarchy_feature_dims = self.train_dataset.get_hierarchy_feature_dims()
        print("pharmacology_feature_dim: ", pharmacology_feature_dim)
        print("target_seq_len ", target_seq_len)

        pharmacology_cnn = Conv2dReduceDims(pharmacology_feature_dim)
        pharmacology_cnn.double()
        pharmacology_cnn.to(self.DEVICE)
        target_cnn = Conv2dReduceDims(target_seq_len)
        target_cnn.double()
        target_cnn.to(self.DEVICE)
        # classifier_in_channels = 500*2 + hierarchy_feature_dims # the output of cnn for reduce dims is 500
        classifier_in_channels = 500 + target_seq_len + hierarchy_feature_dims # the output of cnn for reduce dims is 500
        classifier = Lenet5Classifier(feature_dim=classifier_in_channels).to(self.DEVICE)
        classifier.double()
        classifier.to(self.DEVICE)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        # classifier.parameters(). Returns an iterator over module parameters. This is typically passed to an optimizer.
        optimizer = torch.optim.SGD(classifier.parameters(), lr=self.LEARNING_RATE)

        # Train the classifier
        total_step = len(self.train_loader)
        for epoch in range(self.NUM_EPOCHS):
            correct = 0
            total = 0
            for i, (phy_features, drug_deepwalk, target_seqs, labels) in enumerate(self.train_loader):
                batch_size = len(labels)
                phy_features = phy_features.double().to(self.DEVICE)
                drug_deepwalk = drug_deepwalk.double().to(self.DEVICE).reshape((batch_size, -1))
                target_seqs = target_seqs.double().to(self.DEVICE).reshape((batch_size, -1))

                # reduce dim
                phy_features = pharmacology_cnn(phy_features)
                # target_seqs = target_cnn(target_seqs)

                labels = labels.to(self.DEVICE)  # torch.Size([100])

                # Forward pass
                # print("phy_features: ", phy_features.shape)
                # print("target_seqs: ", target_seqs.shape)
                # print("drug_deepwalk: ", drug_deepwalk.shape)
                # print("batch size: ", batch_size)
                inputs = torch.cat((phy_features, target_seqs, drug_deepwalk), 1).reshape((batch_size, 1, -1, 1))
                # print("inputs: ", inputs.shape)
                outputs = classifier(inputs)
                loss = criterion(outputs, labels)

                # Backward and optimize
                # 根据pytorch中的backward()函数的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉；
                # 但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad 了
                optimizer.zero_grad()  # Sets gradients of all classifier parameters to zero.

                loss.backward()
                optimizer.step()

                total, correct = self.calculate_accuracy(outputs, labels, total, correct)

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                          .format(epoch + 1, self.NUM_EPOCHS, i + 1, total_step, loss.item(), correct / total * 100))

        # Test the classifier
        classifier.eval()

        with torch.no_grad():
            correct = 0
            total = 0
            y_true = None
            y_pred = None
            pred_score = None
            for (phy_features, drug_deepwalk, target_seqs, labels) in self.test_loader:
                batch_size = len(labels)
                phy_features = phy_features.double().to(self.DEVICE)
                drug_deepwalk = drug_deepwalk.double().to(self.DEVICE).reshape((batch_size, -1))
                target_seqs = target_seqs.double().to(self.DEVICE).reshape((batch_size, -1))

                # reduce dim
                phy_features = pharmacology_cnn(phy_features)
                # target_seqs = target_cnn(target_seqs)

                labels = labels.to(self.DEVICE)  # torch.Size([100])

                # Forward pass
                # print("phy_features: ", phy_features.shape)
                # print("target_seqs: ", target_seqs.shape)
                # print("drug_deepwalk: ", drug_deepwalk.shape)
                # print("batch size: ", batch_size)
                inputs = torch.cat((phy_features, target_seqs, drug_deepwalk), 1).reshape((batch_size, 1, -1, 1))
                # print("inputs: ", inputs.shape)
                outputs = classifier(inputs)

                y_true, total, correct, y_pred, pred_score = self.eval_model(outputs, labels, total, correct, y_true,
                                                                             y_pred, pred_score)

            print("************test**************")
            accuracy = correct / total
            precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred)
            auroc = metrics.roc_auc_score(y_true, pred_score)  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
            aupr_precision, aupr_recall, _ = metrics.precision_recall_curve(y_true, pred_score, pos_label=1)
            aupr = metrics.auc(aupr_recall, aupr_precision)
            print("auroc: ", auroc)
            print("accuracy: ", accuracy)
            print("precision: ", precision)
            print("recall: ", recall)
            print("f1: ", f1)
            print("aupr: ", aupr)

        # # Save the classifier checkpoint
        torch.save(classifier.state_dict(), 'pharmacology_cnn_cuda_model.ckpt')

if __name__ == '__main__':
    PharmacologyDeepwalkModel(PharmacologyDeepwalkPadDataset, "../Data/DTI/e_train.pickle", "../Data/DTI/e_test.pickle").train_and_test()
