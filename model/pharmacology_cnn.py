# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: pharmacology_cnn.py
@time: 19-1-1 上午10:09
@description:
"""
import sys
sys.path.extend(['/home/kqyuan/DDISuccess', '/home/kqyuan/DDISuccess/util',
                 '/home/lei/DDISuccess', '/home/lei/DDISuccess/util'])
from util.neural_network import Lenet5Classifier
import sklearn.metrics as metrics
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from util.data_loader import PharmacologyDataset

class BaseModel:
    BATCH_SIZE = 3
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 1

    def __init__(self, dataset, train_data_file, model_save_path='pharmacology_cnn_cuda_model.ckpt'):
        self.dataset_func = dataset
        self.train_data_file = train_data_file
        self.model_save_path = model_save_path

        self.device = torch.device('cuda:1')
        self.train_dataset = self.dataset_func(train_data_file)
        self.train_loader = data.DataLoader(dataset=self.train_dataset,
                                            batch_size=self.BATCH_SIZE,
                                            shuffle=True)

    def cat_input_data(self, phy_features, target_seqs):
        tensor_cat = torch.cat((phy_features, target_seqs), 2)
        return tensor_cat

    def calculate_accuracy(self, outputs, labels, total, correct):
        # print(outputs.shape)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        return total, correct

    def eval_model(self, outputs, labels, total, correct, y_true, y_pred, y_score):
        softmax_model = torch.nn.Softmax(1)
        softmax_out = softmax_model(outputs)
        _, predicted = torch.max(softmax_out, 1)
        batch_score = softmax_out.cpu().numpy()[:, 1].reshape(-1, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if y_true is None:
            y_true = labels.cpu().numpy()
            y_pred = predicted.cpu().numpy()
            y_score = batch_score
        else:
            y_true = np.concatenate((y_true, labels.cpu().numpy()), axis=0)
            y_pred = np.concatenate((y_pred, predicted.cpu().numpy()), axis=0)
            y_score = np.concatenate((y_score, batch_score), axis=0)
        # print("batch_score: ", batch_score.shape)
        # print("y_true: ", y_true.shape)
        # print("y_pred: ", y_pred.shape)
        # print("y_score: ", y_score.shape)

        return y_true, total, correct, y_pred, y_score

    def train(self):
        feature_dim = self.train_dataset.get_feature_dim()
        print("feature_dim: ", feature_dim)
        self.classifier = Lenet5Classifier(feature_dim=feature_dim).to(self.device)
        self.classifier.double()

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
                phy_features = phy_features.double().to(self.device) # torch.Size([100, 1, 28, 28])
                target_seqs = target_seqs.double().to(self.device) # torch.Size([100, 1, 28, 28])

                labels = labels.to(self.device) # torch.Size([100])

                # Forward pass
                inputs = self.cat_input_data(phy_features, target_seqs)
                outputs = self.classifier(inputs)
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
                          .format(epoch + 1, self.NUM_EPOCHS, i + 1, total_step, loss.item(), correct/total*100))

        # # Test the classifier
        # self.classifier.eval()

    def test(self, test_data_file):
        test_dataset = self.dataset_func(test_data_file)
        test_loader = data.DataLoader(dataset=test_dataset,
                                      batch_size=self.BATCH_SIZE,
                                      shuffle=True)

        with torch.no_grad():
            correct = 0
            total = 0
            y_true = None
            y_pred = None
            pred_score = None
            for (phy_features, target_seqs, labels) in test_loader:
                phy_features = phy_features.double().to(self.device)  # torch.Size([100, 1, 28, 28])
                target_seqs = target_seqs.double().to(self.device)  # torch.Size([100, 1, 28, 28])

                labels = labels.to(self.device)  # torch.Size([100])

                # Forward pass
                inputs = self.cat_input_data(phy_features, target_seqs)
                outputs = self.classifier(inputs)
                y_true, total, correct, y_pred, pred_score = self.eval_model(outputs, labels, total, correct, y_true, y_pred, pred_score)

            print("************test**************")
            accuracy = correct / total
            precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred)
            auroc = metrics.roc_auc_score(y_true, pred_score)  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
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
    model = BaseModel(PharmacologyDataset, "../Data/DTI/dti_train.pickle")
    model.train()
    model.classifier.eval()
    model.test("../Data/DTI/e_test.pickle")
    model.test("../Data/DTI/ic_test.pickle")
    model.test("../Data/DTI/gpcr_test.pickle")
