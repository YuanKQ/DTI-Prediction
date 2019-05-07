# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: pharmacology_cnn_cpu.py
@time: 19-1-1 上午10:09
@description:
"""
import sys

import sklearn.metrics as metrics

sys.path.extend(['/run/media/yuan/data/PycharmProjects/DDISuccess', '/run/media/yuan/data/PycharmProjects/DDISuccess/util'])


import torch

import torch.nn as nn
import torch.utils.data as data
from util.data_loader import PharmacologyDataset

BATCH_SIZE = 16
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 0.001
NUM_EPOCHS = 800


class Lenet5Classifier(nn.Module):

    def __init__(self, feature_dim, in_channels=1):
        super(Lenet5Classifier, self).__init__()
        self.num_classes = 2
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=5, stride=1, padding=2), # Conv1d, https://pytorch.org/docs/stable/nn.html#conv1d
            nn.BatchNorm2d(4),  # BatchNorm1d(channel_out), https://pytorch.org/docs/stable/nn.html#batchnorm1d
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # Maxpool1d, https://pytorch.org/docs/stable/nn.html#maxpool1d
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(int(feature_dim/4)*1*4, 2) # https://pytorch.org/docs/stable/nn.html#linear

    def forward(self, x):
        in_tensor = x.type(torch.DoubleTensor)
        # print(in_tensor.type)
        out = self.layer1(in_tensor)
        # print("layer1: ", out.shape)
        out = self.layer2(out)
        # print("layer2: ", out.shape)
        out = out.reshape((out.size(0), -1))  # [100, 7, 7, 32] -> [100, 1568])
        out = self.fc(out)
        return out


def cat_input_data(phy_features, target_seqs):

    tensor_cat = torch.cat((phy_features, target_seqs), 2)
    tensor_cat.shape
    return tensor_cat


def calculate_accuracy(outputs, labels, total, correct):
    # print(outputs.shape)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    return total, correct


def eval_model(outputs, labels, total, correct, y_true, y_pred, pred_score):
    softmax = torch.nn.Softmax(1)
    score, predicted = torch.max(softmax(outputs), 1)
    # print("output shape: ", outputs.shape)
    # print("softmax(outputs): ", softmax(outputs).shape)
    # print("score: ", score)
    # print("predicted: ", predicted)
    y_true = torch.cat((y_true.long(), labels), 0)
    correct += (predicted == labels).sum().item()
    total += labels.size(0)
    pred_score = torch.cat((pred_score.double(), score), 0)
    y_pred = torch.cat((y_pred.long(), predicted), 0)

    # precision, recall, f1, _ = metrics.precision_recall_fscore_support(labels, predicted)

    auroc = metrics.roc_auc_score(labels, score) # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score

    # print("accuracy: ", accuracy)
    # print("precision: ", precision)
    # print("recall: ", recall)
    # print("auroc: ", auroc)
    # return accuracy, precision, recall, auroc
    return y_true, total, correct, y_pred, pred_score

def train_and_test(train_data_filepath, test_data_filepath):
    train_dataset = PharmacologyDataset(train_data_filepath)
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True)
    test_dataset = PharmacologyDataset(test_data_filepath)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)

    feature_dim = train_dataset.get_feature_dim()
    print("feature_dim: ", feature_dim)
    model = Lenet5Classifier(feature_dim=feature_dim).to(DEVICE)
    model.double()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # model.parameters(). Returns an iterator over module parameters. This is typically passed to an optimizer.
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(NUM_EPOCHS):
        correct = 0
        total = 0
        for i, (phy_features, target_seqs, labels) in enumerate(train_loader):
            phy_features = phy_features.to(DEVICE) # torch.Size([100, 1, 28, 28])
            target_seqs = target_seqs.to(DEVICE) # torch.Size([100, 1, 28, 28])

            labels = labels.to(DEVICE) # torch.Size([100])

            # Forward pass
            inputs = cat_input_data(phy_features, target_seqs)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            # 根据pytorch中的backward()函数的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉；
            # 但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad 了
            optimizer.zero_grad()  # Sets gradients of all model parameters to zero.

            loss.backward()
            optimizer.step()

            total, correct = calculate_accuracy(outputs, labels, total, correct)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item(), correct/total*100))

    # Test the model
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        y_true = torch.tensor([])
        y_pred = torch.tensor([])
        pred_score = torch.tensor([])
        for (phy_features, target_seqs, labels) in test_loader:
            phy_features = phy_features.to(DEVICE)  # torch.Size([100, 1, 28, 28])
            target_seqs = target_seqs.to(DEVICE)  # torch.Size([100, 1, 28, 28])

            labels = labels.to(DEVICE)  # torch.Size([100])

            # Forward pass
            inputs = cat_input_data(phy_features, target_seqs)
            outputs = model(inputs)
            # print("inputs:", inputs.shape, " outputs: ", outputs.shape, "   labels: ", labels.size)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()\
            y_true, total, correct, y_pred, pred_score = eval_model(outputs, labels, total, correct, y_true, y_pred, pred_score)

        # print("y_true: ", y_true.shape, y_true)
        # print("y_pred:", y_pred.shape, y_pred)
        print("************test**************")
        accuracy = correct / total
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred)
        auroc = metrics.roc_auc_score(y_true, pred_score)  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
        print("accuracy: ", accuracy)
        print("precision: ", precision)
        print("recall: ", recall)
        print("auroc: ", auroc)
        # print('Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, AUROC: {:.2f}%'.format(accuracy*100, precision*100, recall*100, auroc*100))

    # # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')

if __name__ == '__main__':
    train_and_test("../Data/DTI/e_train.pickle", "../Data/DTI/e_test.pickle")
