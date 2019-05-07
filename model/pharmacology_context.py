# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: pharmacology_context.py
@time: 19-1-24 下午8:50
@description: 药物cnn降维至500
              将药物描述语句经过lstm, 构建context matrix
              拼接target_seq
              放到cnn分类器中处理
"""
import sys
sys.path.extend(['/home/kqyuan/DDISuccess', '/home/kqyuan/DDISuccess/util',
                 '/home/lei/DDISuccess', '/home/lei/DDISuccess/util'])
import torch
from sklearn import metrics

from torch import nn
from torch.utils import data

from model.pharmacology_cnn import BaseModel
from util.data_loader import PharmacologyDescriptionDataset
from util.neural_network import Conv2dReduceDims, BilstmDescription, Lenet5Classifier, GruHierarchy
from util.utility import hierarchy_rows_padding, hierarchy_path_align


class PharmacologyContextModel(BaseModel):
    def __init__(self, dataset, train_data_file, model_save_path='pharmacology_context_model.ckpt'):
        super().__init__(dataset, train_data_file, model_save_path)
        # self.NUM_EPOCHS = 1
        print("self.NUM_EPOCHS: ", self.NUM_EPOCHS)

    def train(self):
        pharmacology_feature_dim = self.train_dataset.get_pharmacologicy_feature_dim()
        target_seq_len = self.train_dataset.get_target_seq_len()
        description_wordembedding_dim = self.train_dataset.get_description_wordembedding_dim()
        print("pharmacology_feature_dim: ", pharmacology_feature_dim)
        print("target_seq_len ", target_seq_len)
        print("description_wordembedding_dim: ", description_wordembedding_dim)

        self.pharmacology_cnn = Conv2dReduceDims(pharmacology_feature_dim)
        self.pharmacology_cnn.double()
        self.pharmacology_cnn.to(self.device)

        self.description_lstm = BilstmDescription(description_wordembedding_dim, self.device)
        self.description_lstm.double()
        self.description_lstm.to(self.device)

        self.hierarchy_gru = GruHierarchy(self.description_lstm.get_out_dims(), self.device)
        self.hierarchy_gru.double()
        self.hierarchy_gru.to(self.device)

        classifier_in_channels = 500 + target_seq_len + 10*self.hierarchy_gru.get_out_dims()  # the output of cnn for reduce dims is self.description_lstm.get_out_dims()
        print("classifier_in_channels: ", classifier_in_channels)
        self.classifier = Lenet5Classifier(feature_dim=classifier_in_channels).to(self.device)
        self.classifier.double()
        self.classifier.to(self.device)

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
                phy_features = phy_features.double().to(self.device)
                phy_features = self.pharmacology_cnn(phy_features)

                this_batch_size = len(labels)
                hierarchy_description = self.train_dataset.feed_batch_data(i, self.BATCH_SIZE)
                hierarchy_description = self.process_hierarchy_description(hierarchy_description).double().to(self.device).reshape((this_batch_size, -1))  # torch.Size([100, 1, 28, 28])
                target_seqs = target_seqs.double().to(self.device).reshape((this_batch_size, -1))  # torch.Size([100, 1, 28, 28])
                labels = labels.to(self.device)  # torch.Size([100])

                # Forward pass
                # print("phy_features.size()", phy_features.size())
                # print("target_seqs.size()", target_seqs.size())
                # print("hierarchy_description.size()", hierarchy_description.size())
                inputs = torch.cat((phy_features, target_seqs, hierarchy_description), 1).reshape((this_batch_size, 1, -1, 1))
                # print("inputs:", inputs.size())
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
                                      shuffle=True)

        with torch.no_grad():
            correct = 0
            total = 0
            y_true = None
            y_pred = None
            pred_score = None
            for i, (phy_features, target_seqs, labels) in enumerate(test_loader):
                phy_features = phy_features.double().to(self.device)
                phy_features = self.pharmacology_cnn(phy_features)

                this_batch_size = len(labels)
                hierarchy_description = test_dataset.feed_batch_data(i, self.BATCH_SIZE)
                hierarchy_description = self.process_hierarchy_description(hierarchy_description).reshape((this_batch_size, -1)).double().to(self.device)  # torch.Size([100, 1, 28, 28])
                target_seqs = target_seqs.double().to(self.device).reshape((this_batch_size, -1))  # torch.Size([100, 1, 28, 28])
                labels = labels.to(self.device)  # torch.Size([100])

                # Forward pass
                inputs = torch.cat((phy_features, target_seqs, hierarchy_description), 1).reshape(
                    (this_batch_size, 1, -1, 1))
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

    def process_hierarchy_description(self, hierarchy_descriptions):
        tmp_outs = []
        for descriptions in hierarchy_descriptions:
            out = self.description_lstm(descriptions) # path_len * self.description_lstm.get_out_dims()
            # print("description lstm out: ", out.size(), type(out), torch.is_tensor(out))
            tmp_outs.append(out.cpu().detach().numpy())

        batch_gru_outs, x_len = self.hierarchy_gru(tmp_outs) # tmp_outs: [batch_size, 10, num_directions*hidder_size]
        # print("x_len: ", type(x_len), x_len)
        batch_gru_outs = hierarchy_path_align(batch_gru_outs, x_len) # batch内对齐（每个batch内以最长的路径为准）

        # 整个数据集内对齐
        batch_outs = []
        batch_outs_numpy = batch_gru_outs.cpu().detach().numpy()
        for outs in batch_outs_numpy:
            batch_outs.append(hierarchy_rows_padding(10, self.hierarchy_gru.get_out_dims(), outs))

        return torch.tensor(batch_outs)


if __name__ == '__main__':
    # model = PharmacologyContextModel(PharmacologyDescriptionDataset, "../Data/DTI/e_train.pickle", "e_pharmacology_context_model.ckpt")
    # model.train()
    # model.classifier.eval()
    # model.test("../Data/DTI/e_test.pickle")

    # model = PharmacologyContextModel(PharmacologyDescriptionDataset, "../Data/DTI/ic_train.pickle", "ic_pharmacology_context_model.ckpt")
    # model.train()
    # model.classifier.eval()
    # model.test("../Data/DTI/ic_test.pickle")
    #
    model = PharmacologyContextModel(PharmacologyDescriptionDataset, "../Data/DTI/gpcr_train.pickle",
                                     "gpcr_pharmacology_context_model.ckpt")
    model.train()
    model.classifier.eval()
    model.test("../Data/DTI/gpcr_test.pickle")

