# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: drug_feature_detect.py
@time: 18-12-24 下午9:28
@description:
    1.检验特征矩阵中是否有全为０的药物？   None
    2.检验特征矩阵中是否有全为０的特征？   None
"""
import glob
import pickle

import numpy as np


def main_process(filename):
    with open(filename, "rb") as rf:
        matrix = pickle.load(rf)
    if matrix is None or len(matrix) == 0:
        return
    drug_count = len(matrix)
    feature_dimension = 0

    drug_feature_is_empty_count = 0
    drug_feature_is_empty_set = {}

    for key in matrix.keys():
        if feature_dimension == 0:
            feature_dimension = len(matrix[key])

        if sum(matrix[key]) == 0:
            drug_feature_is_empty_count += 1
            drug_feature_is_empty_set.add(key)
    print("{} rows: {}, cols: {}".format(filename, drug_count, feature_dimension))
    print("the number of drugs whose feature matrix is zero: {}".format(drug_feature_is_empty_count))
    print("the drugs whose feature matrix is zero: {}".format(drug_feature_is_empty_set))

    feature_matrix = np.zeros([1, feature_dimension])
    for key in matrix.keys():
        feature_matrix += matrix[key]
    feature_is_zero_count = np.sum(feature_matrix == 0)
    print("number of feature whom there is no drug have: {}".format(feature_is_zero_count))
    print("----------------------------/n/n")


if __name__ == '__main__':
    feature_files_list = glob.glob("./drug_*_matrix_dict.pickle")
    print(feature_files_list)

    for filename in feature_files_list:
        main_process(filename)
