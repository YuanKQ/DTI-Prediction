# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: get_draft_data.py
@time: 19-1-3 下午8:50
@description: 获取DTI数据集
"""
import csv
import os
import pickle
import sys
from typing import Any

import urllib3
import re
import numpy as np
import pandas as pd
import pandas as pd

from Data.DrugFeature.feature_process import BuildHierarchyPaths


class GetDrugBankIDSet:
    """
    根据DTI　gpcr药物数集据获取其在Drugbank中ID, 统计DTI药物数据集与drugbank药物数据集的重合程度
    """
    def __init__(self, source_file, target_file, drugbank_id_set):
        self.source_file = source_file
        self.target_file = target_file
        self.drugbank_id_set = drugbank_id_set

    def extract_kegg_id(self):
        """
        从DTI数据集中获取药物kegg ID集合
        :param filename:
        :return: kegg ID集合kegg_id_set
        """
        self.kegg_id_set = set()
        with open(self.source_file, "r") as rf:
            lines = rf.readlines()
        for line in lines:
            ids = line.split()
            if len(ids) >= 2:
                self.kegg_id_set.add(ids[-1])


    def get_drugbank_id_from_kegg_api(self):
        """
        从通过kegg　API获取药物在drugbank中的id, 将kegg_id与drugbank_id的mapping放到target_file中
        :param kegg_id_set: DTI数据集的药物kegg ID集合
        :return: drugbank ID集合drugbank_id_set
        """
        drugbank_id_set = set()
        pattern = re.compile("(?<=DrugBank:\s)DB\d{5}")
        wf = open(self.target_file, "w")
        # wf.write("kegg_id,drugbank_id\n")

        kegg_id_no_drugbank_count = 0
        kegg_id_drugbank_count = 0
        pool_manager = urllib3.PoolManager()
        prefix = "http://rest.kegg.jp/get/"
        for kegg_id in self.kegg_id_set:
            url = prefix + kegg_id
            request = pool_manager.request("GET", url)
            response = request.data.decode()
            print("success get response from " + url)

            results = pattern.findall(response)
            if results:
                drugbank_id = results[0]
                drugbank_id_set.add(drugbank_id)
                kegg_id_drugbank_count += 1
                wf.write(kegg_id + "," + drugbank_id + "\n")
            else:
                print(kegg_id)
                kegg_id_no_drugbank_count += 1

        wf.close()

        # kegg_id_drugbank_count <  len(self.drugbank_id_set), 因为kegg与drugbank是多对一的关系
        print("kegg drug count: ", len(self.kegg_id_set))
        print("count of kegg drug in drugbank: ", kegg_id_drugbank_count)
        print("count of kegg drug not in drugbank: ", kegg_id_no_drugbank_count)
        print("count of self.drugbank_id_set aquired from kegg api: ", len(drugbank_id_set))

        return drugbank_id_set


    def get_DTI_drug_id_set_from_csv(self, filename):
        """
        从target_file(kegg_id与drugbank_id的mapping)读取DTI数据集中药物的DrugBank id.
        :param filename: 保存kegg_id与drugbank_id的mapping的csv文件
        :return: DTI数据集中药物的DrugBank id集合
        """
        DTI_drugbank_id_set = set()
        with open(filename, "r") as rf:
            lines = rf.readlines()
        print("the number of lines: ", len(lines))
        for line in lines:
            if "DB" in line:
                ids = line.split()
                if ids:
                    self.DTI_drugbank_id_set.add(ids[-1])
            else:
                print("line has no drugbank id: ",line)

        return DTI_drugbank_id_set


    def detect_DTI_drugbank_percent(self, DTI_drugbank_id_set):
        """
        查看DTI药物数据集与drugbank药物数据集的重合程度：
        :param DTI_drugbank_id_set: DTI药物数据集
        :return: DTI药物数量，DTI与drugbank重合的药物数量，最终使用DTI药物覆盖率
        """
        intersection = DTI_drugbank_id_set & self.drugbank_id_set
        return len(DTI_drugbank_id_set), len(intersection), len(intersection)/len(DTI_drugbank_id_set)

    def main_process(self):
        """
        0. 从DTI数据集中获取药物kegg ID集合。
        1. 从通过kegg　API获取药物在drugbank中的id, 将kegg_id与drugbank_id的mapping放到csv中
        3. 统计：DTI药物数量，DTI与drugbank重合的药物数量，最终使用DTI药物覆盖率
        :return:
        """
        self.extract_kegg_id()
        DTI_drugbank_id_set = self.get_drugbank_id_from_kegg_api()
        print(self.detect_DTI_drugbank_percent(DTI_drugbank_id_set))


class GetTargetNdSeq:
    """
    获取DTI数据集中靶向物质的nd seq
    """
    def __init__(self, source_file, target_file):
        self.source_file = source_file
        self.target_file = target_file
        self.hsa_id_set = set()

    def extract_hsa_id(self):
        """
        从DTI数据集中获取靶向物质hsa ID集合
        """
        with open(self.source_file, "r") as rf:
            lines = rf.readlines()
        for line in lines:
            ids = line.split()
            if len(ids) >= 2:
                self.hsa_id_set.add(ids[0])

    def get_nt_seq_from_kegg_api(self):
        """
        通过KEGG API获取target的nd seq，
        将结果保存至csv文件中,
        统计有nd seq的靶向物质占整个DTI数据集靶向物质的覆盖率
        """
        wf = open(self.target_file, "w")
        pattern = re.compile("[atcg]+")
        no_nt_seq_count = 0

        pool_manager = urllib3.PoolManager()
        prefix = "http://rest.kegg.jp/get/"
        suffix = "/ntseq"
        for hsa_id in self.hsa_id_set:
            url = prefix + hsa_id + suffix
            request = pool_manager.request("GET", url)
            response = request.data.decode()
            print("success get response from " + url)

            # 提取response中nd seq, 举个例子（http://rest.kegg.jp/get/hsa:1134/ntseq）
            lines = response.split("\n")
            line_no = 0
            nt_seq = ""
            for line in lines:
                line_no += 1
                if line_no == 1: # 跳过第一行
                    continue
                if pattern.findall(line): # 只匹配atcg字符串
                    nt_seq += line

            if nt_seq:
                wf.write(hsa_id + "," + nt_seq + "\n")
            else:
                no_nt_seq_count += 1

        wf.close()
        print("target has not nd seq:", no_nt_seq_count)

    def main_process(self):
        self.extract_hsa_id()
        self.get_nt_seq_from_kegg_api()


class NtSeqProcess:
    code_dict = {"a": 1, "c": 2, "g": 3, "t": 4}  # nt_seq 编码规则

    def __init__(self, max_seq_len, source_file, target_file):
        self.max_seq_len = max_seq_len
        self.source_file = source_file
        self.target_file = target_file


    def encode_hsa_id(self):
        """
        将DTI数据集中的nt_seq进行编码
        :return:
        """
        target_id_encode_seq_dict = {}

        with open(self.source_file, "r") as rf:
            lines = rf.readlines()
            for line in lines:
                items = line.split()
                hsa_id = items[0]
                if hsa_id not in target_id_encode_seq_dict.keys():
                    encode_seq = self.encode(items[1])
                    target_id_encode_seq_dict[hsa_id] = encode_seq

        with open(self.target_file, "wb") as wf:
            pickle.dump(target_id_encode_seq_dict, wf)

    def encode(self, seq):
        """
        对seq进行编码，为了对齐，对长度不足max_seq_len的字符串以０补全
        :param seq:
        :return: seq_array
        """
        seq_array = np.zeros([1, self.max_seq_len]) # seq to array
        seq_no = 0
        for item in seq:
            seq_array[0][seq_no] = self.code_dict[item]
            seq_no += 1

        return seq_array


def nt_seq_statistic():
    """
    统计靶向物质nd seq的最大与最小长度
    """
    max_seq_len = 0
    min_seq_len = sys.maxsize
    file_list = ["ic_target.csv", "e_target.csv", "gpcr_target.csv"]
    for file in file_list:
        with open("DTI/" + file, "r") as rf:
            lines = rf.readlines()
            for line in lines:
                line.split()
                if line:
                    seq_len = len(line[-1])
                    if max_seq_len < seq_len:
                        max_seq_len = seq_len
                    if min_seq_len > seq_len:
                        min_seq_len = seq_len

    print("max nt seq: ", max_seq_len)  # 15126
    print("min nt seq: ", min_seq_len)  # 261

def map_keggId_drugName():
    """
    统计DTI数据集的所有药物的keggId, drugbankId, 药物名称
    :return:
    """
    with open("/run/media/yuan/data/PycharmProjects/DDISuccess/Data/DDI/drugbankID_drug.pickle", "rb") as rf:
        drugbank_id_drug = pickle.load(rf)

    drug_file_list = ["ic_drug.csv", "e_drug.csv", "gpcr_drug.csv"]
    wf = open("keggId_drugbankId_name.csv",  "w")
    wf.write("kegg_id,drugbank_id,drug_name\n")
    kegg_id_set = set()
    for file in drug_file_list:
        with open(file, "r") as rf:
            lines = csv.reader(rf)
            for line in lines:
                if not line:
                    continue
                kegg_id = line[0]
                if kegg_id in kegg_id_set:
                    continue
                kegg_id_set.add(kegg_id)
                drugbank_id = line[1]
                drug_name = drugbank_id_drug.get(drugbank_id)
                if drug_name:
                    wf.write(kegg_id + "," + drugbank_id + "," + drug_name + "\n")

    print("kegg_id_set size: ", len(kegg_id_set))

# def filter_keggid_set(self, drug_file):
def filter_keggid_set(drug_file, target_file, draft_dataset_file):
    keggid_drugname_dict = {}
    with open("keggId_drugbankId_name.csv", "r") as rf:
        lines = csv.reader(rf)
        for line in lines:
            keggid_drugname_dict[line[0]] = line[-1]
    dataset_keggid_set = set(pd.read_csv(drug_file).iloc[:, 0])
    final_dataset_keggid_set = dataset_keggid_set

    feature_filename_list = ["drug_actionCode_matrix_dict.pickle", "drug_atc_dict.pickle",
                             "drug_MACCS166_dict.pickle", "drug_physiologicalCode_matrix_dict.pickle",
                             "drug_SIDER.pickle", "drug_target.pickle"]

    for drug_features_dict_file in feature_filename_list:
        keggid_set = set()
        with open("/run/media/yuan/data/PycharmProjects/DDISuccess/Data/DrugFeature/" + drug_features_dict_file, "rb") as rf:
            feature_drugname_set = pickle.load(rf).keys()
        # print("DDI drugname size: ", len(feature_drugname_set))

        for keggid in dataset_keggid_set:
            if keggid_drugname_dict.get(keggid) in feature_drugname_set:
                keggid_set.add(keggid)
        final_dataset_keggid_set = keggid_set & final_dataset_keggid_set
        # print(drug_features_dict_file, "drugname size: ", len(feature_drugname_set))
        # print("size of available drugname in dataset: ", len(keggid_set))
        # print("available drug / total drug = ", len(keggid_set) / len(dataset_keggid_set))

    print("size of final available drug in dataset: ", len(final_dataset_keggid_set))
    print("###available percentage: ", len(final_dataset_keggid_set)/len(dataset_keggid_set))

    # # 将具有完整特征的drug保存至文件中
    # df = pd.read_csv("keggId_drugbankId_name.csv")
    # kegg_id_list = list(df.iloc[:, 0])
    # drugbank_id_list = list(df.iloc[:, 1])
    # drugname_list = list(df.iloc[:, 2])
    # wf = open(fileter_drug_file, "w")
    # for kegg_id in final_dataset_keggid_set:
    #     index = kegg_id_list.index(kegg_id)
    #     drugbank_id = drugbank_id_list[index]
    #     drugname = drugname_list[index]
    #     wf.write(kegg_id + "," + drugbank_id + "," + drugname + "\n")
    # wf.close()

    # available DTI drug in DDI
    with open("/run/media/yuan/data/PycharmProjects/DDISuccess/Data/DDI/ddi_drug_features_dict_v5.pickle", "rb") as rf:
        ddi_drugname_set = pickle.load(rf).keys()

    keggid_set = set()
    for keggid in dataset_keggid_set:
        if keggid_drugname_dict.get(keggid) in ddi_drugname_set:
            keggid_set.add(keggid)
    print("drugbank drugname size: ", len(ddi_drugname_set))
    print("size of available drugname in dataset: ", len(keggid_set))
    print("***available drug / total drug = ", len(keggid_set) / len(dataset_keggid_set))

    # available DTI pair of drugbank feature in Data/draft/****.pickle
    target_set = set()
    with open(target_file, "r") as rf:
        lines = csv.reader(rf)
        for line in lines:
            target_set.add(line[0])

    positive_dti_dataset = set()
    draft_positive_dti_size = 0
    with open(draft_dataset_file, "r") as rf:
        lines = rf.readlines()
        for line in lines:
            dti = tuple(line.split())
            draft_positive_dti_size += 1
            # 去除一些缺失特征的drug/target的DTI pair
            if dti[-1] in final_dataset_keggid_set and dti[0] in target_set:
                positive_dti_dataset.add(dti)
    positive_dti_size = len(positive_dti_dataset)
    print("final positive dti size: ", positive_dti_size, "percentage: ", positive_dti_size/draft_positive_dti_size)

    return final_dataset_keggid_set

def get_dti_available_drugname_set(fileter_drug_file):
    """
    统计DTI数据集中具有完整特征的drug/target情况,并保存到csv文件中
    :return:
    """
    drug_file_list = ["e_drug.csv", "ic_drug.csv", "gpcr_drug.csv"]
    target_file_list = ["e_target.csv", "ic_target.csv", "gpcr_target.csv"]
    draft_dataset_file_list = ["bind_orfhsa_drug_e.txt", "bind_orfhsa_drug_ic.txt", "bind_orfhsa_drug_gpcr.txt"]
    filter_drug_set = set()
    for i in range(3):
        print(drug_file_list[i])
        filter_drug_set = filter_drug_set |filter_keggid_set(drug_file_list[i], target_file_list[i], draft_dataset_file_list[i])
        print("====================")

    df = pd.read_csv("keggId_drugbankId_name.csv")
    kegg_id_list = list(df.iloc[:, 0])
    drugbank_id_list = list(df.iloc[:, 1])
    drugname_list = list(df.iloc[:, 2])
    wf = open(fileter_drug_file, "w")
    for kegg_id in filter_drug_set:
        index = kegg_id_list.index(kegg_id)
        drugbank_id = drugbank_id_list[index]
        drugname = drugname_list[index]
        wf.write(kegg_id + "," + drugbank_id + "," + drugname + "\n")
    wf.close()


class BuildDrugFeatureDict:
    def __init__(self):
        self.feature_file_prefix = "/run/media/yuan/data/PycharmProjects/DDISuccess/Data/DrugFeature/"
        self.feature_filename_list = ["drug_actionCode_matrix_dict.pickle", "drug_atc_dict.pickle",
                                      "drug_MACCS166_dict.pickle", "drug_physiologicalCode_matrix_dict.pickle",
                                      "drug_SIDER.pickle", "drug_target.pickle"]
        self.feature_names = ["actionCode", "atc", "MACCS", "phyCode", "SIDER", "target"]

        self.keggid_drugname_dict = {}
        with open("keggId_drugbankId_name.csv", "r") as rf:
            lines = csv.reader(rf)
            for line in lines:
                self.keggid_drugname_dict[line[0]] = line[-1]

        self.drugname_set = self.get_available_drugname_set()  # now can read the files keggId_drugbankId_name_filter.csv

    def get_available_drugname_set(self):
        filter_mapping_file = "/run/media/yuan/data/PycharmProjects/DDISuccess/Data/DTI/draft/filter_keggId_drugbankId_name.csv"
        if os.path.isfile(filter_mapping_file):
            drugname_set = set(pd.read_csv(filter_mapping_file).iloc[:, 2])
            return drugname_set

        drugname_set = self.keggid_drugname_dict.values()
        for drug_features_dict_file in self.feature_filename_list:
            with open(self.feature_file_prefix + drug_features_dict_file,
                      "rb") as rf:
                feature_drugname_set = pickle.load(rf).keys()

            drugname_set = drugname_set & feature_drugname_set

        return drugname_set

    def main_process(self):
        old_feature_drug_matrix_dict = {}
        for i in range(len(self.feature_names)):
            old_feature_drug_matrix_dict[self.feature_names[i]] = self.filter_zero_character(
                self.get_dti_drug_matrix(self.feature_file_prefix
                                         + self.feature_filename_list[i]), self.feature_names[i])
        self.drugname_features_matrix_dict = {}
        for drugname in self.drugname_set:
            self.drugname_features_matrix_dict[drugname] = {}
            for feature in self.feature_names:
                self.drugname_features_matrix_dict[drugname][feature] = old_feature_drug_matrix_dict[feature][drugname]

        with open("dti_drug_features_dict.pickle", "wb") as wf:
            pickle.dump(self.drugname_features_matrix_dict, wf)

    def get_dti_drug_matrix(self, pickle_file):
        old_drug_character_dict = pd.read_pickle(pickle_file)
        temp_key = list(old_drug_character_dict.keys())[0]
        vector_size = len(old_drug_character_dict[temp_key])
        drug_character_dict = {}
        for drug in self.drugname_set:
            drug_character_dict[drug] = old_drug_character_dict[drug]
        print(pickle_file, "dimension: ", vector_size, "size: ", len(old_drug_character_dict))
        return drug_character_dict

    def filter_zero_character(self, old_drug_character_dict, character=None):
        """
        过滤掉冗余特征
        :param old_drug_character_dict:
        :return:
        """
        old_vector_size = len(old_drug_character_dict[list(old_drug_character_dict.keys()).pop()])
        print("old vector size:", old_vector_size)
        drug_size = len(old_drug_character_dict)
        old_metrix = np.zeros((drug_size, old_vector_size))
        key_list = list(old_drug_character_dict.keys())
        index = 0
        for key in key_list:
            old_metrix[index] = old_drug_character_dict[key]
            index += 1
        new_metrix = old_metrix[:, ~np.all(old_metrix == 0, axis=0)]
        # #---------------------------
        # sum_matrix = old_metrix.sum(axis=0)
        # sum_matrix[np.all(sum_matrix==0, axis=0)]
        # print(sum_matrix)
        # #---------------------------
        new_vector_size = len(new_metrix[0])
        print("delete vectors:", (old_vector_size - new_vector_size))
        drug_character_dict = {}

        index = 0
        for key in key_list:
            drug_character_dict[key] = new_metrix[index]
            index += 1
            if index == 1:
                print("new vector size:", len(drug_character_dict[key]))

        return drug_character_dict


class BuildDrugHierarchyPathsEmbeddingDict:
    PATH_LEN = 8
    VEC_SIZE = 128

    def __init__(self):
        with open("../../DrugFeature/id_deepwalkvec_dict.pickle", "rb") as rf:
            self.compoundid_deepwalkvec_dict = pickle.load(rf)

        df = pd.read_csv("../keggId_drugbankId_name.csv")
        self.drugbankid_list = list(df.iloc[:, 1])
        self.drugname_list = list(df.iloc[:, 2])

        self.drug_path_list = self.build_drug_hierarchy_paths()

    def build_drug_hierarchy_paths(self):

        drug_path_list = BuildHierarchyPaths(self.drugbankid_list).build_hierarchy_paths()

        # # ignore the same ancestor: 'CHEM0000000', 'CHEM9999999'
        # # because the dataset contains inorganic compounds: D01108,DB00653,magnesium sulfate199
        # filter_drug_path_list = []
        # for path in drug_path_list:
        #     drug_path = [path[i] for i in range(len(path)-2)]
        #     filter_drug_path_list.append(drug_path)

        return drug_path_list

    def main_process(self):
        dti_drug_hierarchy_pathsVec_dict = {}
        for i in range(len(self.drugbankid_list)):
            drugbankid = self.drugbankid_list[i]
            drugname = self.drugname_list[i]

            value_list = []
            path_len = 0
            for id in self.drug_path_list[i]:
                value_list.append(self.compoundid_deepwalkvec_dict[id])
                path_len += 1

            # no need to have the same length of path
            # while path_len < self.PATH_LEN:
            #     value_list.append(np.zeros(self.VEC_SIZE))
            #     path_len += 1

            dti_drug_hierarchy_pathsVec_dict[drugname] = np.array(value_list)

        print("dti_drug_hierarchy_pathsVec_dict size: ", len(dti_drug_hierarchy_pathsVec_dict))

        with open("dti_drug_hierarchy_pathsVec_dict.pickle", "wb") as wf:
            pickle.dump(dti_drug_hierarchy_pathsVec_dict, wf)


class BuildDrugHierarchyDescripEmbeddingDict:
    def __init__(self):
        with open("../../DrugFeature/cid_description_word2ve_all.pickle", "rb") as rf:
            self.cid_description_word2ve_dict = pickle.load(rf)
        with open("../../DrugFeature/drug_description_word2ve_all.pickle", "rb") as rf:
            self.drugbankid_description_word2ve_dict = pickle.load(rf)

        df = pd.read_csv("../keggId_drugbankId_name_filter.csv")
        self.drugbankid_list = list(df.iloc[:, 1])
        self.drugname_list = list(df.iloc[:, 2])

        self.drug_path_list = self.build_drug_hierarchy_paths()

    def build_drug_hierarchy_paths(self):

        drug_path_list = BuildHierarchyPaths(self.drugbankid_list).build_hierarchy_paths()

        # # ignore the same ancestor: 'CHEM0000000', 'CHEM9999999'
        # # because the dataset contains inorganic compounds: D01108,DB00653,magnesium sulfate199
        # filter_drug_path_list = []
        # for path in drug_path_list:
        #     drug_path = [path[i] for i in range(len(path)-2)]
        #     filter_drug_path_list.append(drug_path)

        return drug_path_list

    def main_process(self):
        dti_drug_hierarchy_descripVec_dict = {}
        drug_noDescrip_list = []
        empty_count = 0

        for i in range(len(self.drugbankid_list)):
            drugname = self.drugname_list[i]

            hierarchy_descripVec_list = []
            path_len = 0
            for id in self.drug_path_list[i]:
                if "DB" in id:
                    item_descrip_vecs = self.drugbankid_description_word2ve_dict.get(drugname)

                else:
                    item_descrip_vecs = self.cid_description_word2ve_dict.get(id)

                hierarchy_descripVec_list.append(item_descrip_vecs)

            dti_drug_hierarchy_descripVec_dict[drugname] = hierarchy_descripVec_list
            print(i, "drug: ", drugname, "vec shape:", np.shape(dti_drug_hierarchy_descripVec_dict[drugname]))

        print(drug_noDescrip_list)
        print("dti_drug_hierarchy_pathsVec_dict size: ", len(dti_drug_hierarchy_descripVec_dict))
        print("empty count: ", empty_count)
        print("all drug", len(self.drugbankid_list), len(self.drugbankid_list)) # keggid:drugbankid = n:1, so len(self.drugbankid_list), len(self.drugbankid_list)
        print("all drug path", len(self.drug_path_list))

        with open("dti_drug_hierarchy_descripVec_dict.pickle", "wb") as wf:
            pickle.dump(dti_drug_hierarchy_descripVec_dict, wf)


MAX_SEQ_LEN = 15126
if __name__ == '__main__':
    # with open("DDI/drugbankID_drug.pickle", "rb") as rf:
    #     drugbank_id_drug = pickle.load(rf)
    # drugbank_drug_id_set = drugbank_id_drug.keys()
    # #根据DTI　gpcr药物数集据获取其在Drugbank中ID, 统计DTI药物数据集与drugbank药物数据集的重合程度
    # source_file = "DTI/bind_orfhsa_drug_gpcr.txt"
    # target_file = "DTI/gpcr_drug.csv"
    # get_drugbank_dataset = GetDrugBankIDSet(source_file, target_file, drugbank_drug_id_set)
    # get_drugbank_dataset.main_process()

    # # 根据根据DTI　gpcr药物数集据获取靶向物质的nt seq
    # source_file = "DTI/bind_orfhsa_drug_ic.txt"
    # target_file = "DTI/ic_target.csv"
    # get_hsa_dataset = GetTargetNdSeq(source_file, target_file)
    # get_hsa_dataset.main_process()

    ## 统计靶向物质nd seq的最大与最小长度
    #nt_seq_statistic()


    # # 将DTI数据集中的nt_seq进行编码
    # ic_source_file = "DTI/ic_target.csv"
    # ic_target_file = "DTI/ic_targetId_encodeSeq_dict.pickle"
    # ic_nt_seq_process = NtSeqProcess(15126, ic_source_file, ic_target_file)
    # ic_nt_seq_process.encode_hsa_id()
    #
    # e_source_file = "DTI/e_target.csv"
    # e_target_file = "DTI/e_targetId_encodeSeq_dict.pickle"
    # e_nt_seq_process = NtSeqProcess(15126, e_source_file, e_target_file)
    # e_nt_seq_process.encode_hsa_id()
    #
    # gpcr_source_file = "DTI/gpcr_target.csv"
    # gpcr_target_file = "DTI/gpcr_targetId_encodeSeq_dict.pickle"
    # gpcr_nt_seq_process = NtSeqProcess(15126, gpcr_source_file, gpcr_target_file)
    # gpcr_nt_seq_process.encode_hsa_id()

    # map_keggId_drugName()
    # BuildDrugFeatureDict().main_process()

    # get_dti_available_drugname_set("../keggId_drugbankId_name_filter.csv")

    # 去掉DDI-DataSource中提取的cid_description_word2ve_all.pickle中key的空格
    # with open("../../DrugFeature/cid_description_word2ve_all.pickle", "rb") as rf:
    #     cid_description_word2ve_dict = pickle.load(rf)
    #
    # new_cid_description_word2ve_dict = {}
    #
    # for cid in cid_description_word2ve_dict.keys():
    #     new_cid = cid.replace(" ", "")
    #     new_cid_description_word2ve_dict[new_cid] = cid_description_word2ve_dict[cid]
    #
    # with open("../../DrugFeature/cid_description_word2ve_all.pickle", "wb") as wf:
    #     pickle.dump(new_cid_description_word2ve_dict, wf)

    BuildDrugHierarchyPathsEmbeddingDict().main_process()
    BuildDrugHierarchyDescripEmbeddingDict().main_process()
