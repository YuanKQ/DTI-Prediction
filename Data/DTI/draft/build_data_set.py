# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: build_data_set.py
@time: 19-1-7 下午7:44
@description:
"""
import csv
import itertools
import json
import random
import re
import pandas as pd
import numpy as np
import pickle

from Data.DrugFeature.feature_process import BuildHierarchyPaths


def get_newset_dti_dataset():
    filename_target_dict = {"gpcr": "G Protein-coupled receptors", "ic": "Ion channels", "e": "Enzymes"}
    with open("br08310.keg", "r") as rf:
        content = rf.read()

    # br08310.keg包含不同的DTI数据集, 在提取时需要进行划分
    dti_dataset = content.split("#")
    for filename in filename_target_dict.keys():
        for dtis in dti_dataset:
            if filename_target_dict[filename] in dtis:
                dti_list = extract_dti_dataset_from_kegfile(dtis)
                print(filename, "size: ", len(dti_list))

                with open("{}_newest.csv".format(filename), "w") as wf:
                    wf_csv = csv.writer(wf)
                    wf_csv.writerows(dti_list)
                break

def extract_dti_dataset_from_kegfile(content):
    """
    通过正则匹配提取keg file中的DTI pair
    :param content: 包含DTI 数据集的keg文件内容
    :return:
    """
    dti_list = []
    pattern = re.compile("D\d{5}")
    target_drug_parts = content.split("HSA:")
    for part in target_drug_parts:
        temps = part.split("]")
        if len(temps) < 2:
            continue
        hsa_str = temps[0]
        drug_str = temps[-1]

        # extract HSA set
        hsa_list = ["hsa:" + item for item in hsa_str.split()]

        # extract drug set
        drug_list = pattern.findall(drug_str)

        # combine hsa set and drug set  '
        dti_list.extend(list(itertools.product(hsa_list, drug_list)))

    return set(dti_list)

class BuildDataset:
    """
    构建DTI数据集：
    1. 构建负样本数据集
    2. 给样本数据集加上tag
    3. shuffle正负样本
    4. 保存DTI数据集至csv文件中
    """
    SEED = 144
    def __init__(self, draft_dataset_file, newest_dataset_file, drug_features_dict_file, kegg_drugbank_map_file,
                 drug_file, target_file, dataset_tag_file):
        self.drug_features_dict_file = drug_features_dict_file
        self.kegg_drugbank_map_file = kegg_drugbank_map_file
        self.drug_file = drug_file
        # the file which stored the newest kegg DTI dataset
        self.newest_dataset_file = newest_dataset_file

        self.dataset_file = draft_dataset_file
        self.tag_dataset_file = dataset_tag_file

        # 只留取DTI具有完整特征的drug
        self.keggid_set = self.filter_keggid_set()

        self.target_set = set()
        with open(target_file, "r") as rf:
            lines = csv.reader(rf)
            for line in lines:
                self.target_set.add(line[0])

        self.positive_dti_dataset = set()
        with open(draft_dataset_file, "r") as rf:
            lines = rf.readlines()
            for line in lines:
                dti = tuple(line.split())
                # 去除一些缺失特征的drug/target的DTI pair
                if dti[-1] in self.keggid_set and dti[0] in self.target_set:
                    self.positive_dti_dataset.add(dti)
        self.positive_dti_size = len(self.positive_dti_dataset)

        # the newest kegg dti dataset
        self.newest_dti_dataset = set()
        with open(newest_dataset_file, "r") as rf:
            lines = csv.reader(rf)
            for line in lines:
                self.newest_dti_dataset.add(tuple(line))

        self.negative_dti_dataset = self.build_negative_dataset()

        print("finish initializing...")
        print("drug set size: ", len(self.keggid_set))
        print("target set size: ", len(self.target_set))
        print("newest dti dataset size: ", len(self.newest_dti_dataset))
        print("positive dti dataset size: ", self.positive_dti_size)
        print("negative dti dataset size: ", len(self.negative_dti_dataset))
        print("\n\n")

    def filter_keggid_set(self):
        keggid_set = set()
        with open(self.drug_features_dict_file, "rb") as rf:
            DDI_drugname_set = pickle.load(rf).keys()
        print("DDI drugname size: ", len(DDI_drugname_set))

        keggid_drugname_dict = {}
        with open(self.kegg_drugbank_map_file, "r") as rf:
            lines = csv.reader(rf)
            for line in lines:
                keggid_drugname_dict[line[0]] = line[-1]

        keggid_list = list(pd.read_csv(self.drug_file).iloc[:, 0])

        for keggid in keggid_list:
            if keggid_drugname_dict.get(keggid) in DDI_drugname_set:
                keggid_set.add(keggid)
        print("available drug / total drug in dataset: ", len(keggid_set)/len(keggid_list))
        return keggid_set

    def build_negative_dataset(self):
        """
        构建负样本数据集
        :return: negative_dti_set 负样本数据集
        """
        # 将target与drug随机组合为negative_DTI_set(drug随机取3个target进行组合）
        random_dti_set = set()
        for drug in self.keggid_set:
            random_target_list = random.sample(self.target_set, 10)
            for target in random_target_list:
                random_dti_list = (target, drug)
                random_dti_set.add(random_dti_list)

        # 取与new_DTI_set的差集为最终的negative_DTI_set
        random_negative_dti_set = random_dti_set - self.newest_dti_dataset
        if len(random_negative_dti_set) < self.positive_dti_size:
            print("*** Please adjust the size of target randomly selected. ***")
            return None
        else:
            negative_dti_set = random.sample(random_negative_dti_set, self.positive_dti_size)
            print("intersection between random_negative_dti_set and newest_dti_dataset: ",
                  len(random_negative_dti_set & self.newest_dti_dataset))

            return negative_dti_set

    def build_dataset(self):
        # add tag to positive dti dataset
        tag_dataset = self.add_tag_to_dataset(self.positive_dti_dataset, 1)
        # add tag to negative dti dataset
        tag_dataset.extend(self.add_tag_to_dataset(self.negative_dti_dataset, 0))

        # shuffle正负样本
        random.seed(self.SEED)
        random.shuffle(tag_dataset)

        with open(self.tag_dataset_file, "w") as wf:
            csv_writer = csv.writer(wf)
            csv_writer.writerows(tag_dataset)


    def add_tag_to_dataset(self, dataset, tag):
        """
        add tag to each pair in dataset
        :param dataset:
        :param tag: 1 means positive, 0 means negative
        :return: a dataset with tag stored in list type
        """
        tag_dataset = list()
        for pair in dataset:
            pair += (tag, )
            tag_dataset.append(pair)

        return tag_dataset

class BuildInputData:
    TRAIN_TEST_RATIO = 19/20
    # all kind of data save to one h5 file
    def __init__(self, train_data_file, test_data_file,
                 tag_dataset_file, kegg_drugbank_map_file,
                 drug_features_dict_file, drug_hierarchy_pathsVec_file, dti_drug_hierarchy_descripVec_file,
                 target_seq_file):
        self.target_seq_file = target_seq_file
        self.train_data_file = train_data_file
        self.test_data_file = test_data_file

        data = pd.read_csv(tag_dataset_file)
        self.target_hsaid_list = list(data.iloc[:, 0])
        drug_keggid_list = list(data.iloc[:, 1])
        self.label_list = list(data.iloc[:, 2])

        self.keggid_drugname_dict = {}
        self.keggid_drugbankid_dict = {}
        with open(kegg_drugbank_map_file, "r") as rf:
            lines = csv.reader(rf)
            for line in lines:
                self.keggid_drugname_dict[line[0]] = line[-1]
                self.keggid_drugbankid_dict[line[0]] = line[1]

        self.drugname_list = []
        for keggid in drug_keggid_list:
            self.drugname_list.append(self.keggid_drugname_dict.get(keggid))

        # extract drug feature embedding
        self.target_seq_list = self.build_target_seq()
        self.drug_pharmacologicy_list = self.build_drug_pharmacologicy(drug_features_dict_file)
        self.drug_hierarchy_text_list = self.build_drug_hierarchy_text(dti_drug_hierarchy_descripVec_file)
        self.drug_hierarchy_structure_list = self.build_drug_hierarchy_structure(drug_hierarchy_pathsVec_file)

    def build_target_seq(self):
        with open(self.target_seq_file, "rb") as rf:
            target_id_encode_seq_dict = pickle.load(rf)

        target_seq_list = []
        for hsa_id in self.target_hsaid_list:
            target_seq_list.append(target_id_encode_seq_dict[hsa_id][0])

        return target_seq_list

    def build_drug_pharmacologicy(self, drug_features_dict_file):
        with open(drug_features_dict_file, "rb") as rf:
            drug_features_dict = pickle.load(rf)

        features_list = []
        for drug in self.drugname_list:
            drug_features = drug_features_dict.get(drug)
            single_drug_features_array = np.hstack((drug_features["actionCode"], drug_features["atc"],
                                                    drug_features["MACCS"], drug_features["SIDER"],
                                                    drug_features["phyCode"], drug_features["target"]))
            features_list.append(single_drug_features_array)

        return features_list

    def build_drug_hierarchy_text(self, dti_drug_hierarchy_descripVec_file):
        with open(dti_drug_hierarchy_descripVec_file, "rb") as rf:
            drug_hierarchy_descripVec_dict = pickle.load(rf)

        descripVec_list = []
        for drug in self.drugname_list:
            descripVec_list.append(drug_hierarchy_descripVec_dict[drug])
        return descripVec_list

    def build_drug_hierarchy_structure(self, drug_hierarchy_pathsVec_file):
        with open(drug_hierarchy_pathsVec_file, "rb") as rf:
            drug_hierarchy_pathsVec_dict = pickle.load(rf)

        pathsVec_list = []
        for drug in self.drugname_list:
            pathsVec_list.append(drug_hierarchy_pathsVec_dict[drug])

        return pathsVec_list


    def main_process(self):
        data_size = len(self.label_list)
        train_size = int(data_size * self.TRAIN_TEST_RATIO)

        # change to pickle
        with open(self.train_data_file, "wb") as wf:
            pickle.dump(self.label_list[0:train_size], wf)
            pickle.dump(self.target_seq_list[0:train_size], wf)
            pickle.dump(self.drug_pharmacologicy_list[0:train_size], wf)
            pickle.dump(self.drug_hierarchy_structure_list[0:train_size], wf)
            pickle.dump(self.drug_hierarchy_text_list[0:train_size], wf)

        with open(self.test_data_file, "wb") as wf:
            pickle.dump(self.label_list[train_size:], wf)
            pickle.dump(self.target_seq_list[train_size:], wf)
            pickle.dump(self.drug_pharmacologicy_list[train_size:], wf)
            pickle.dump(self.drug_hierarchy_structure_list[train_size:], wf)
            pickle.dump(self.drug_hierarchy_text_list[train_size:], wf)

        # train_label_df = pd.DataFrame(np.array(self.label_list)[0:train_size])
        # train_label_df.to_hdf(self.train_data_file, key="labels")
        # train_target_df = pd.DataFrame(np.array(self.target_seq_list)[0:train_size])
        # train_target_df.to_hdf(self.train_data_file, key="target")
        # train_pharmacologicy_df = pd.DataFrame(np.array(self.drug_pharmacologicy_list)[0:train_size])
        # train_pharmacologicy_df.to_hdf(self.train_data_file, key="pharmacologicy")
        # train_pathsvec_df = pd.DataFrame(self.drug_hierarchy_structure_list[0:train_size]) # DataFrame Must pass 2-d input
        # train_pathsvec_df.to_hdf(self.train_data_file, key="paths")
        # train_descripvec_df = pd.DataFrame(self.drug_hierarchy_text_list[0:train_size]) # read_hdf can't reduce the list which contains different shape of arrays
        # train_descripvec_df.to_hdf(self.train_data_file, key="text")
        #
        # test_lable_df = pd.DataFrame(np.array(self.label_list)[train_size:])
        # test_lable_df.to_hdf(self.test_data_file, key="labels")
        # test_target_df = pd.DataFrame(np.array(self.target_seq_list)[train_size:])
        # test_target_df.to_hdf(self.test_data_file, key="target")
        # test_pharmacologicy_df = pd.DataFrame(np.array(self.drug_pharmacologicy_list)[train_size:])
        # test_pharmacologicy_df.to_hdf(self.test_data_file, key="pharmacologicy")
        # test_pathsvec_df = pd.DataFrame(self.drug_hierarchy_structure_list[train_size:])
        # test_pathsvec_df.to_hdf(self.test_data_file, key="paths")
        # test_descripvec_df = pd.DataFrame(self.drug_hierarchy_text_list[train_size:])
        # test_descripvec_df.to_hdf(self.test_data_file, key="text")


def merge_train_dataset():
    file_folder = "../"
    train_file_list = ["e_train.pickle", "gpcr_train.pickle", "ic_train.pickle"]
    label_list = []
    target_seq_list = []
    pharmacologicy_list = []
    hierarchy_deepwalk_list = []
    hierarchy_descrip_vec_list = []
    for file in train_file_list:
        file_path = file_folder + file
        with open(file_path, "rb") as rf:
            label_list.extend(pickle.load(rf))
            target_seq_list.extend(pickle.load(rf))
            pharmacologicy_list.extend(pickle.load(rf))
            hierarchy_deepwalk_list.extend(pickle.load(rf))
            hierarchy_descrip_vec_list.extend(pickle.load(rf))

    with open("../dti_train.pickle", "wb") as wf:
        pickle.dump(label_list, wf)
        pickle.dump(target_seq_list, wf)
        pickle.dump(pharmacologicy_list, wf)
        pickle.dump(hierarchy_deepwalk_list, wf)
        pickle.dump(hierarchy_descrip_vec_list, wf)


if __name__ == '__main__':
    ## 构建DTI数据集
    # e_dataset = BuildDataset("bind_orfhsa_drug_e.txt", "e_newest.csv",
    #                          "dti_drug_features_dict.pickle", "../keggId_drugbankId_name.csv", "e_drug.csv", "e_target.csv", "e_dataset.csv")
    # e_dataset.build_dataset()
    #
    # ic_dataset = BuildDataset("bind_orfhsa_drug_ic.txt", "ic_newest.csv",
    #                           "dti_drug_features_dict.pickle", "../keggId_drugbankId_name.csv", "ic_drug.csv", "ic_target.csv", "ic_dataset.csv")
    # ic_dataset.build_dataset()
    #
    # gpcr_dataset = BuildDataset("bind_orfhsa_drug_gpcr.txt", "gpcr_newest.csv",
    #                             "dti_drug_features_dict.pickle", "../keggId_drugbankId_name.csv", "gpcr_drug.csv", "gpcr_target.csv", "gpcr_dataset.csv")
    # gpcr_dataset.build_dataset()

    # 构建train/test数据集
    e_dataset = BuildInputData("../e_train.pickle", "../e_test.pickle", "e_dataset.csv", "keggId_drugbankId_name.csv",
                               "dti_drug_features_dict.pickle", "dti_drug_hierarchy_pathsVec_dict.pickle", "dti_drug_hierarchy_descripVec_dict.pickle",
                               "e_targetId_encodeSeq_dict.pickle")
    e_dataset.main_process()

    ic_dataset = BuildInputData("../ic_train.pickle", "../ic_test.pickle", "ic_dataset.csv", "keggId_drugbankId_name.csv",
                                "dti_drug_features_dict.pickle", "dti_drug_hierarchy_pathsVec_dict.pickle", "dti_drug_hierarchy_descripVec_dict.pickle",
                                "ic_targetId_encodeSeq_dict.pickle")
    ic_dataset.main_process()

    gpcr_dataset = BuildInputData("../gpcr_train.pickle", "../gpcr_test.pickle", "gpcr_dataset.csv", "keggId_drugbankId_name.csv",
                                  "dti_drug_features_dict.pickle", "dti_drug_hierarchy_pathsVec_dict.pickle", "dti_drug_hierarchy_descripVec_dict.pickle",
                                  "gpcr_targetId_encodeSeq_dict.pickle")
    gpcr_dataset.main_process()

    # merge_train_dataset()
