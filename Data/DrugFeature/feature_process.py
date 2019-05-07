# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: feature_process.py
@time: 19-1-14 下午9:52
@description:
"""
import numpy
import pickle
import sys
import re

class BuildHierarchyPaths:

    def __init__(self, drugbank_id_list, save_path=None):
        self.drugbank_id_list = drugbank_id_list
        self.save_path = save_path
        self.hierarchy_dict = self.parse_edgelist()

    def parse_edgelist(self):
        with open("../../DrugFeature/hierarchy.edgelist", "r") as rf:
            lines = rf.readlines()

        son_father_dict = {}
        for line in lines:
            items = line.split()
            if not items:
                continue
            son = items[0]
            father = items[1]
            if son in son_father_dict.keys():
                print("a son has more than one father: ", son)
                break
            else:
                son_father_dict[son] = father

        return son_father_dict

    def build_hierarchy_paths(self):
        drug_path_list = []
        drug_has_no_path = []
        for drug_id in self.drugbank_id_list:
            if drug_id not in self.hierarchy_dict:
                drug_path_list.append([])
                drug_has_no_path.append(drug_id)
            else:
                drug_path_list.append(self.detect_hierarchy_path(drug_id))

        print("drug has no path: ", drug_has_no_path)
        self.path_statistics(drug_path_list)
        print("=============\n\n")
        ## save
        #with open(self.save_path, "wb") as wf:
        #    pickle.dump(drug_path_list, wf)

        return drug_path_list

    def detect_hierarchy_path(self, drug_id):
        path = [drug_id, ]
        son_id = drug_id
        while son_id in self.hierarchy_dict.keys():
            son_id = self.hierarchy_dict[son_id]
            path.append(son_id)
        if not path:
            print(drug_id)
        return path

    def path_statistics(self, drug_path_list):
        """
        统计药物hierarchy path的情况：
        最长路径，最短路径，以及属于无机物的药物
        :param drug_path_list:
        :return:
        """
        max_path_len = -1
        max_path = ""
        min_path_len = sys.maxsize
        min_path = ""
        inorganic_paths = []
        for path in drug_path_list:
            path_len = len(path)
            if path_len > max_path_len:
                max_path_len = path_len
                max_path = path
            if path_len < min_path_len:
                min_path_len = path_len
                min_path = path
            # print("path_len:", path_len)
            if path_len == 0:
                print("***", path)
            if path[-2] == "CHEM0000001":
                inorganic_paths.append(path)

        print("max length: ", max_path_len, max_path)
        print("min lenght: ", min_path_len, min_path)
        print("inorganic_path", inorganic_paths)

def pares_network_embedding_file():
    drug_pattern = re.compile("DB\d{5}")
    compound_pattern = re.compile("CHEM\d{7}")
    compoundid_deepwalkvec_dict = {}
    with open("deepwalk_vec.txt", "r") as rf:
        lines = rf.readlines(1)
        while lines:
            lines = rf.readlines(1000)
            for line in lines:
                if "DB" in line:
                    key = drug_pattern.findall(line)[0] if drug_pattern.findall(line) else None
                else:
                    key = compound_pattern.findall(line)[0] if compound_pattern.findall(line) else None

                if not key:
                    print(line)
                    continue
                value_str = "[" + line.replace(key+" ", "").replace(" ", ",") + "]"
                # print(value_str)
                array = numpy.array(eval(value_str))
            compoundid_deepwalkvec_dict[key] = array

    print("compoundid_deepwalkvec_dict size: ", len(compoundid_deepwalkvec_dict))

    with open("id_deepwalkvec_dict.pickle", "wb") as wf:
        pickle.dump(compoundid_deepwalkvec_dict, wf)


if __name__ == '__main__':
    pares_network_embedding_file()
