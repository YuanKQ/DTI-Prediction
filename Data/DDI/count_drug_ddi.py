# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: count_drug_ddi.py
@time: 2019/3/7 22:05
@description:
"""

import pickle

# 统计ddi pair的数目
with open("ddi_rel_v5.pickle", "rb") as rf:
    DDIs = pickle.load(rf)
print("ddi: ", len(DDIs))

# 统计ddi中drug的数目
with open("drug_word2vec_matrix_dict.pickle", "rb") as rf:
    drugs = pickle.load(rf)
print("drugs: ", len(drugs))

def find_ddi_drugs(key):
    key_set = set()
    for ddi in DDIs:
        if key in ddi:
            if key == ddi[0]:
                key_set.add(ddi[1])
            else:
                key_set.add(ddi[0])
    return key_set

keys = ["ceftriaxone", "cefotaxime"]
ceftriaxone_set = find_ddi_drugs(keys[0])
cefotaxime_set = find_ddi_drugs(keys[1])
intersection = cefotaxime_set & cefotaxime_set
print("intersection: ", intersection)
cefotaxime_difference_set = cefotaxime_set - ceftriaxone_set
ceftriaxone_difference_set = ceftriaxone_set - cefotaxime_set
print("cefotaxime_difference_set：", cefotaxime_difference_set)
print("ceftriaxone_difference_set: ", ceftriaxone_difference_set)

