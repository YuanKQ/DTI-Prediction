# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: case_study.py
@time: 2019/3/20 14:35
@description: 处理两个药物头孢曲松（Ceftriaxone）和头孢噻肟（Cefotaxime）的DDI重合情况
"""

import pandas as pd

def parse_rel_ddi(file):
    increase_ddi_drugs = set()
    decrease_ddi_drugs = set()
    with open(file, "r") as rf:
        lines = rf.readlines()
    for line in lines:
        items = line.split(",")
        if "increase" in line:
            increase_ddi_drugs.add(items[0])
        else:
            decrease_ddi_drugs.add(items[0])

    return increase_ddi_drugs, decrease_ddi_drugs


def statistics_ddi(cefotaxime_ddi_increase_drugs, cefotaxime_ddi_decrease_drugs,
                   ceftriaxone_ddi_increase_drugs, ceftriaxone_ddi_decrease_drugs,
                   name1="cefotaxime",name2="ceftriaxone"):
    increase_intersection = cefotaxime_ddi_increase_drugs & ceftriaxone_ddi_increase_drugs
    ceftriaxone_increase_difference = ceftriaxone_ddi_increase_drugs - cefotaxime_ddi_increase_drugs
    cefotaxime_increase_difference = cefotaxime_ddi_increase_drugs - ceftriaxone_ddi_increase_drugs

    print("increase:")
    print("intersection: ", len(increase_intersection))#, increase_intersection)
    print("{}_difference: ".format(name1), len(ceftriaxone_increase_difference))#, ceftriaxone_increase_difference)
    print("{}_difference: ".format(name2), len(cefotaxime_increase_difference))#, cefotaxime_increase_difference)
    print("=============================")

    decrease_intersection = cefotaxime_ddi_decrease_drugs & ceftriaxone_ddi_decrease_drugs
    ceftriaxone_decrease_difference = ceftriaxone_ddi_decrease_drugs - cefotaxime_ddi_decrease_drugs
    cefotaxime_decrease_difference = cefotaxime_ddi_decrease_drugs - ceftriaxone_ddi_decrease_drugs
    print("decrease:")
    print("intersection: ", len(decrease_intersection))#, decrease_intersection)
    print("{}_difference: ".format(name1), len(ceftriaxone_decrease_difference))#, ceftriaxone_decrease_difference)
    print("{}_difference: ".format(name2), len(cefotaxime_decrease_difference))#, cefotaxime_decrease_difference)

    print("=============================")
    print("contrast: ")
    ceftriaxone_increase_cefotaxime_decrease = ceftriaxone_ddi_increase_drugs & cefotaxime_ddi_decrease_drugs
    cefotaxime_increase_ceftriaxone_decrease = cefotaxime_ddi_increase_drugs & ceftriaxone_ddi_decrease_drugs
    print("{}_increase_{}_decrease: ".format(name1, name2), len(ceftriaxone_increase_cefotaxime_decrease))#,ceftriaxone_increase_cefotaxime_decrease)
    print("{}_increase_{}_decrease: ".format(name2, name1), len(cefotaxime_increase_ceftriaxone_decrease))#,cefotaxime_increase_ceftriaxone_decrease)

    print("=============================")
    print("total:")
    ceftriaxone= cefotaxime_ddi_increase_drugs | cefotaxime_ddi_decrease_drugs
    cefotaxime = ceftriaxone_ddi_increase_drugs | ceftriaxone_ddi_decrease_drugs
    intersection = cefotaxime & ceftriaxone
    ceftriaxone_difference = ceftriaxone - cefotaxime
    cefotaxime_difference = cefotaxime - ceftriaxone
    print("intersection: ", len(intersection))#, intersection)
    print("{}_difference: ".format(name1), len(ceftriaxone_difference))#, ceftriaxone_difference)
    print("{}_difference: ".format(name2), len(cefotaxime_difference))#, cefotaxime_difference)

    print("\n\n\n")

cefotaxime_ddi_increase_drugs, cefotaxime_ddi_decrease_drugs = parse_rel_ddi("cefotaxime_all.csv")
ceftriaxone_ddi_increase_drugs, ceftriaxone_ddi_decrease_drugs = parse_rel_ddi("ceftriaxone_all.csv")
meropenem_ddi_increase_drugs, meropenem_ddi_decrease_drugs = parse_rel_ddi("meropenem_all.csv")
statistics_ddi(cefotaxime_ddi_increase_drugs, cefotaxime_ddi_decrease_drugs,
                ceftriaxone_ddi_increase_drugs, ceftriaxone_ddi_decrease_drugs)
statistics_ddi(meropenem_ddi_increase_drugs, meropenem_ddi_decrease_drugs,
               ceftriaxone_ddi_increase_drugs, ceftriaxone_ddi_decrease_drugs,
               "meropenem", "ceftiraxone")
statistics_ddi(meropenem_ddi_increase_drugs, meropenem_ddi_decrease_drugs,
               cefotaxime_ddi_increase_drugs, cefotaxime_ddi_decrease_drugs,
               "meropenem", "cefotaxime")


