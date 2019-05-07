
- drug_actionCode_matrix_dict 584 221
- drug_phyCode_matrix_dict    584 326
- drug_SIDER_matrix_dict      584 4467
- drug_target_matrix_dict     584 681
- drug_atc_matrix_dict        584 562
- drug_MACCS_matrix_dict      584 157
- drug_$hierarchy_matrix_dict 584 127
- drug_word2vec_matrix_dict   584 100

- drugs_ddi_v5.pickle: 以上特征矩阵的keys集合（药物名称），eg.drug_actionCode_matrix_dict.keys()
- ddi_rel_v5.pickle: 三元组列表：[ [drug1, drug2, "increase"], ...], [[drug1, drug2, "decreae"], ...]
- ddi_drug_features_dict_v5.pickle: 字典结构

  - key为药物名称（全小写）

    key includes: actionCode, atc, MACCS, SIDER, phyCode, target

  - value为药物特征: 字典结构，key为特征名称,value为特征向量

- drugbankID_drug.pickle
  字典结构，涵盖了drugbank v5.09共10500药物名称及其drugbank ID

  - key为drugbank ID

  - name为drug name,
