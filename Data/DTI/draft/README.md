# File Description
### br08310.keg
Compared to the file downloaded from http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/,
**br08310.keg** has deleted some content:
```
+E  Drug
#<h2><a href="/kegg/kegg2.html"><img src="/Fig/bget/kegg3.gif" align="middle" border=0></a>&nbsp; Target-based Classification of Drugs</h2>
!
```

### e_dataset.csv, gpcr_dataset.csv, ic_dataset.csv
- col 1 is  targetId
- col 2 is drug ID,
- col 3 DTI tag

  1 is positive, which download from http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/

  0 is negative, whose pair is random combination)

### keggId_drugbankId_name.csv
- col 1 is kegg id of drug
- col 2 is drugbank id of drug
- col 3 is drug name in drugbank

### dti_drug_features_dict.pickle
- key: drug name (lower case)
- value: np.array, shape= 1 * 5080

#### ./Data/DrugFeature/drug_actionCode_matrix_dict.pickle dimension:  626 size:  1388
old vector size: 626

delete vectors: 463

new vector size: 163

#### ./Data/DrugFeature/drug_atc_dict.pickle dimension:  867 size:  1946
old vector size: 867

delete vectors: 453

new vector size: 414

#### ./Data/DrugFeature/drug_MACCS166_dict.pickle dimension:  166 size:  8716
old vector size: 166

delete vectors: 15

new vector size: 151

#### ./Data/DrugFeature/drug_physiologicalCode_matrix_dict.pickle dimension:  1866 size:  1321
old vector size: 1866

delete vectors: 1612

new vector size: 254

#### ./Data/DrugFeature/drug_SIDER.pickle dimension:  4876 size:  892
old vector size: 4876

delete vectors: 1270

new vector size: 3606

#### ./Data/DrugFeature/drug_target.pickle dimension:  3880 size:  6837
old vector size: 3880

delete vectors: 3388

new vector size: 492


### dti_drug_hierarchy_descripVec_dict.pickle
#### key
drug name (lower case)
#### value
list of arrays, which are ancesters descrription word embedding

shape: drug_hierarchy_path_length * ancestors_description_word_count * 100 (word embedding dimension)
> different drug has different values of drug_hierarchy_path_length.

since the numbers of words of ancesters descrription are multiple, the value can't be transformed by `np.array(value)`

    each ancester descrription word embedding: {ndarray}\[wordvec1, wordvec2,..., ]

    dimension of each word embedding is 100.

    the len of the arrays of ancester descrription word embedding are multiple, depends on the len of ancester description.

    ancestor descriptions are extracted from ChemObO.ont

    drug descriptions are extracted from drugbank

### dti_drug_hierarchy_pathsVec_dict.pickle

#### key
drug name (lower case)
#### value
list of arrays, which are ancesters deepwalkvec.

dimension of each word embedding is 128.

shape: drug_hierarchy_path_length * 128

  > different drug has different values of drug_hierarchy_path_length.

### e_targetId_encodeSeq_dict.pickle,

- key: target id in kegg knowledge base, format like `hsa:\d{1,6}`
- value: np.array, shape=1 * 15125
