Data files:

1.PRAD_genes_list1.csv
Expression values of 46 Prolaris genes (15 reference genes and 31 test genes) in 498 TCGA PRAD patients 

2.PRAD_genes_list2a.csv
Expression value of 17 Onctoypte DX genes (5 reference genes and 12 test genes) in 498 TCGA PRAD patients 

3.PRAD_genes_list3a.csv
Expression value of 16 Decipher genes  in 498 TCGA PRAD patients 

4. PRAD_clinical_pfi_os.csv
Clinical data including progression free interval and censor status of 498 TCGA PRAD patients


Python code
1. gexp1.ipynb
Divide patients in three classes: high risk: PFI < 2 years, medium risk: 2 years <= PFI < 5 years, low risk: PFI >=5 years. Number of samples in high, medium, and low risk groups: 52, 30, 63. Total 145 patients

Extract expression values of 46 Prolaris genes in 145 samples and add class label


2. gexp2.ipynb
Extract expression values of 17 Oncotype DX genes in 145 samples and add class label


3. gexp3.ipynb
Extract expression values of 16 Decipher genes in 145 samples and add class label

4. testprolaris.py
Implement the original Prolaris test
5. Xgbprolaris_CV.py
Implement XGBoost classifier for Prolaris test

6. testoncoprad.py
Implement the original Oncotype DX test
7. XgbOncotype_CV.py
Implement XGBoost classifier for Oncotype DX test

8. rfDecipher_CV.py
Implement the original Decipher test using random forest
9. xgbDecipher_CV.py
Implement XGBoost classifier for Decipher test