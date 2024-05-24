# Darth-Imputer

<!-- Brain Data: CSV file containing the gene expression levels of 54676 genes (columns) from 130 samples (rows). There are 4 different types of brain cancer (plus healthy tissue) represented in this dataset (column "type"). More information about this dataset, as well as other file formats such as TAB and ARFF, data visualization, and classification and clustering benchmarks are freely available at the official CuMiDa website under the id GSE50161: http://sbcb.inf.ufrgs.br/cumida

Breast Data: CSV file containing the gene expression levels of 54676 genes (columns) from 151 samples (rows). There are 5 different types of breast cancer (plus healthy tissue) represented in this dataset (column "type"). More information about this dataset, as well as other file formats such as TAB and ARFF, data visualization, and classification and clustering benchmarks are freely available at the official CuMiDa website under the id GSE45827: http://sbcb.inf.ufrgs.br/cumida -->

# Datasets

There are three datasets used for this project. All three datasets are from the UCI database.

1. [Breast Cancer Dataset](https://archive.ics.uci.edu/dataset/14/breast+cancer)
2. [National Poll on Healthy Aging](https://archive.ics.uci.edu/dataset/936/national+poll+on+healthy+aging+(npha))
3. [Fertility](https://archive.ics.uci.edu/dataset/244/fertility)

# File structure
The directories should be in the following format: 
- src
    - sample_based
- data
    - aging 
    - binary 
    - breast_cancer

# Running the code
There are three jupyter notebooks that should be used to run the code. The other python files in src contain
functions that will be called upon in the jupyter notebooks.

The locations of the notebooks are as follows:
1. src/AFVA.ipynb
2. src/Uncertainty_v2.ipynb
3. src/sample_based/run_simulations.ipynb