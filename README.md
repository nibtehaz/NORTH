# NORTH : 
## a highly accurate and scalable Naive Bayes based ORTHologous gene cluster prediction algorithm

This repository contains the original implementation of NORTH, a machine learning based tool to predict ortholog clusters.

NORTH works on predefined ortholog clusters. A Multinomial Naive Bayes model is trained to predict the ortholog clusters, drawing inspiration from typical BLAST based pipelines. 

However, being trained on a set of predefined clusters, cases may occur when we are faced with a gene out of those predefined clusters. To overcome this issue NORTH also has a robust outlier detection method.

NORTH has been able to outperform the existing methods in the OrthoBench benchmark, when evaluated using a stratified 5-fold cross validation test.

## Publication

[Read the preprint](https://www.biorxiv.org/content/10.1101/528323v2)

## Codes

The codes for NORTH are written in python and can be found [here](https://github.com/nibtehaz/NORTH/tree/master/codes)


## Requirements


> * Numpy
> * Scipy
> * BioPython
> * Scikit-learn
> * tqdm
> * Matplotlib
> * Seaborn

## Pipeline Walkthrough

The NORTH pipeline is demonstrated in the following Jupyter Notebook

[NORTH Pipeline](https://github.com/nibtehaz/NORTH/blob/master/codes/NORTH.ipynb)