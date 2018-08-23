# RecSys Course 2017
This is the official repository for the 2018 Recommender Systems course at Polimi.

#### This repo contains a Cython implementation of:
 - SLIM BPR: Uses a Cython tree-based sparse matrix, suitable for datasets whose number of items is too big for the
    dense similarity matrix to fit in memory. 
    Dense similarity is also supported.
 - MF BPR: Matrix factorization optimizing BPR
 - FunkSVD: Matrix factorization optimizing RMSE
 - AsymmetricSVD

#### This repo contains a Python implementation of:
 - Item-based KNN collaborative
 - Item-based KNN content
 - User-based KNN
 - P3alpha
 - RerankedP3beta
 - SLIM_ElasticNet: SLIM solver using ElasticNet. The solver fits every column in the similarity matrix independently

#### This repo also provides an implementation of:
 
 - Cosine Similarity, Adjusted Cosine, Pearson Correlation, Jaccard Correlation, Tanimoto Coefficient: Implemented both in Python and Cython with the same interface, Base.cosine_similarity and Base.Cython.cosine_similarity
 - MAP, recall, precision, ROC-AUC, MRR, RR, NDCG, Hit Rate, ARHR, Novelty, Coverage, Diversity to be used in testing
 - Movielens10MReader: reads movielens 10M rating file, splits it into three URMs for train, test and validation. 
 

Cython code is already compiled for Linux. To recompile the code just set the recompile_cython flag to True.
For other OS such as Windows the c-imported numpy interface might be different (e.g. return tipe long long insead of long) therefore the code could require modifications in oder to compile.


##### In "all_algorithms.py" you can see how to use every model.