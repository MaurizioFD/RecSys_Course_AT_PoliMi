# RecSys Course 2018
This is the official repository for the 2018 Recommender Systems course at Polimi.

#### This repo contains a Cython implementation of:
 - SLIM BPR: Item-item similarity matrix machine learning algorithm optimizing BPR.
    Uses a Cython tree-based sparse matrix, suitable for datasets whose number of items is too big for the
    dense similarity matrix to fit in memory. 
    Dense similarity is also supported.
 - MF BPR: Matrix factorization optimizing BPR
 - FunkSVD: Matrix factorization optimizing RMSE
 - AsymmetricSVD (older code)

#### This repo contains a Python implementation of:
 - Item-based KNN collaborative
 - Item-based KNN content
 - User-based KNN
 - PureSVD: Matrix factorization applied using the simple SVD decomposition of the URM
 - P3alpha, RP3beta: graph based algorithms modelling a random walk and representing the item-item similarity as a transition probability
 - SLIM ElasticNet Item-item similarity matrix machine learning algorithm optimizing prediction error (MSE)
 
 
#### Bayesian parameter tuning:
A simple wrapper of another library ( https://github.com/fmfn/BayesianOptimization ) allowing for a simple and fast parameter tuning.
The BayesianSearch object will save the following files:
- AlgorithmName_BayesianSearch.txt file with all the cases explored and the recommendation quality
- _best_model file which contains the trained model and can be loaded with recommender.loadModel(path_to_best_model_file)
- _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**dictionary_best_parameter)
- _best_result_validation file which contains a dictionary with the results of the best solution on the validation
- _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set

 
#### This repo contains the following runnable scripts

 - run_all_algorithms.py: Script running sequentially all available algorithms and saving the results in result_all_algorithms.txt
 - run_parameter_search.py: Script performing parameter tuning for all available algorithms. Inside all parameters are listed with some common values.
 

#### This repo also provides an implementation of:
 
 - Similarities: Cosine Similarity, Adjusted Cosine, Pearson Correlation, Jaccard Correlation, Tanimoto Coefficient, Dice coefficinent, Tversky coefficient and Asymmetric Cosine: Implemented both in Python and Cython with the same interface. Base.compute_similarity chooses which to use depending on the density of the data and on whether a compiled cython version is available on your architecture and operative system. 
 - Metrics: MAP, recall (the denominator is the number of user's test items), recall_min_test_len (the denominator is the min between the number of user's test items and the recommendation list length), precision, ROC-AUC, MRR, RR, NDCG, Hit Rate, ARHR, Novelty, Coverage, Shannon entropy, Gini Diversity, Herfindahl Diversity, Mean inter list Diversity, Feature based diversity
 - Dataset: Movielens10MReader, downloads and reads the Movielens 10M rating file, splits it into three URMs for train, test and validation and saves them for later use. 
 

Cython code is already compiled for Linux x86 (your usual personal computer architecture) and ppc64 (IBM Power PC). To recompile the code just set the recompile_cython flag to True.
For other OS such as Windows the c-imported numpy interface might be different (e.g. return tipe long long insead of long) therefore the code could require modifications in oder to compile.

