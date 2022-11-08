# RecSys Course @ Politecnico di Milano
This is the official repository for the Recommender Systems course at Polimi.

Developed by <a href="https://mauriziofd.github.io/" target="_blank">Maurizio Ferrari Dacrema</a>, Postdoc researcher at Politecnico di Milano. 
See the websites of our [Recommender Systems Group](http://recsys.deib.polimi.it/) and our [Quantum Computing Group](https://quantum.polimi.it/) for more information on our team, thesis and research activities.
The introductory slides are available [here](slides/Introduction%20and%20Materials%20for%20RecSys%20Practice%20Sessions.pdf). 
For Installation instructions see the following section [Installation](#Installation).


#### This repository contains a Cython implementation of:
 - SLIM BPR: Item-item similarity matrix machine learning algorithm optimizing BPR.
    Uses a Cython tree-based sparse matrix, suitable for datasets whose number of items is too big for the
    dense similarity matrix to fit in memory. 
    Dense similarity is also supported.
 - MF BPR: Matrix factorization optimizing BPR
 - FunkSVD: Matrix factorization optimizing RMSE
 - AsymmetricSVD

#### This repo contains a Python implementation of:
 - Item-based KNN collaborative
 - Item-based KNN content
 - User-based KNN
 - PureSVD: Matrix factorization applied using the simple SVD decomposition of the URM
 - WRMF or IALS: Matrix factorization developed for implicit interactions (Papers: <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.306.4684&rep=rep1&type=pdf" target="_blank">WRMF</a>, <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.5120&rep=rep1&type=pdf" target="_blank">IALS</a>)
 - P3alpha, RP3beta: graph based algorithms modelling a random walk and representing the item-item similarity as a transition probability (Papers: <a href="https://dl.acm.org/doi/abs/10.1145/2567948.2579244" target="_blank">P3alpha</a>, <a href="https://dl.acm.org/doi/10.1145/2955101" target="_blank">RP3beta</a>)
 - SLIM ElasticNet Item-item similarity matrix machine learning algorithm optimizing prediction error (MSE)
 
 
#### Bayesian parameter tuning:
A simple wrapper of scikit-optimize allowing for a simple and fast parameter tuning.
The BayesianSkoptSearch object will save the following files:
- AlgorithmName_BayesianSkoptSearch.txt file with all the cases explored and the recommendation quality
- _best_model file which contains the trained model and can be loaded with recommender.load_model(path_to_best_model_file)
- _metadata file which contains a dictionary with all the explored cases, for each the fit parameters, the validation results and, if that configuration was the new best one, the test results. It also contains, for all configurations, the train, validation and test time, in seconds.
 
#### This repository contains the following runnable scripts

 - run_all_algorithms.py: Script running sequentially all available algorithms and saving the results in result_all_algorithms.txt
 - run_parameter_search.py: Script performing parameter tuning for all available algorithms. Inside all parameters are listed with some common values.
 

#### This repository also provides an implementation of:
 
 - Similarities: Cosine Similarity, Adjusted Cosine, Pearson Correlation, Jaccard Correlation, Tanimoto Coefficient, Dice coefficinent, Tversky coefficient, Asymmetric Cosine and Euclidean similarity: Implemented both in Python and Cython with the same interface. Base.compute_similarity chooses which to use depending on the density of the data and on whether a compiled cython version is available on your architecture and operative system. 
 - Metrics: MAP, recall (the denominator is the number of user's test items), precision_recall_min_den (the denominator is the min between the number of user's test items and the recommendation list length), precision, ROC-AUC, MRR, RR, NDCG, Hit Rate, ARHR, Novelty, Coverage, Shannon entropy, Gini Diversity, Herfindahl Diversity, Mean inter list Diversity, Feature based diversity
 - Dataset: Movielens10MReader, downloads and reads the Movielens 10M rating file, splits it into three URMs for train, test and validation and saves them for later use. 
 

Cython code is already compiled for Linux and Windows x86 (your usual personal computer architecture) and ppc64 (IBM Power PC). To recompile the code just run the cython compilaton script as described in the installation section.
The code has beend developed for Linux and Windows.




## Installation

Note that this repository requires Python 3.8

First we suggest you create an environment for this project using conda

First checkout this repository, then enter in the repository folder and run this commands to create and activate a new environment:

```console
conda create -n RecSysFramework python=3.8 anaconda
conda activate RecSysFramework
```

Then install all the requirements and dependencies using the following command.
```console
pip install -r requirements.txt
```

At this point you have to compile all Cython algorithms.
In order to compile you must first have installed: _gcc_ and _python3 dev_. Under Linux those can be installed with the following commands:
```console
sudo apt install gcc 
sudo apt-get install python3-dev
```
If you are using Windows as operating system, the installation procedure is a bit more complex. You may refer to [THIS](https://github.com/cython/cython/wiki/InstallingOnWindows) guide.

Now you can compile all Cython algorithms by running the following command. The script will compile within the current active environment. The code has been developed for Linux and Windows platforms. During the compilation you may see some warnings. 
 
```console
python run_compile_all_cython.py
```

If you are importing this repository on a Kaggle notebook, try to compile like this:
```console
!git clone https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi
cd RecSys_Course_AT_PoliMi
!python run_compile_all_cython.py
```


## Project structure

### Base
Contains some basic modules and the base classes for different Recommender types.

#### Base.Evaluation
The Evaluator class is used to evaluate a recommender object. It computes various metrics:
* Accuracy metrics: ROC_AUC, PRECISION, RECALL, MAP, MRR, NDCG, F1, HIT_RATE, ARHR
* Beyond-accuracy metrics: NOVELTY, DIVERSITY, COVERAGE

The evaluator takes as input the URM against which you want to test the recommender, then a list of cutoff values (e.g., 5, 20) and, if necessary, an object to compute diversity.
The function evaluateRecommender will take as input only the recommender object you want to evaluate and return both a dictionary in the form {cutoff: results}, where results is {metric: value} and a well-formatted printable string.

```python

    from Base.Evaluation.Evaluator import EvaluatorHoldout

    evaluator_test = EvaluatorHoldout(URM_test, [5, 20])

    results_run_dict, results_run_string = evaluator_test.evaluateRecommender(recommender_instance)

    print(results_run_string)

```


#### Base.Similarity
The similarity module allows to compute the item-item or user-user similarity.
It is used by calling the Compute_Similarity class and passing which is the desired similarity and the sparse matrix you wish to use.

It is able to compute the following similarities: Cosine, Adjusted Cosine, Jaccard, Tanimoto, Pearson and Euclidean (linear and exponential)

```python

    similarity = Compute_Similarity(URM_train, shrink=shrink, topK=topK, normalize=normalize, similarity = "cosine")

    W_sparse = similarity.compute_similarity()

```


### Recommenders
All recommenders inherit from BaseRecommender, therefore have the same interface.
You must provide the data when instantiating the recommender and then call the _fit_ function to build the corresponding model.

Each recommender has a _compute_item_score function which, given an array of user_id, computes the prediction or _score_ for all items.
Further operations like removing seen items and computing the recommendation list of the desired length are done by the _recommend_ function of BaseRecommender

As an example:

```python
    user_id = 158
    
    recommender_instance = ItemKNNCFRecommender(URM_train)
    recommender_instance.fit(topK=150)
    recommended_items = recommender_instance.recommend(user_id, cutoff = 20, remove_seen_flag=True)
    
    recommender_instance = SLIM_ElasticNet(URM_train)
    recommender_instance.fit(topK=150, l1_ratio=0.1, alpha = 1.0)
    recommended_items = recommender_instance.recommend(user_id, cutoff = 20, remove_seen_flag=True)
    
```

### Data Reader and splitter
DataReader objects read the dataset from its original file and save it as a sparse matrix.

DataSplitter objects take as input a DataReader and split the corresponding dataset in the chosen way.
At each step the data is automatically saved in a folder, though it is possible to prevent this by setting _save_folder_path = False_ when calling _load_data_.
If a DataReader or DataSplitter is called for a dataset which was already processed, the saved data is loaded.

DataPostprocessing can also be applied between the dataReader and the dataSplitter and nested in one another.

When you have bilt the desired combination of dataset/preprocessing/split, get the data calling _load_data_.

```python
dataset = Movielens1MReader()

dataset = DataPostprocessing_K_Cores(dataset, k_cores_value=25)
dataset = DataPostprocessing_User_sample(dataset, user_quota=0.3)
dataset = DataPostprocessing_Implicit_URM(dataset)

dataSplitter = DataSplitter_leave_k_out(dataset)

dataSplitter.load_data()

URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()
