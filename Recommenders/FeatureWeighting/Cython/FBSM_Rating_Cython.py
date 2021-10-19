"""
Created on 03/02/2018

@author: Maurizio Ferrari Dacrema
"""


from Recommenders.Similarity.Compute_Similarity import Compute_Similarity
from Recommenders.DataIO import DataIO
from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.Recommender_utils import check_matrix
from Recommenders.IR_feature_weighting import okapi_BM_25, TF_IDF

from CythonCompiler.run_compile_subprocess import run_compile_subprocess

import scipy.sparse as sps
import sys, time
import numpy as np



class FBSM_Rating_Cython(BaseItemCBFRecommender, BaseItemSimilarityMatrixRecommender, Incremental_Training_Early_Stopping):

    RECOMMENDER_NAME = "FBSM_Rating_Cython"

    INIT_TYPE_VALUES = ["random", "one", "BM25", "TF-IDF"]

    def __init__(self, URM_train, ICM):

        super(FBSM_Rating_Cython, self).__init__(URM_train)

        self.n_items_icm, self.n_features = ICM.shape

        self.ICM = check_matrix(ICM, 'csr')





    def fit(self, topK = 300,
            epochs = 30,
            n_factors=2,
            learning_rate=1e-5,
            precompute_user_feature_count = False,
            initialization_mode_D = "random",
            positive_only_D = True,
            positive_only_V = True,
            l2_reg_D=0.01,
            l2_reg_V=0.01,
            non_negative_weights = False,
            verbose = False,
            sgd_mode='adam', gamma = 0.9, beta_1=0.9, beta_2=0.999,
            **earlystopping_kwargs):


        if initialization_mode_D not in self.INIT_TYPE_VALUES:
           raise ValueError("Value for 'initialization_mode_D' not recognized. Acceptable values are {}, provided was '{}'".format(self.INIT_TYPE_VALUES, initialization_mode_D))

        from FeatureWeighting.Cython.FBSM_Rating_Cython_SGD import FBSM_Rating_Cython_SGD

        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.l2_reg_D = l2_reg_D
        self.l2_reg_V = l2_reg_V
        self.topK = topK
        self.epochs = epochs
        self.verbose = verbose

        # For mean_init use Xavier Initialization
        if self.n_factors != 0:
            std_init = 1/self.n_features/self.n_factors
        else:
            std_init = 0

        mean_init = 0


        weights_initialization_D = None

        if initialization_mode_D == "random":
            weights_initialization_D = np.random.normal(0.001, 0.1, self.n_features).astype(np.float64)
        elif initialization_mode_D == "one":
            weights_initialization_D = np.ones(self.n_features, dtype=np.float64)
        elif initialization_mode_D == "zero":
            weights_initialization_D = np.zeros(self.n_features, dtype=np.float64)
        elif initialization_mode_D == "BM25":
            weights_initialization_D = np.ones(self.n_features, dtype=np.float64)
            self.ICM = self.ICM.astype(np.float32)
            self.ICM = okapi_BM_25(self.ICM)

        elif initialization_mode_D == "TF-IDF":
            weights_initialization_D = np.ones(self.n_features, dtype=np.float64)
            self.ICM = self.ICM.astype(np.float32)
            self.ICM = TF_IDF(self.ICM)

        else:
            raise ValueError("CFW_D_Similarity_Cython: 'init_type' not recognized")





        self.FBSM_Rating = FBSM_Rating_Cython_SGD(self.URM_train, self.ICM,
                                                  n_factors = self.n_factors,
                                                  precompute_user_feature_count = precompute_user_feature_count,
                                                  learning_rate = self.learning_rate,
                                                  l2_reg_D = self.l2_reg_D,
                                                  l2_reg_V = self.l2_reg_V,
                                                  weights_initialization_D = weights_initialization_D,
                                                  weights_initialization_V = None,
                                                  positive_only_D = positive_only_D,
                                                  positive_only_V = positive_only_V,
                                                  verbose = self.verbose,
                                                  sgd_mode = sgd_mode,
                                                  gamma = gamma,
                                                  beta_1=beta_1,
                                                  beta_2=beta_2,
                                                  mean_init = mean_init,
                                                  std_init = std_init)







        if self.verbose:
            print(self.RECOMMENDER_NAME + ": Initialization completed")


        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.compute_W_sparse(model_to_use = "best")

        sys.stdout.flush()







    def _prepare_model_for_validation(self):

        self.D_incremental = self.FBSM_Rating.get_D()
        self.V_incremental = self.FBSM_Rating.get_V()

        self.compute_W_sparse(model_to_use = "last")


    def _update_best_model(self):
        self.D_best = self.D_incremental.copy()
        self.V_best = self.V_incremental.copy()


    def _run_epoch(self, num_epoch):
        self.loss = self.FBSM_Rating.fit()





    def set_ICM_and_recompute_W(self, ICM_new, recompute_w = True):

        self.ICM = ICM_new.copy()

        if recompute_w:
            self.compute_W_sparse(use_D = True, use_V = True, model_to_use = "best")




    def compute_W_sparse(self, use_D = True, use_V = True, model_to_use = "best"):

        assert model_to_use in ["last", "best"], "{}: compute_W_sparse, 'model_to_use' parameter not recognized".format(self.RECOMMENDER_NAME)

        if self.verbose:
            print("FBSM_Rating_Cython: Building similarity matrix...")

        start_time = time.time()
        start_time_print_batch = start_time

        # Diagonal
        if use_D:

            if model_to_use == "last":
                D = self.D_incremental
            else:
                D = self.D_best

            similarity = Compute_Similarity(self.ICM.T, shrink=0, topK=self.topK, normalize=False, row_weights=D)
            self.W_sparse = similarity.compute_similarity()
        else:
            self.W_sparse = sps.csr_matrix((self.n_items, self.n_items))


        if use_V:

            if model_to_use == "last":
                V = self.V_incremental
            else:
                V = self.V_best


            # V * V.T
            W1 = self.ICM.dot(V.T)

            #self.W_sparse += W1.dot(W1.T)

            # Use array as it reduces memory requirements compared to lists
            dataBlock = 10000000

            values = np.zeros(dataBlock, dtype=np.float32)
            rows = np.zeros(dataBlock, dtype=np.int32)
            cols = np.zeros(dataBlock, dtype=np.int32)

            numCells = 0


            for numItem in range(self.n_items):

                V_weights = W1[numItem,:].dot(W1.T)
                V_weights[numItem] = 0.0

                relevant_items_partition = (-V_weights).argpartition(self.topK-1)[0:self.topK]
                relevant_items_partition_sorting = np.argsort(-V_weights[relevant_items_partition])
                top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

                # Incrementally build sparse matrix, do not add zeros
                notZerosMask = V_weights[top_k_idx] != 0.0
                numNotZeros = np.sum(notZerosMask)


                values_to_add = V_weights[top_k_idx][notZerosMask]
                rows_to_add = top_k_idx[notZerosMask]
                cols_to_add = np.ones(numNotZeros) * numItem

                for index in range(len(values_to_add)):

                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))


                    rows[numCells] = rows_to_add[index]
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1

                if self.verbose and (time.time() - start_time_print_batch >= 30 or numItem==self.n_items-1):
                    columnPerSec = numItem / (time.time() - start_time)

                    print("Weighted similarity column {} ( {:2.0f} % ), {:.2f} column/sec, elapsed time {:.2f} min".format(
                        numItem, numItem / self.n_items * 100, columnPerSec, (time.time() - start_time)/ 60))

                    sys.stdout.flush()
                    sys.stderr.flush()

                    start_time_print_batch = time.time()


            V_weights = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                              shape=(self.n_items, self.n_items),
                              dtype=np.float32)

            self.W_sparse += V_weights
            self.W_sparse = check_matrix(self.W_sparse, format='csr')



        if self.verbose:
            print("FBSM_Rating_Cython: Building similarity matrix... complete")



    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        data_dict_to_save = {
            "D_best": self.D_best,
            "V_best": self.V_best,
            "topK":self.topK,
            "W_sparse":self.W_sparse
        }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        print("{}: Saving complete".format(self.RECOMMENDER_NAME))

