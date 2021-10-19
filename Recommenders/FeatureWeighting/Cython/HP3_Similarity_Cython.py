#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/09/17

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.Recommender_utils import check_matrix
from Recommenders.DataIO import DataIO
from CythonCompiler.run_compile_subprocess import run_compile_subprocess

import time, sys
import numpy as np
import scipy.sparse as sps

from sklearn.preprocessing import normalize



class HP3_Similarity_Cython(BaseItemCBFRecommender, BaseItemSimilarityMatrixRecommender, Incremental_Training_Early_Stopping):

    RECOMMENDER_NAME = "HP3_Similarity_Cython"

    INIT_VALUE = 1e-08

    def __init__(self, URM_train, ICM, S_matrix_target):


        super(HP3_Similarity_Cython, self).__init__(URM_train)

        if (URM_train.shape[1] != ICM.shape[0]):
            raise ValueError("Number of items not consistent. URM contains {} but ICM contains {}".format(URM_train.shape[1],
                                                                                                          ICM.shape[0]))

        if(S_matrix_target.shape[0] != S_matrix_target.shape[1]):
            raise ValueError("Items imilarity matrix is not square: rows are {}, columns are {}".format(S_matrix_target.shape[0],
                                                                                                        S_matrix_target.shape[1]))

        if(S_matrix_target.shape[0] != ICM.shape[0]):
            raise ValueError("Number of items not consistent. S_matrix contains {} but ICM contains {}".format(S_matrix_target.shape[0],
                                                                                                          ICM.shape[0]))

        self.S_matrix_target = check_matrix(S_matrix_target, 'csr')
        self.ICM = check_matrix(ICM, 'csr')

        self.n_features = self.ICM.shape[1]


        self.D_incremental = np.ones(self.n_features, dtype=np.float64)
        self.D_best = self.D_incremental.copy()



    def fit(self, show_max_performance = False,
            log_file = None,
            precompute_common_features = True,
            learning_rate = 1e-08,
            init_value = 1e-08,
            use_dropout = True,
            dropout_perc = 0.3,
            l1_reg = 0.0,
            l2_reg = 0.0,
            epochs = 50,
            topK = 300,
            verbose = False,
            add_zeros_quota = 0.0,
            sgd_mode = 'adagrad', gamma = 0.9, beta_1 = 0.9, beta_2 = 0.999,
            **earlystopping_kwargs):


        if init_value <= 0:
            init_value = self.INIT_VALUE
            print(self.RECOMMENDER_NAME + ": Invalid init value, using default (" + str(self.INIT_VALUE) + ")")

        # Import compiled module
        from FeatureWeighting.Cython.HP3_Similarity_Cython_SGD import HP3_Similarity_Cython_SGD

        self.log_file = log_file
        self.show_max_performance = show_max_performance
        self.learning_rate = learning_rate
        self.add_zeros_quota = add_zeros_quota
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.topK = topK
        self.verbose = verbose

        self._generate_train_data()


        weights_initialization_D = np.ones(self.n_features, dtype=np.float64) * init_value

        # Instantiate fast Cython implementation
        self.HP3_Similarity = HP3_Similarity_Cython_SGD(self.row_list, self.col_list, self.data_list,
                                                 self.n_features,
                                                 self.ICM,
                                                 simplify_model=True,
                                                 precompute_common_features = precompute_common_features,
                                                 weights_initialization_D = weights_initialization_D,
                                                 use_dropout = use_dropout,
                                                 dropout_perc = dropout_perc,
                                                 learning_rate=learning_rate,
                                                 l1_reg=l1_reg,
                                                 l2_reg=l2_reg,
                                                 sgd_mode=sgd_mode,
                                                 gamma = gamma, beta_1=beta_1, beta_2=beta_2)


        if self.verbose:
            print(self.RECOMMENDER_NAME + ": Initialization completed")

        self.D_incremental = self.HP3_Similarity.get_weights()
        self.D_best = self.D_incremental.copy()


        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.compute_W_sparse(model_to_use = "best")

        sys.stdout.flush()



    def _prepare_model_for_validation(self):
        self.D_incremental = self.HP3_Similarity.get_weights()
        self.compute_W_sparse(model_to_use = "last")


    def _update_best_model(self):
        self.D_best = self.D_incremental.copy()


    def _run_epoch(self, num_epoch):
        self.loss = self.HP3_Similarity.fit()






    def _generate_train_data(self):

        if self.verbose:
            print(self.RECOMMENDER_NAME + ": Generating train data")

        start_time_batch = time.time()

        # Here is important only the structure
        self.compute_W_sparse()
        S_matrix_contentKNN = check_matrix(self.W_sparse, "csr")


        self.write_log("Collaborative S density: {:.2E}, nonzero cells {}".format(
            self.S_matrix_target.nnz/self.S_matrix_target.shape[0]**2, self.S_matrix_target.nnz))

        self.write_log("Content S density: {:.2E}, nonzero cells {}".format(
            S_matrix_contentKNN.nnz/S_matrix_contentKNN.shape[0]**2, S_matrix_contentKNN.nnz))


        num_common_coordinates = 0

        estimated_n_samples = int(S_matrix_contentKNN.nnz*(1+self.add_zeros_quota)*1.2)

        self.row_list = np.zeros(estimated_n_samples, dtype=np.int32)
        self.col_list = np.zeros(estimated_n_samples, dtype=np.int32)
        self.data_list = np.zeros(estimated_n_samples, dtype=np.float64)

        num_samples = 0


        for row_index in range(self.n_items):

            start_pos_content = S_matrix_contentKNN.indptr[row_index]
            end_pos_content = S_matrix_contentKNN.indptr[row_index+1]

            content_coordinates = S_matrix_contentKNN.indices[start_pos_content:end_pos_content]

            start_pos_target = self.S_matrix_target.indptr[row_index]
            end_pos_target = self.S_matrix_target.indptr[row_index+1]

            target_coordinates = self.S_matrix_target.indices[start_pos_target:end_pos_target]

            # Chech whether the content coordinate is associated to a non zero target value
            # If true, the content coordinate has a collaborative non-zero value
            # if false, the content coordinate has a collaborative zero value
            is_common = np.in1d(content_coordinates, target_coordinates)

            num_common_in_current_row = is_common.sum()
            num_common_coordinates += num_common_in_current_row


            for index in range(len(is_common)):

                if num_samples == estimated_n_samples:
                    dataBlock = 1000000
                    self.row_list = np.concatenate((self.row_list, np.zeros(dataBlock, dtype=np.int32)))
                    self.col_list = np.concatenate((self.col_list, np.zeros(dataBlock, dtype=np.int32)))
                    self.data_list = np.concatenate((self.data_list, np.zeros(dataBlock, dtype=np.float64)))

                if is_common[index]:
                    # If cell exists in target matrix, add its value
                    # Otherwise it will remain zero with a certain probability

                    col_index = content_coordinates[index]

                    self.row_list[num_samples] = row_index
                    self.col_list[num_samples] = col_index
                    self.data_list[num_samples] = self.S_matrix_target[row_index, col_index]

                    num_samples += 1


                elif np.random.rand() <= self.add_zeros_quota:

                    col_index = content_coordinates[index]

                    self.row_list[num_samples] = row_index
                    self.col_list[num_samples] = col_index
                    self.data_list[num_samples] = 0.0

                    num_samples += 1



            if self.verbose and (time.time() - start_time_batch > 30 or num_samples == S_matrix_contentKNN.nnz*(1+self.add_zeros_quota)):

                print(self.RECOMMENDER_NAME + ": Generating train data. Sample {} ({:4.1f}%) ".format(
                    num_samples, num_samples/ S_matrix_contentKNN.nnz*(1+self.add_zeros_quota) *100))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_batch = time.time()


        self.write_log("Content S structure has {} out of {} ({:4.1f}%) nonzero collaborative cells".format(
            num_common_coordinates, S_matrix_contentKNN.nnz, num_common_coordinates/S_matrix_contentKNN.nnz*100))



        # Discard extra cells at the left of the array
        self.row_list = self.row_list[:num_samples]
        self.col_list = self.col_list[:num_samples]
        self.data_list = self.data_list[:num_samples]


        data_nnz = sum(np.array(self.data_list)!=0)
        data_sum = sum(self.data_list)

        collaborative_nnz = self.S_matrix_target.nnz
        collaborative_sum = sum(self.S_matrix_target.data)

        self.write_log("Nonzero collaborative cell sum is: {:.2E}, average is: {:.2E}, "
                      "average over all collaborative data is {:.2E}".format(
                      data_sum, data_sum/data_nnz, collaborative_sum/collaborative_nnz))

    #     if self.URM_validation is not None and self.show_max_performance:
    #         self.computeMaxTheoreticalPerformance()
    #
    #
    #
    #
    # def computeMaxTheoreticalPerformance(self):
    #
    #     # Max performance would be if we were able to learn the content matrix having for each non-zero cell exactly
    #     # the value that appears in the collaborative similarity
    #
    #     print(self.RECOMMENDER_NAME + ": Computing collaborative performance")
    #
    #     recommender = ItemKNNCustomSimilarityRecommender()
    #     recommender.fit(self.S_matrix_target, self.URM_train)
    #
    #     results_run = recommender.evaluateRecommendations(self.URM_validation)
    #
    #     self.write_log(self.RECOMMENDER_NAME + ": Collaborative performance is: {}".format(results_run))
    #
    #
    #     print(self.RECOMMENDER_NAME + ": Computing top structural performance")
    #
    #     n_items = self.ICM.shape[0]
    #
    #     S_optimal = sps.csr_matrix((self.data_list, (self.row_list, self.col_list)), shape=(n_items, n_items))
    #     S_optimal.eliminate_zeros()
    #
    #     recommender = ItemKNNCustomSimilarityRecommender()
    #     recommender.fit(S_optimal, self.URM_train)
    #
    #     results_run = recommender.evaluateRecommendations(self.URM_validation)
    #
    #     self.write_log(self.RECOMMENDER_NAME + ": Top structural performance is: {}".format(results_run))



    def write_log(self, string):
        string = self.RECOMMENDER_NAME + ": " + string

        if self.verbose:
            print(string)
            sys.stdout.flush()
            sys.stderr.flush()

        if self.log_file is not None:
            self.log_file.write(string + "\n")
            self.log_file.flush()



    def compute_W_sparse(self, model_to_use = "best"):

        if model_to_use == "last":
            feature_weights = self.D_incremental
        elif model_to_use == "best":
            feature_weights = self.D_best
        else:
            assert False, "{}: compute_W_sparse, 'model_to_use' parameter not recognized".format(self.RECOMMENDER_NAME)

        block_dim = 300
        d_t = self.ICM * sps.diags([feature_weights.squeeze()], [0])
        icm_t = self.ICM.astype(np.bool).T
        indptr, indices, data = [0], [], []
        for r in range(0, self.n_items, block_dim):
            if r + block_dim > self.n_items:
                block_dim = self.n_items - r
            sim = d_t[r:r + block_dim, :] * icm_t
            for s in range(block_dim):
                row = sim[s].toarray().ravel()
                row[r + s] = 0
                best = row.argsort()[::-1][:self.topK]
                indices.extend(best)
                indptr.append(len(indices))
                data.extend(row[best].flatten().tolist())

        self.W_sparse = normalize(sps.csr_matrix((data, indices, indptr), shape=(self.n_items, self.n_items)), norm="l1", axis=1)
        self.W_sparse = check_matrix(self.W_sparse, format='csr')



    def set_cold_start_items(self, ICM_cold):

        self.ICM = ICM_cold.copy()
        self.compute_W_sparse()


    def runCompilationScript(self):

        # Run compile script setting the working directory to ensure the compiled file are contained in the
        # appropriate subfolder and not the project root

        file_subfolder = "/FW_Similarity/Cython"
        file_to_compile_list = ['HP3_Similarity_Cython_SGD.pyx']


        run_compile_subprocess(file_subfolder, file_to_compile_list)

        print("{}: Compiled module {} in subfolder: {}".format(self.RECOMMENDER_NAME, file_to_compile_list, file_subfolder))

        # Command to run compilation script
        # python compile_script.py HP3_Similarity_Cython_SGD.pyx build_ext --inplace

        # Command to generate html report
        # cython -a HP3_Similarity_Cython_SGD.pyx





    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        data_dict_to_save = {
            "D_best":self.D_best,
            "topK":self.topK,
            "W_sparse":self.W_sparse
        }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)


        print("{}: Saving complete".format(self.RECOMMENDER_NAME))




