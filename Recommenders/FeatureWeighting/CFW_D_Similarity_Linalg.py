#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/09/17

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.Recommender_utils import check_matrix
from Recommenders.Similarity.Compute_Similarity import Compute_Similarity
from scipy.sparse import linalg
import time, sys
import numpy as np
from Recommenders.DataIO import DataIO



class CFW_D_Similarity_Linalg(BaseItemCBFRecommender, BaseItemSimilarityMatrixRecommender):

    RECOMMENDER_NAME = "CFW_D_Similarity_Linalg"



    def __init__(self, URM_train, ICM_train, S_matrix_target):
        super(CFW_D_Similarity_Linalg, self).__init__(URM_train, ICM_train)

        if (URM_train.shape[1] != ICM_train.shape[0]):
            raise ValueError("Number of items not consistent. URM contains {} but ICM contains {}".format(URM_train.shape[1],
                                                                                                          ICM_train.shape[0]))

        if(S_matrix_target.shape[0] != S_matrix_target.shape[1]):
            raise ValueError("Items imilarity matrix is not square: rows are {}, columns are {}".format(S_matrix_target.shape[0],
                                                                                                        S_matrix_target.shape[1]))

        if(S_matrix_target.shape[0] != ICM_train.shape[0]):
            raise ValueError("Number of items not consistent. S_matrix contains {} but ICM contains {}".format(S_matrix_target.shape[0],
                                                                                                          ICM_train.shape[0]))

        self.S_matrix_target = check_matrix(S_matrix_target, 'csr')

        self.ICM = check_matrix(ICM_train, 'csr')
        self.n_features = self.ICM.shape[1]



    def fit(self, show_max_performance = False,
            loss_tolerance = 1e-6,
            iteration_limit = 50000,
            damp_coeff=0.0,
            topK = 300,
            add_zeros_quota = 0.0,
            normalize_similarity = False):

        self.show_max_performance = show_max_performance
        self.add_zeros_quota = add_zeros_quota
        self.normalize_similarity = normalize_similarity
        self.topK = topK

        self._generate_train_data()

        commonFeatures = self.ICM[self.row_list].multiply(self.ICM[self.col_list])

        linalg_result = linalg.lsqr(commonFeatures,
                                    self.data_list,
                                    show = False,
                                    atol=loss_tolerance,
                                    btol=loss_tolerance,
                                    iter_lim = iteration_limit,
                                    damp=damp_coeff)

        # res = linalg.lsmr(commonFeatures, self.data_list, show = False, atol=loss_tolerance, btol=loss_tolerance,
        #                   maxiter = iteration_limit, damp=damp_coeff)

        self.D_best = linalg_result[0].copy()
        self.loss = linalg_result[3]

        self.compute_W_sparse()


    def _generate_train_data(self):

        if self.verbose:
            print(self.RECOMMENDER_NAME + ": Generating train data")

        start_time_batch = time.time()


        # Here is important only the structure
        self.similarity = Compute_Similarity(self.ICM.T, shrink=0, topK=self.topK, normalize=False)
        S_matrix_contentKNN = self.similarity.compute_similarity()
        S_matrix_contentKNN = check_matrix(S_matrix_contentKNN, "csr")


        self._print("Collaborative S density: {:.2E}, nonzero cells {}".format(
            self.S_matrix_target.nnz/self.S_matrix_target.shape[0]**2, self.S_matrix_target.nnz))

        self._print("Content S density: {:.2E}, nonzero cells {}".format(
            S_matrix_contentKNN.nnz/S_matrix_contentKNN.shape[0]**2, S_matrix_contentKNN.nnz))


        if self.normalize_similarity:

            # Compute sum of squared
            sum_of_squared_features = np.array(self.ICM.T.power(2).sum(axis=0)).ravel()
            sum_of_squared_features = np.sqrt(sum_of_squared_features)



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

                    new_data_value = self.S_matrix_target[row_index, col_index]

                    if self.normalize_similarity:
                        new_data_value *= sum_of_squared_features[row_index]*sum_of_squared_features[col_index]

                    self.data_list[num_samples] = new_data_value

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


        self._print("Content S structure has {} out of {} ({:4.1f}%) nonzero collaborative cells".format(
            num_common_coordinates, S_matrix_contentKNN.nnz, num_common_coordinates/S_matrix_contentKNN.nnz*100))



        # Discard extra cells at the left of the array
        self.row_list = self.row_list[:num_samples]
        self.col_list = self.col_list[:num_samples]
        self.data_list = self.data_list[:num_samples]


        data_nnz = sum(np.array(self.data_list)!=0)
        data_sum = sum(self.data_list)

        collaborative_nnz = self.S_matrix_target.nnz
        collaborative_sum = sum(self.S_matrix_target.data)

        self._print("Nonzero collaborative cell sum is: {:.2E}, average is: {:.2E}, "
                      "average over all collaborative data is {:.2E}".format(
                      data_sum, data_sum/data_nnz, collaborative_sum/collaborative_nnz))


    def compute_W_sparse(self):

        self.similarity = Compute_Similarity(self.ICM.T, shrink=0, topK=self.topK,
                                            normalize=self.normalize_similarity, row_weights=self.D_best)

        self.W_sparse = self.similarity.compute_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')



    def save_model(self, folder_path, file_name = None):


        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        data_dict_to_save = {
            "D_best":self.D_best,
            "topK":self.topK,
            "W_sparse":self.W_sparse,
            "normalize_similarity":self.normalize_similarity
        }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)


        print("{}: Saving complete".format(self.RECOMMENDER_NAME))






