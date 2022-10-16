#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/06/18

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.Similarity.Compute_Similarity_Python import Incremental_Similarity_Builder
from Recommenders.BaseMatrixFactorizationRecommender import BaseSVDRecommender
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit
from sklearn.utils.extmath import randomized_svd
import scipy.sparse as sps
import numpy as np
import time



class PureSVDRecommender(BaseSVDRecommender):
    """ PureSVDRecommender
    Formulation with user latent factors and item latent factors

    As described in Section 3.3.1 of the following article:
    Paolo Cremonesi, Yehuda Koren, and Roberto Turrin. 2010.
    Performance of recommender algorithms on top-n recommendation tasks.
    In Proceedings of the fourth ACM conference on Recommender systems (RecSys ’10).
    Association for Computing Machinery, New York, NY, USA, 39–46.
    DOI:https://doi.org/10.1145/1864708.1864721
    """

    RECOMMENDER_NAME = "PureSVDRecommender"

    def __init__(self, URM_train, verbose = True):
        super(PureSVDRecommender, self).__init__(URM_train, verbose = verbose)


    def fit(self, num_factors=100, random_seed = None):

        start_time = time.time()
        self._print("Computing SVD decomposition...")

        U, Sigma, VT = randomized_svd(self.URM_train,
                                      n_components=num_factors,
                                      #n_iter=5,
                                      random_state = random_seed)

        self.USER_factors = U
        self.ITEM_factors = VT.T
        self.Sigma = Sigma

        new_time_value, new_time_unit = seconds_to_biggest_unit(time.time()-start_time)
        self._print("Computing SVD decomposition... done in {:.2f} {}".format( new_time_value, new_time_unit))







def compute_W_sparse_from_item_latent_factors(ITEM_factors, topK = 100):

    n_items, n_factors = ITEM_factors.shape

    block_size = 100
    start_item = 0
    end_item = 0

    similarity_builder = Incremental_Similarity_Builder(n_items, initial_data_block=n_items*topK, dtype = np.float32)

    # Compute all similarities for each item using vectorization
    while start_item < n_items:

        end_item = min(n_items, start_item + block_size)

        this_block_weight = np.dot(ITEM_factors[start_item:end_item, :], ITEM_factors.T)


        for col_index_in_block in range(this_block_weight.shape[0]):

            this_column_weights = this_block_weight[col_index_in_block, :]
            item_original_index = start_item + col_index_in_block

            # Select TopK
            relevant_items_partition = np.argpartition(-this_column_weights, topK-1, axis=0)[0:topK]
            this_column_weights = this_column_weights[relevant_items_partition]

            # Incrementally build sparse matrix, do not add zeros
            if np.any(this_column_weights == 0.0):
                non_zero_mask = this_column_weights != 0.0
                relevant_items_partition = relevant_items_partition[non_zero_mask]
                this_column_weights = this_column_weights[non_zero_mask]

            similarity_builder.add_data_lists(row_list_to_add=relevant_items_partition,
                                              col_list_to_add=np.ones(len(relevant_items_partition), dtype = np.int) * item_original_index,
                                              data_list_to_add=this_column_weights)


        start_item += block_size

    return similarity_builder.get_SparseMatrix()


from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender

class PureSVDItemRecommender(BaseItemSimilarityMatrixRecommender):
    """ PureSVDItemRecommender
    Formulation with the item-item similarity

    As described in Section 3.3.1 of the following article:
    Paolo Cremonesi, Yehuda Koren, and Roberto Turrin. 2010.
    Performance of recommender algorithms on top-n recommendation tasks.
    In Proceedings of the fourth ACM conference on Recommender systems (RecSys ’10).
    Association for Computing Machinery, New York, NY, USA, 39–46.
    DOI:https://doi.org/10.1145/1864708.1864721
    """

    RECOMMENDER_NAME = "PureSVDItemRecommender"

    def __init__(self, URM_train, verbose = True):
        super(PureSVDItemRecommender, self).__init__(URM_train, verbose = verbose)


    def fit(self, num_factors=100, topK = None, random_seed = None):

        self._print("Computing SVD decomposition...")

        U, Sigma, VT = randomized_svd(self.URM_train,
                                      n_components=num_factors,
                                      #n_iter=5,
                                      random_state = random_seed)

        if topK is None:
            topK = self.n_items

        W_sparse = compute_W_sparse_from_item_latent_factors(VT.T, topK=topK)

        self.W_sparse = sps.csr_matrix(W_sparse)

        self._print("Computing SVD decomposition... Done!")



class ScaledPureSVDRecommender(PureSVDRecommender):
    """ ScaledPureSVDRecommender"""

    RECOMMENDER_NAME = "ScaledPureSVDRecommender"

    def __init__(self, URM_train, verbose = True):
        super(ScaledPureSVDRecommender, self).__init__(URM_train, verbose = verbose)


    def fit(self, num_factors = 100, random_seed = None, scaling_items = 1.0, scaling_users = 1.0):

        item_pop = np.ediff1d(sps.csc_matrix(self.URM_train).indptr)
        item_scaling_matrix = sps.diags(np.power(item_pop + 1e-6, scaling_items))

        user_pop = np.ediff1d(sps.csr_matrix(self.URM_train).indptr)
        user_scaling_matrix = sps.diags(np.power(user_pop + 1e-6, scaling_users))

        self.URM_train = user_scaling_matrix.dot(self.URM_train).dot(item_scaling_matrix)

        super(ScaledPureSVDRecommender, self).fit(num_factors = num_factors, random_seed = random_seed)


