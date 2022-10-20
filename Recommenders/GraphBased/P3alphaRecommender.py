#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Cesare Bernardis
"""

import numpy as np
import scipy.sparse as sps

from sklearn.preprocessing import normalize
from Recommenders.Recommender_utils import check_matrix, similarityMatrixTopK
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.Similarity.Compute_Similarity_Python import Incremental_Similarity_Builder
import time, sys




class P3alphaRecommender(BaseItemSimilarityMatrixRecommender):
    """ P3alpha recommender """

    RECOMMENDER_NAME = "P3alphaRecommender"

    def __init__(self, URM_train, verbose = True):
        super(P3alphaRecommender, self).__init__(URM_train, verbose = verbose)


    def __str__(self):
        return "P3alpha(alpha={}, min_rating={}, topk={}, implicit={}, normalize_similarity={})".format(self.alpha,
                                                                            self.min_rating, self.topK, self.implicit,
                                                                            self.normalize_similarity)

    def fit(self, topK=100, alpha=1., min_rating=0, implicit=False, normalize_similarity=False):

        self.topK = topK
        self.alpha = alpha
        self.min_rating = min_rating
        self.implicit = implicit
        self.normalize_similarity = normalize_similarity


        #
        # if X.dtype != np.float32:
        #     print("P3ALPHA fit: For memory usage reasons, we suggest to use np.float32 as dtype for the dataset")

        if self.min_rating > 0:
            self.URM_train.data[self.URM_train.data < self.min_rating] = 0
            self.URM_train.eliminate_zeros()
            if self.implicit:
                self.URM_train.data = np.ones(self.URM_train.data.size, dtype=np.float32)

        #Pui is the row-normalized urm
        Pui = normalize(self.URM_train, norm='l1', axis=1)

        #Piu is the column-normalized, "boolean" urm transposed
        X_bool = self.URM_train.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)
        #ATTENTION: axis is still 1 because i transposed before the normalization
        Piu = normalize(X_bool, norm='l1', axis=1)
        del(X_bool)

        # Alfa power
        if self.alpha != 1.:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        # Final matrix is computed as Pui * Piu * Pui
        # Multiplication unpacked for memory usage reasons
        block_dim = 200
        d_t = Piu

        similarity_builder = Incremental_Similarity_Builder(Pui.shape[1], initial_data_block=Pui.shape[1]*self.topK, dtype = np.float32)

        start_time = time.time()
        start_time_printBatch = start_time

        for current_block_start_row in range(0, Pui.shape[1], block_dim):

            if current_block_start_row + block_dim > Pui.shape[1]:
                block_dim = Pui.shape[1] - current_block_start_row

            similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pui
            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_dim):
                row_data = similarity_block[row_in_block, :]
                row_data[current_block_start_row + row_in_block] = 0

                relevant_items_partition = np.argpartition(-row_data, self.topK-1, axis=0)[:self.topK]
                row_data = row_data[relevant_items_partition]

                # Incrementally build sparse matrix, do not add zeros
                if np.any(row_data == 0.0):
                    non_zero_mask = row_data != 0.0
                    relevant_items_partition = relevant_items_partition[non_zero_mask]
                    row_data = row_data[non_zero_mask]

                similarity_builder.add_data_lists(row_list_to_add=np.ones(len(row_data), dtype = np.int) * (current_block_start_row + row_in_block),
                                                  col_list_to_add=relevant_items_partition,
                                                  data_list_to_add=row_data)


            if time.time() - start_time_printBatch > 300 or current_block_start_row + block_dim == Pui.shape[1]:
                new_time_value, new_time_unit = seconds_to_biggest_unit(time.time() - start_time)

                self._print("Similarity column {} ({:4.1f}%), {:.2f} column/sec. Elapsed time {:.2f} {}".format(
                     current_block_start_row + block_dim,
                    100.0 * float( current_block_start_row + block_dim) / Pui.shape[1],
                    float( current_block_start_row + block_dim) / (time.time() - start_time),
                    new_time_value, new_time_unit))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()


        self.W_sparse = similarity_builder.get_SparseMatrix()


        if self.normalize_similarity:
            self.W_sparse = normalize(self.W_sparse, norm='l1', axis=1)


        if self.topK != False:
            self.W_sparse = similarityMatrixTopK(self.W_sparse, k=self.topK)

        self.W_sparse = check_matrix(self.W_sparse, format='csr')
