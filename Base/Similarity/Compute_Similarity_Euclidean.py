#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import time, sys
import scipy.sparse as sps
from Base.Recommender_utils import check_matrix


class Compute_Similarity_Euclidean:


    def __init__(self, dataMatrix, topK=100, shrink = 0,
                 similarity_from_distance_mode ="linear", row_weights = None, **args):
        """
        Computes the euclidean similarity on the columns of dataMatrix
        If it is computed on URM=|users|x|items|, pass the URM as is.
        If it is computed on ICM=|items|x|features|, pass the ICM transposed.
        :param dataMatrix:
        :param topK:
        :param row_weights:         Multiply the values in each row by a specified value. Array
        :param similarity_from_distance_mode:       "exponential"   euclidean_similarity = 1/(e ^ euclidean_distance)
                                                    "linear"        euclidean_similarity = 1/(1 + euclidean_distance)
        :param args:                accepts other parameters not needed by the current object

        """

        super(Compute_Similarity_Euclidean, self).__init__()

        self.TopK = topK
        self.shrink = shrink
        self.n_columns = dataMatrix.shape[1]
        self.n_rows = dataMatrix.shape[0]

        self.dataMatrix = dataMatrix.copy()

        self.similarity_is_exponential = False
        self.similarity_is_linear = False

        if similarity_from_distance_mode == "exponential":
            self.similarity_is_exponential = True
        elif similarity_from_distance_mode == "linear":
            self.similarity_is_linear = True
        else:
            raise ValueError("Compute_Similarity_Euclidean: value for paramether 'mode' not recognized."
                             " Allowed values are: 'exponential', 'linear'."
                             " Passed value was '{}'".format(similarity_from_distance_mode))



        if self.TopK == 0:
            self.W_dense = np.zeros((self.n_columns, self.n_columns))


        self.use_row_weights = False

        if row_weights is not None:

            if dataMatrix.shape[0] != len(row_weights):
                raise ValueError("Compute_Similarity_Euclidean: provided row_weights and dataMatrix have different number of rows."
                                 "row_weights has {} rows, dataMatrix has {}.".format(len(row_weights), dataMatrix.shape[0]))

            self.use_row_weights = True
            self.row_weights = row_weights.copy()
            self.row_weights_diag = sps.diags(self.row_weights)

            self.dataMatrix_weighted = self.dataMatrix.T.dot(self.row_weights_diag).T








    def compute_similarity(self, start_col=None, end_col=None):
        """
        Compute the similarity for the given dataset
        :param self:
        :param start_col: column to begin with
        :param end_col: column to stop before, end_col is excluded
        :return:
        """

        values = []
        rows = []
        cols = []

        start_time = time.time()
        start_time_print_batch = start_time
        processedItems = 0


        self.dataMatrix = self.dataMatrix.toarray()

        start_col_local = 0
        end_col_local = self.n_columns

        if start_col is not None and start_col>0 and start_col<self.n_columns:
            start_col_local = start_col

        if end_col is not None and end_col>start_col_local and end_col<self.n_columns:
            end_col_local = end_col


        # Compute all similarities for each item using vectorization
        for columnIndex in range(start_col_local, end_col_local):

            processedItems += 1

            if time.time() - start_time_print_batch >= 30 or columnIndex==end_col_local:
                columnPerSec = processedItems / (time.time() - start_time)

                print("Similarity column {} ( {:2.0f} % ), {:.2f} column/sec, elapsed time {:.2f} min".format(
                    processedItems, processedItems / (end_col_local - start_col_local) * 100, columnPerSec, (time.time() - start_time)/ 60))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_print_batch = time.time()



            item_data = self.dataMatrix[:,columnIndex]

            delta = item_data - self.dataMatrix.T

            item_distance = np.sum(delta**2, axis=1)

            if self.use_row_weights:
                item_distance = np.multiply(item_distance, self.row_weights)


            item_distance = np.sqrt(item_distance/self.n_rows)

            if self.similarity_is_exponential:
                item_similarity = 1/np.exp(item_distance + self.shrink)

            elif self.similarity_is_linear:
                item_similarity = 1/(1 + item_distance + self.shrink)


            item_similarity[columnIndex] = 0.0

            this_column_weights = item_similarity



            if self.TopK == 0:
                self.W_dense[:, columnIndex] = this_column_weights

            else:
                # Sort indices and select TopK
                # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                # - Partition the data to extract the set of relevant items
                # - Sort only the relevant items
                # - Get the original item index
                relevant_items_partition = (-this_column_weights).argpartition(self.TopK-1)[0:self.TopK]
                relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
                top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

                # Incrementally build sparse matrix, do not add zeros
                notZerosMask = this_column_weights[top_k_idx] != 0.0
                numNotZeros = np.sum(notZerosMask)

                values.extend(this_column_weights[top_k_idx][notZerosMask])
                rows.extend(top_k_idx[notZerosMask])
                cols.extend(np.ones(numNotZeros) * columnIndex)



        # End while on columns


        if self.TopK == 0:
            return self.W_dense

        else:

            W_sparse = sps.csr_matrix((values, (rows, cols)),
                                      shape=(self.n_columns, self.n_columns),
                                      dtype=np.float32)


            return W_sparse