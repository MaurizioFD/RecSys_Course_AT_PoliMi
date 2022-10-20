#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import time, sys
import scipy.sparse as sps
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit
from Recommenders.Similarity.Compute_Similarity_Python import Incremental_Similarity_Builder

class Compute_Similarity_Euclidean:

    def __init__(self, dataMatrix, topK=100, shrink = 0, normalize=False, normalize_avg_row=False,
                 similarity_from_distance_mode ="lin", row_weights = None, **args):
        """
        Computes the euclidean similarity on the columns of dataMatrix
        If it is computed on URM=|users|x|items|, pass the URM as is.
        If it is computed on ICM=|items|x|features|, pass the ICM transposed.
        :param dataMatrix:
        :param topK:
        :param normalize
        :param row_weights:         Multiply the values in each row by a specified value. Array
        :param similarity_from_distance_mode:       "exp"        euclidean_similarity = 1/(e ^ euclidean_distance)
                                                    "lin"        euclidean_similarity = 1/(1 + euclidean_distance)
                                                    "log"        euclidean_similarity = 1/log(1 + euclidean_distance)
        :param args:                accepts other arguments not needed by the current object

        """

        super(Compute_Similarity_Euclidean, self).__init__()

        self.shrink = shrink
        self.normalize = normalize
        self.normalize_avg_row = normalize_avg_row

        self.n_rows, self.n_columns = dataMatrix.shape
        self.topK = min(topK, self.n_columns)

        self.dataMatrix = dataMatrix.copy()

        self.similarity_is_exp = False
        self.similarity_is_lin = False
        self.similarity_is_log = False

        if similarity_from_distance_mode == "exp":
            self.similarity_is_exp = True
        elif similarity_from_distance_mode == "lin":
            self.similarity_is_lin = True
        elif similarity_from_distance_mode == "log":
            self.similarity_is_log = True
        else:
            raise ValueError("Compute_Similarity_Euclidean: value for argument 'mode' not recognized."
                             " Allowed values are: 'exp', 'lin', 'log'."
                             " Passed value was '{}'".format(similarity_from_distance_mode))



        self.use_row_weights = False

        if row_weights is not None:

            if dataMatrix.shape[0] != len(row_weights):
                raise ValueError("Compute_Similarity_Euclidean: provided row_weights and dataMatrix have different number of rows."
                                 "row_weights has {} rows, dataMatrix has {}.".format(len(row_weights), dataMatrix.shape[0]))

            self.use_row_weights = True
            self.row_weights = row_weights.copy()
            self.row_weights_diag = sps.diags(self.row_weights)

            self.dataMatrix_weighted = self.dataMatrix.T.dot(self.row_weights_diag).T








    def compute_similarity(self, start_col=None, end_col=None, block_size = 100):
        """
        Compute the similarity for the given dataset
        :param self:
        :param start_col: column to begin with
        :param end_col: column to stop before, end_col is excluded
        :return:
        """


        similarity_builder = Incremental_Similarity_Builder(self.n_columns, initial_data_block=self.n_columns*self.topK, dtype = np.float32)

        start_time = time.time()
        start_time_print_batch = start_time
        processed_items = 0

        start_col_local = 0
        end_col_local = self.n_columns

        if start_col is not None and start_col>0 and start_col<self.n_columns:
            start_col_local = start_col

        if end_col is not None and end_col>start_col_local and end_col<self.n_columns:
            end_col_local = end_col

        # Compute sum of squared values
        item_distance_initial = np.array(self.dataMatrix.power(2).sum(axis=0)).ravel()
        sumOfSquared = np.sqrt(item_distance_initial)

        start_col_block = start_col_local

        this_block_size = 0

        # Compute all similarities for each item using vectorization
        while start_col_block < end_col_local:

            # Compute block first and last column
            end_col_block = min(start_col_block + block_size, end_col_local)
            this_block_size = end_col_block-start_col_block

            # All data points for a given item
            item_data = self.dataMatrix[:, start_col_block:end_col_block]
            item_data = item_data.toarray()

            # Compute item similarities
            if self.use_row_weights:
                this_block_weights = self.dataMatrix_weighted.T.dot(item_data)
            else:
                this_block_weights = self.dataMatrix.T.dot(item_data)


            for col_index_in_block in range(this_block_size):

                if this_block_size == 1:
                    this_column_weights = this_block_weights.ravel()
                else:
                    this_column_weights = this_block_weights[:,col_index_in_block]


                columnIndex = col_index_in_block + start_col_block

                # (a-b)^2 = a^2 + b^2 - 2ab
                item_distance = item_distance_initial.copy()
                item_distance += item_distance_initial[columnIndex]

                item_distance -= 2 * this_column_weights
                item_distance[columnIndex] = 0.0


                if self.use_row_weights:
                    item_distance = np.multiply(item_distance, self.row_weights)


                if self.normalize:
                    denominator = sumOfSquared[columnIndex] * sumOfSquared
                    item_distance[denominator!=0.0] /=  denominator[denominator!=0.0]

                if self.normalize_avg_row:
                    item_distance /= self.n_rows

                nonzero_distance_mask = item_distance > 0.0
                item_distance[nonzero_distance_mask] = np.sqrt(item_distance[nonzero_distance_mask])

                if self.similarity_is_exp:
                    item_similarity = 1/(np.exp(item_distance) + self.shrink + 1e-9)

                elif self.similarity_is_lin:
                    item_similarity = 1/(item_distance + self.shrink + 1e-9)

                elif self.similarity_is_log:
                    item_similarity = 1/(np.log(item_distance+1) + self.shrink + 1e-9)

                else:
                    assert False

                item_similarity[columnIndex] = 0.0
                this_column_weights = item_similarity

                # Sort indices and select topK, partition the data to extract the set of relevant items
                relevant_items_partition = np.argpartition(-this_column_weights, self.topK - 1, axis=0)[0:self.topK]
                this_column_weights = this_column_weights[relevant_items_partition]

                # Incrementally build sparse matrix, do not add zeros
                if np.any(this_column_weights == 0.0):
                    non_zero_mask = this_column_weights != 0.0
                    relevant_items_partition = relevant_items_partition[non_zero_mask]
                    this_column_weights = this_column_weights[non_zero_mask]

                similarity_builder.add_data_lists(row_list_to_add=relevant_items_partition,
                                                  col_list_to_add=np.ones(len(relevant_items_partition), dtype = np.int) *  columnIndex,
                                                  data_list_to_add=this_column_weights)


            # Add previous block size
            start_col_block += this_block_size
            processed_items += this_block_size

            if time.time() - start_time_print_batch >= 300 or end_col_block==end_col_local:
                column_per_sec = processed_items / (time.time() - start_time + 1e-9)
                new_time_value, new_time_unit = seconds_to_biggest_unit(time.time() - start_time)

                print("Similarity column {} ({:4.1f}%), {:.2f} column/sec. Elapsed time {:.2f} {}".format(
                    processed_items, processed_items / (end_col_local - start_col_local) * 100, column_per_sec, new_time_value, new_time_unit))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_print_batch = time.time()


        # End while on columns
        W_sparse = similarity_builder.get_SparseMatrix()

        return W_sparse