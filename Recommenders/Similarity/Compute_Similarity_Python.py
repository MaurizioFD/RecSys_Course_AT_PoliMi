#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import time, sys
import scipy.sparse as sps
import Recommenders.Recommender_utils as recommender_utils
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit



class Incremental_Similarity_Builder:
    """
    This class can be used to create a square "matrix_size x matrix_size" matrix iteratively.
    It is developed for all recommenders that need to build, for example, an item-item or user-user similarity one
    column at a time.
    This class uses arrays to store the partial data and only when requested creates the sparse matrix. The arrays are
    pre-initialized with a size equal to the attribute initial_data_block. If the data points exceed the data_block size then a new
    array of length additional_data_block is created and concatenated to the previous one. This may cause memory spikes.
    """

    def __init__(self, matrix_size, initial_data_block = 10000000, additional_data_block = 10000000, dtype = np.float32):

        self._matrix_size = matrix_size
        self._initial_data_block = initial_data_block
        self._additional_data_block = additional_data_block
        self._next_cell_pointer = 0

        self._dtype_data = dtype
        self._dtype_coordinates = np.uint32
        self._max_value_of_coordinate_dtype = np.iinfo(self._dtype_coordinates).max

        self._row_array = np.zeros(self._initial_data_block, dtype=self._dtype_coordinates)
        self._col_array = np.zeros(self._initial_data_block, dtype=self._dtype_coordinates)
        self._data_array = np.zeros(self._initial_data_block, dtype=self._dtype_data)


    def add_data_lists(self, row_list_to_add, col_list_to_add, data_list_to_add):

        for data_point_index in range(len(row_list_to_add)):

            if self._next_cell_pointer == len(self._row_array):
                self._row_array = np.concatenate((self._row_array, np.zeros(self._additional_data_block, dtype=self._dtype_coordinates)))
                self._col_array = np.concatenate((self._col_array, np.zeros(self._additional_data_block, dtype=self._dtype_coordinates)))
                self._data_array = np.concatenate((self._data_array, np.zeros(self._additional_data_block, dtype=self._dtype_data)))

            self._row_array[self._next_cell_pointer] = row_list_to_add[data_point_index]
            self._col_array[self._next_cell_pointer] = col_list_to_add[data_point_index]
            self._data_array[self._next_cell_pointer] = data_list_to_add[data_point_index]

            self._next_cell_pointer += 1



    def get_SparseMatrix(self):

        shape = (self._matrix_size, self._matrix_size)

        sparseMatrix = sps.csr_matrix((self._data_array[:self._next_cell_pointer],
                                       (self._row_array[:self._next_cell_pointer], self._col_array[:self._next_cell_pointer])),
                                      shape=shape,
                                      dtype=self._dtype_data)

        sparseMatrix.eliminate_zeros()


        return sparseMatrix




class Compute_Similarity_Python:

    def __init__(self, dataMatrix, topK=100, shrink = 0, normalize = True,
                 asymmetric_alpha = 0.5, tversky_alpha = 1.0, tversky_beta = 1.0,
                 similarity = "cosine", row_weights = None):
        """
        Computes the cosine similarity on the columns of dataMatrix
        If it is computed on URM=|users|x|items|, pass the URM as is.
        If it is computed on ICM=|items|x|features|, pass the ICM transposed.
        :param dataMatrix:
        :param topK:
        :param shrink:
        :param normalize:           If True divide the dot product by the product of the norms
        :param row_weights:         Multiply the values in each row by a specified value. Array
        :param asymmetric_alpha     Coefficient alpha for the asymmetric cosine
        :param similarity:  "cosine"        computes Cosine similarity
                            "adjusted"      computes Adjusted Cosine, removing the average of the users
                            "asymmetric"    computes Asymmetric Cosine
                            "pearson"       computes Pearson Correlation, removing the average of the items
                            "jaccard"       computes Jaccard similarity for binary interactions using Tanimoto
                            "dice"          computes Dice similarity for binary interactions
                            "tversky"       computes Tversky similarity for binary interactions
                            "tanimoto"      computes Tanimoto coefficient for binary interactions

        """
        """
        Asymmetric Cosine as described in: 
        Aiolli, F. (2013, October). Efficient top-n recommendation for very large scale binary rated datasets. In Proceedings of the 7th ACM conference on Recommender systems (pp. 273-280). ACM.
        
        """

        super(Compute_Similarity_Python, self).__init__()

        self.shrink = shrink
        self.normalize = normalize

        self.n_rows, self.n_columns = dataMatrix.shape
        self.topK = min(topK, self.n_columns)

        self.asymmetric_alpha = asymmetric_alpha
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta

        self.dataMatrix = dataMatrix.copy()

        self.adjusted_cosine = False
        self.asymmetric_cosine = False
        self.pearson_correlation = False
        self.tanimoto_coefficient = False
        self.dice_coefficient = False
        self.tversky_coefficient = False

        if similarity == "adjusted":
            self.adjusted_cosine = True
        elif similarity == "asymmetric":
            self.asymmetric_cosine = True
        elif similarity == "pearson":
            self.pearson_correlation = True
        elif similarity == "jaccard" or similarity == "tanimoto":
            self.tanimoto_coefficient = True
            # Tanimoto has a specific kind of normalization
            self.normalize = False

        elif similarity == "dice":
            self.dice_coefficient = True
            self.normalize = False

        elif similarity == "tversky":
            self.tversky_coefficient = True
            self.normalize = False

        elif similarity == "cosine":
            pass
        else:
            raise ValueError("Cosine_Similarity: value for argument 'mode' not recognized."
                             " Allowed values are: 'cosine', 'pearson', 'adjusted', 'asymmetric', 'jaccard', 'tanimoto',"
                             "dice, tversky."
                             " Passed value was '{}'".format(similarity))


        self.use_row_weights = False

        if row_weights is not None:

            if dataMatrix.shape[0] != len(row_weights):
                raise ValueError("Cosine_Similarity: provided row_weights and dataMatrix have different number of rows."
                                 "Col_weights has {} columns, dataMatrix has {}.".format(len(row_weights), dataMatrix.shape[0]))

            self.use_row_weights = True
            self.row_weights = row_weights.copy()
            self.row_weights_diag = sps.diags(self.row_weights)

            self.dataMatrix_weighted = self.dataMatrix.T.dot(self.row_weights_diag).T






    def applyAdjustedCosine(self):
        """
        Remove from every data point the average for the corresponding row
        :return:
        """

        self.dataMatrix = recommender_utils.check_matrix(self.dataMatrix, 'csr')


        interactionsPerRow = np.diff(self.dataMatrix.indptr)

        nonzeroRows = interactionsPerRow > 0
        sumPerRow = np.asarray(self.dataMatrix.sum(axis=1)).ravel()

        rowAverage = np.zeros_like(sumPerRow)
        rowAverage[nonzeroRows] = sumPerRow[nonzeroRows] / interactionsPerRow[nonzeroRows]


        # Split in blocks to avoid duplicating the whole data structure
        start_row = 0
        end_row= 0

        blockSize = 1000


        while end_row < self.n_rows:

            end_row = min(self.n_rows, end_row + blockSize)

            self.dataMatrix.data[self.dataMatrix.indptr[start_row]:self.dataMatrix.indptr[end_row]] -= \
                np.repeat(rowAverage[start_row:end_row], interactionsPerRow[start_row:end_row])

            start_row += blockSize




    def applyPearsonCorrelation(self):
        """
        Remove from every data point the average for the corresponding column
        :return:
        """

        self.dataMatrix = recommender_utils.check_matrix(self.dataMatrix, 'csc')


        interactionsPerCol = np.diff(self.dataMatrix.indptr)

        nonzeroCols = interactionsPerCol > 0
        sumPerCol = np.asarray(self.dataMatrix.sum(axis=0)).ravel()

        colAverage = np.zeros_like(sumPerCol)
        colAverage[nonzeroCols] = sumPerCol[nonzeroCols] / interactionsPerCol[nonzeroCols]


        # Split in blocks to avoid duplicating the whole data structure
        start_col = 0
        end_col= 0

        blockSize = 1000


        while end_col < self.n_columns:

            end_col = min(self.n_columns, end_col + blockSize)

            self.dataMatrix.data[self.dataMatrix.indptr[start_col]:self.dataMatrix.indptr[end_col]] -= \
                np.repeat(colAverage[start_col:end_col], interactionsPerCol[start_col:end_col])

            start_col += blockSize


    def useOnlyBooleanInteractions(self):

        # Split in blocks to avoid duplicating the whole data structure
        start_pos = 0
        end_pos= 0

        blockSize = 1000


        while end_pos < len(self.dataMatrix.data):

            end_pos = min(len(self.dataMatrix.data), end_pos + blockSize)

            self.dataMatrix.data[start_pos:end_pos] = np.ones(end_pos-start_pos)

            start_pos += blockSize




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


        if self.adjusted_cosine:
            self.applyAdjustedCosine()

        elif self.pearson_correlation:
            self.applyPearsonCorrelation()

        elif self.tanimoto_coefficient or self.dice_coefficient or self.tversky_coefficient:
            self.useOnlyBooleanInteractions()


        # We explore the matrix column-wise
        self.dataMatrix = recommender_utils.check_matrix(self.dataMatrix, 'csc')


        # Compute sum of squared values to be used in normalization
        sum_of_squared = np.array(self.dataMatrix.power(2).sum(axis=0)).ravel()

        # Tanimoto does not require the square root to be applied
        if not (self.tanimoto_coefficient or self.dice_coefficient or self.tversky_coefficient):
            sum_of_squared = np.sqrt(sum_of_squared)


        if self.asymmetric_cosine:
            sum_of_squared_to_alpha = np.power(sum_of_squared + 1e-6, 2 * self.asymmetric_alpha)
            sum_of_squared_to_1_minus_alpha = np.power(sum_of_squared + 1e-6, 2 * (1 - self.asymmetric_alpha))


        self.dataMatrix = recommender_utils.check_matrix(self.dataMatrix, 'csc')

        start_col_local = 0
        end_col_local = self.n_columns

        if start_col is not None and start_col>0 and start_col<self.n_columns:
            start_col_local = start_col

        if end_col is not None and end_col>start_col_local and end_col<self.n_columns:
            end_col_local = end_col




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
                this_column_weights[columnIndex] = 0.0

                # Apply normalization and shrinkage, ensure denominator != 0
                if self.normalize:

                    if self.asymmetric_cosine:
                        denominator = sum_of_squared_to_alpha[columnIndex] * sum_of_squared_to_1_minus_alpha + self.shrink + 1e-6
                    else:
                        denominator = sum_of_squared[columnIndex] * sum_of_squared + self.shrink + 1e-6

                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)


                # Apply the specific denominator for Tanimoto
                elif self.tanimoto_coefficient:
                    denominator = sum_of_squared[columnIndex] + sum_of_squared - this_column_weights + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                elif self.dice_coefficient:
                    denominator = sum_of_squared[columnIndex] + sum_of_squared + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                elif self.tversky_coefficient:
                    denominator = this_column_weights + \
                                  (sum_of_squared[columnIndex] - this_column_weights)*self.tversky_alpha + \
                                  (sum_of_squared - this_column_weights)*self.tversky_beta + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                # If no normalization or tanimoto is selected, apply only shrink
                elif self.shrink != 0:
                    this_column_weights = this_column_weights/self.shrink

                # Sort indices and select topK, partition the data to extract the set of relevant items
                relevant_items_partition = np.argpartition(-this_column_weights, self.topK - 1, axis=0)[0:self.topK]
                this_column_weights = this_column_weights[relevant_items_partition]

                # Incrementally build sparse matrix, do not add zeros
                if np.any(this_column_weights == 0.0):
                    non_zero_mask = this_column_weights != 0.0
                    relevant_items_partition = relevant_items_partition[non_zero_mask]
                    this_column_weights = this_column_weights[non_zero_mask]

                similarity_builder.add_data_lists(row_list_to_add=relevant_items_partition,
                                                  col_list_to_add=np.ones(len(relevant_items_partition), dtype = np.int) * columnIndex,
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