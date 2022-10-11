#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.Recommender_utils import similarityMatrixTopK, check_matrix
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit
from sklearn.preprocessing import normalize
import numpy as np
import time
import scipy.sparse as sps

class EASE_R_Recommender(BaseItemSimilarityMatrixRecommender):
    """ EASE_R_Recommender

        https://arxiv.org/pdf/1905.03375.pdf

    @inproceedings{DBLP:conf/www/Steck19,
          author    = {Harald Steck},
          editor    = {Ling Liu and
                       Ryen W. White and
                       Amin Mantrach and
                       Fabrizio Silvestri and
                       Julian J. McAuley and
                       Ricardo Baeza{-}Yates and
                       Leila Zia},
          title     = {Embarrassingly Shallow Autoencoders for Sparse Data},
          booktitle = {The World Wide Web Conference, {WWW} 2019, San Francisco, CA, USA,
                       May 13-17, 2019},
          pages     = {3251--3257},
          publisher = {{ACM}},
          year      = {2019},
          url       = {https://doi.org/10.1145/3308558.3313710},
          doi       = {10.1145/3308558.3313710},
          timestamp = {Sun, 22 Sep 2019 18:12:47 +0200},
          biburl    = {https://dblp.org/rec/conf/www/Steck19.bib},
          bibsource = {dblp computer science bibliography, https://dblp.org}
    }

    """

    RECOMMENDER_NAME = "EASE_R_Recommender"


    def __init__(self, URM_train, sparse_threshold_quota = None, verbose = True):
        super(EASE_R_Recommender, self).__init__(URM_train, verbose = verbose)
        self.sparse_threshold_quota = sparse_threshold_quota

    def fit(self, topK=None, l2_norm = 1e3, normalize_matrix = False):

        start_time = time.time()
        self._print("Fitting model... ")

        if normalize_matrix:
            # Normalize rows and then columns
            self.URM_train = normalize(self.URM_train, norm='l2', axis=1)
            self.URM_train = normalize(self.URM_train, norm='l2', axis=0)
            self.URM_train = sps.csr_matrix(self.URM_train)


        # Grahm matrix is X^t X, compute dot product
        grahm_matrix = self.URM_train.T.dot(self.URM_train).toarray()

        diag_indices = np.diag_indices(grahm_matrix.shape[0])
        grahm_matrix[diag_indices] += l2_norm

        P = np.linalg.inv(grahm_matrix)

        B = P / (-np.diag(P))

        B[diag_indices] = 0.0


        new_time_value, new_time_unit = seconds_to_biggest_unit(time.time()-start_time)
        self._print("Fitting model... done in {:.2f} {}".format( new_time_value, new_time_unit))

        # Check if the matrix should be saved in a sparse or dense format
        # The matrix is sparse, regardless of the presence of the topK, if nonzero cells are less than sparse_threshold_quota %
        # B contains positive and negative values, so topK is selected based on the *absolute* value to preserve strong negatives
        if topK is not None:
            B = similarityMatrixTopK(B, k = topK, use_absolute_values = True, verbose = False)


        if self._is_content_sparse_check(B):
            self._print("Detected model matrix to be sparse, changing format.")
            self.W_sparse = check_matrix(B, format='csr', dtype=np.float32)

        else:
            self.W_sparse = check_matrix(B, format='npy', dtype=np.float32)
            self._W_sparse_format_checked = True
            self._compute_item_score = self._compute_score_W_dense
        #
        #
        # if topK is None:
        #     self.W_sparse = B
        #     self._W_sparse_format_checked = True
        #     self._compute_item_score = self._compute_score_W_dense
        #
        # else:
        #     self.W_sparse = similarityMatrixTopK(B, k = topK, verbose = False)
        #     self.W_sparse = sps.csr_matrix(self.W_sparse)


    def _is_content_sparse_check(self, matrix):

        if self.sparse_threshold_quota is None:
            return False

        if sps.issparse(matrix):
            nonzero = matrix.nnz
        else:
            nonzero = np.count_nonzero(matrix)

        return nonzero / (matrix.shape[0]**2) <= self.sparse_threshold_quota



    def _compute_score_W_dense(self, user_id_array, items_to_compute = None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        self._check_format()

        user_profile_array = self.URM_train[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)*np.inf
            item_scores_all = user_profile_array.dot(self.W_sparse)#.toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_profile_array.dot(self.W_sparse)#.toarray()

        return item_scores





    def load_model(self, folder_path, file_name = None):
        super(EASE_R_Recommender, self).load_model(folder_path, file_name = file_name)

        if not sps.issparse(self.W_sparse):
            self._W_sparse_format_checked = True
            self._compute_item_score = self._compute_score_W_dense