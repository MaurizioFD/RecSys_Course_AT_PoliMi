#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16/09/2017

@author: Maurizio Ferrari Dacrema
"""

import pickle


class SimilarityMatrixRecommender(object):
    """
    This class refers to a Recommender KNN which uses a similarity matrix, it provides two function to compute item's score
    bot for user-based and Item-based models as well as a function to save the W_matrix
    """

    def __init__(self):
        super(SimilarityMatrixRecommender, self).__init__()

        self.sparse_weights = True
        self._compute_item_score = self._compute_score_item_based


    def _compute_score_item_based(self, user_id, items_to_compute = None):

        assert self.sparse_weights, "SimilarityMatrixRecommender: sparse_weights False not supported"

        scores = self.URM_train[user_id].dot(self.W_sparse).toarray()

        return scores if items_to_compute is None else scores[:, items_to_compute]


    def _compute_score_user_based(self, user_id, items_to_compute = None):

        assert self.sparse_weights, "SimilarityMatrixRecommender: sparse_weights False not supported"
        scores = self.W_sparse[user_id].dot(self.URM_train).toarray()

        return scores if items_to_compute is None else scores[:, items_to_compute]


    def saveModel(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        dictionary_to_save = {"sparse_weights": self.sparse_weights}

        if self.sparse_weights:
            dictionary_to_save["W_sparse"] = self.W_sparse
        else:
            dictionary_to_save["W"] = self.W

        pickle.dump(dictionary_to_save,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        print("{}: Saving complete".format(self.RECOMMENDER_NAME))
