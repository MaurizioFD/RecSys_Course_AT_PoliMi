#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/06/18

@author: Maurizio Ferrari Dacrema
"""

from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix

import scipy.sparse as sps


class PureSVDRecommender(Recommender):
    """ PureSVDRecommender"""

    RECOMMENDER_NAME = "PureSVDRecommender"

    def __init__(self, URM_train):

        super(PureSVDRecommender, self).__init__()

        # CSR is faster during evaluation
        self.URM_train = check_matrix(URM_train, 'csr')
        self._compute_item_score = self._compute_score_SVD


    def fit(self, num_factors=100):

        from sklearn.utils.extmath import randomized_svd

        print(self.RECOMMENDER_NAME + " Computing SVD decomposition...")

        self.U, self.Sigma, self.VT = randomized_svd(self.URM_train,
                                      n_components=num_factors,
                                      #n_iter=5,
                                      random_state=None)

        self.s_Vt = sps.diags(self.Sigma)*self.VT

        print(self.RECOMMENDER_NAME + " Computing SVD decomposition... Done!")


    def _compute_score_SVD(self, user_id_array, items_to_compute = None):

        assert self.U.shape[0] > user_id_array.max(),\
                "PureSVDRecommender: Cold users not allowed. Users in trained model are {}, requested prediction for users up to {}".format(
                self.U.shape[0], user_id_array.max())

        if items_to_compute is not None:
            item_scores = self.U[user_id_array, :].dot(self.s_Vt[:,items_to_compute])
        else:
            item_scores = self.U[user_id_array, :].dot(self.s_Vt)

        return item_scores




    def saveModel(self, folder_path, file_name = None):

        import pickle

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        data_dict = {
            "U":self.U,
            "Sigma":self.Sigma,
            "VT":self.VT,
            "s_Vt":self.s_Vt
        }

        pickle.dump(data_dict,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        print("{}: Saving complete".format(self.RECOMMENDER_NAME, folder_path + file_name))

