#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Massimo Quadrana
"""

import numpy as np
from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix

import scipy.sparse as sps

class PredefinedListRecommender(Recommender):
    """PredefinedListRecommender recommender"""

    RECOMMENDER_NAME = "PredefinedListRecommenderRecommender"

    def __init__(self, URM_recommendations_items):
        super(PredefinedListRecommender, self).__init__()

        # convert to csc matrix for faster column-wise sum
        self.URM_recommendations = check_matrix(URM_recommendations_items, 'csr', dtype=np.int)

        self.URM_train = sps.csr_matrix((self.URM_recommendations.shape))

        #self._base_item_score = np.ones(self.URM_recommendations.shape[1]) * np.inf * (-1)
        #self.compute_item_score = self.compute_score_predefined_list



    def fit(self):
        pass
    #
    # def compute_score_predefined_list(self, user_id):
    #
    #     #
    #
    #     start_pos = self.URM_recommendations.indptr[user_id]
    #     end_pos = self.URM_recommendations.indptr[user_id+1]
    #
    #     recommendation_list = self.URM_recommendations.data[start_pos:end_pos]
    #     recommendation_list_score = np.flip(np.arange(0, len(recommendation_list)), axis=0)
    #
    #     item_scores = self._base_item_score.copy()
    #     item_scores[recommendation_list] = recommendation_list_score
    #
    #     return item_scores
    #




    def recommend(self, user_id, cutoff = None, remove_seen_flag=True, remove_top_pop_flag = False, remove_CustomItems_flag = False):

        if cutoff is None:
            cutoff= self.URM_train.shape[1] - 1

        start_pos = self.URM_recommendations.indptr[user_id]
        end_pos = self.URM_recommendations.indptr[user_id+1]

        recommendation_list = self.URM_recommendations.data[start_pos:end_pos]

        if len(recommendation_list[:cutoff]) == 0:
            pass

        return recommendation_list[:cutoff]



    def __str__(self):
        return "PredefinedListRecommender"



