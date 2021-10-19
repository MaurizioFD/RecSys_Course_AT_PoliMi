#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Recommender_utils import check_matrix

import scipy.sparse as sps

class PredefinedListRecommender(BaseRecommender):
    """PredefinedListRecommender recommender"""

    RECOMMENDER_NAME = "PredefinedListRecommenderRecommender"

    def __init__(self, URM_recommendations_items):
        super(PredefinedListRecommender, self).__init__()

        # convert to csc matrix for faster column-wise sum
        self.URM_recommendations = check_matrix(URM_recommendations_items, 'csr', dtype=np.int)

        self.URM_train = sps.csr_matrix((self.URM_recommendations.shape))



    def fit(self):
        pass
 

    def recommend(self, user_id, cutoff = None, remove_seen_flag=True, remove_top_pop_flag = False, remove_custom_items_flag = False):

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



