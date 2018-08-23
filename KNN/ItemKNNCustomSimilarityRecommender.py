#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""


import numpy as np
from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
from Base.Recommender import Recommender


class ItemKNNCustomSimilarityRecommender(SimilarityMatrixRecommender, Recommender):
    """ ItemKNN recommender"""

    RECOMMENDER_NAME = "ItemKNNCustomSimilarityRecommender"

    def __init__(self, topK=50, shrinkage=100, normalize=False, sparse_weights=True):
        super(ItemKNNCustomSimilarityRecommender, self).__init__()
        self.topK = topK
        self.shrinkage = shrinkage
        self.normalize = normalize
        self.dataset = None
        self.similarity_name = None
        self.sparse_weights = sparse_weights



    def fit(self, item_weights, URM_train, selectTopK = False):

        self.URM_train = check_matrix(URM_train, format='csc')

        if self.URM_train.shape[1] != item_weights.shape[0]:
            raise ValueError("ItemKNNCustomSimilarityRecommender: URM_train and item_weights matrices are not consistent. "
                             "The number of columns in URM_train must be equal to the rows in item_weights."
                             "Current shapes are: URM_train {}, item_weights {}".format(self.URM_train.shape, item_weights.shape))

        if item_weights.shape[0] != item_weights.shape[1]:
            raise ValueError("ItemKNNCustomSimilarityRecommender: item_weights matrice is not square. "
                             "Current shape is {}".format(item_weights.shape))



        # If no topK selection is required, just save the similarity
        if not selectTopK:
            if isinstance(item_weights, np.ndarray):
                self.W = item_weights
                self.sparse_weights = False
            else:
                self.W_sparse = check_matrix(item_weights, format='csr')
                self.sparse_weights = True

            return

        else:
            self.W_sparse = similarityMatrixTopK(item_weights, forceSparseOutput = True, k=self.topK)
            self.sparse_weights = True
