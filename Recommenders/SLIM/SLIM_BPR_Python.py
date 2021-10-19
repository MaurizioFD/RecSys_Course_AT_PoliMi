#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 28 June 2017
Updated on 28 November 2020

@author: Maurizio Ferrari Dacrema
"""

import time
import numpy as np
import scipy.sparse as sps

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.Recommender_utils import similarityMatrixTopK


class SLIM_BPR_Python(BaseItemSimilarityMatrixRecommender):
    """
    This class is a python porting of the BPRSLIM algorithm in MyMediaLite written in C#

    This class does not implement early stopping
    """

    def __init__(self, URM_train, ):
        super(SLIM_BPR_Python, self).__init__(URM_train)



    def fit(self, topK = 100, epochs = 25, lambda_i = 0.0025, lambda_j = 0.00025, learning_rate = 0.05):
        """

        :param topK:
        :param epochs:
        :param lambda_i:
        :param lambda_j:
        :param learning_rate:
        :return:
        """


        # Initialize similarity with zero values
        self.item_item_S = np.zeros((self.n_items, self.n_items), dtype = np.float)

        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate

        start_time_train = time.time()

        for n_epoch in range(epochs):
            self._run_epoch(n_epoch)

        print("Train completed in {:.2f} minutes".format(float(time.time()-start_time_train)/60))

        self.W_sparse = similarityMatrixTopK(self.item_item_S, k=topK, verbose=False)
        self.W_sparse = sps.csr_matrix(self.W_sparse)


    def _run_epoch(self, n_epoch):

        start_time = time.time()

        # Uniform user sampling without replacement
        for sample_num in range(self.n_users):

            user_id, pos_item_id, neg_item_id = self._sample_triplet()

            # Calculate current predicted score
            user_seen_items = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id+1]]

            # Compute positive and negative item predictions. Assuming implicit interactions.
            x_ui = self.item_item_S[pos_item_id, user_seen_items].sum()
            x_uj = self.item_item_S[neg_item_id, user_seen_items].sum()

            # Gradient
            x_uij = x_ui - x_uj
            sigmoid_gradient = 1 / (1 + np.exp(x_uij))

            # Update
            self.item_item_S[pos_item_id, user_seen_items] += self.learning_rate * (sigmoid_gradient - self.lambda_i * self.item_item_S[pos_item_id, user_seen_items])
            self.item_item_S[pos_item_id, pos_item_id] = 0

            self.item_item_S[neg_item_id, user_seen_items] -= self.learning_rate * (sigmoid_gradient - self.lambda_j * self.item_item_S[neg_item_id, user_seen_items])
            self.item_item_S[neg_item_id, neg_item_id] = 0

            # Print some stats
            if (sample_num + 1) % 150000 == 0 or (sample_num + 1) == self.n_users:
                elapsed_time = time.time() - start_time
                samples_per_second = (sample_num + 1) / elapsed_time
                print("Epoch {}, Iteration {} in {:.2f} seconds. Samples per second {:.2f}".format(n_epoch+1, sample_num+1, elapsed_time, samples_per_second))

                start_time = time.time()



    def _sample_triplet(self):

        non_empty_user = False

        while not non_empty_user:
            user_id = np.random.choice(self.n_users)
            user_seen_items = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

            if len(user_seen_items) > 0:
                non_empty_user = True

        pos_item_id = np.random.choice(user_seen_items)

        neg_item_selected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        while (not neg_item_selected):
            neg_item_id = np.random.randint(0, self.n_items)

            if (neg_item_id not in user_seen_items):
                neg_item_selected = True

        return user_id, pos_item_id, neg_item_id

