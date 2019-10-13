#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 28 June 2017

@author: Maurizio Ferrari Dacrema
"""

import sys
import time

import numpy as np
from scipy.special import expit

from Base.BaseRecommender import BaseRecommender


class SLIM_BPR(BaseRecommender):
    """
    This class is a python porting of the BPRSLIM algorithm in MyMediaLite written in C#
    The code is identical with no optimizations
    """

    def __init__(self, URM_train, lambda_i = 0.0025, lambda_j = 0.00025, learning_rate = 0.05):
        super(SLIM_BPR, self).__init__()

        self.URM_train = URM_train
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate

        self.normalize = False
        self.sparse_weights = False


    def updateFactors(self, user_id, pos_item_id, neg_item_id):

        # Calculate current predicted score
        userSeenItems = self.URM_train[user_id].indices
        prediction = 0

        for userSeenItem in userSeenItems:
            prediction += self.S[pos_item_id, userSeenItem] - self.S[neg_item_id, userSeenItem]


        x_uij = prediction
        logisticFunction = expit(-x_uij)

        # Update similarities for all items except those sampled
        for userSeenItem in userSeenItems:

            # For positive item is PLUS logistic minus lambda*S
            if(pos_item_id != userSeenItem):
                update = logisticFunction - self.lambda_i*self.S[pos_item_id, userSeenItem]
                self.S[pos_item_id, userSeenItem] += self.learning_rate*update

            # For positive item is MINUS logistic minus lambda*S
            if (neg_item_id != userSeenItem):
                update = - logisticFunction - self.lambda_j*self.S[neg_item_id, userSeenItem]
                self.S[neg_item_id, userSeenItem] += self.learning_rate*update





    def fit(self, epochs=15):
        """
        Train SLIM wit BPR. If the model was already trained, overwrites matrix S
        :param epochs:
        :return: -
        """

        # Initialize similarity with random values and zero-out diagonal
        self.S = np.random.random((self.n_items, self.n_items)).astype('float32')
        self.S[np.arange(self.n_items),np.arange(self.n_items)] = 0

        start_time_train = time.time()

        for currentEpoch in range(epochs):

            start_time_epoch = time.time()

            self.epochIteration()
            print("Epoch {} of {} complete in {:.2f} minutes".format(currentEpoch+1, epochs, float(time.time()-start_time_epoch)/60))

        print("Train completed in {:.2f} minutes".format(float(time.time()-start_time_train)/60))

        # The similarity matrix is learnt row-wise
        # To be used in the product URM*S must be transposed to be column-wise
        self.W = self.S.T

        del self.S


    def epochIteration(self):

        # Get number of available interactions
        numPositiveIteractions = self.URM_train.nnz

        start_time = time.time()

        # Uniform user sampling without replacement
        for numSample in range(numPositiveIteractions):

            user_id, pos_item_id, neg_item_id = self.sampleTriple()
            self.updateFactors(user_id, pos_item_id, neg_item_id)

            if(numSample % 5000 == 0):
                print("Processed {} ( {:.2f}% ) in {:.4f} seconds".format(numSample,
                                  100.0* float(numSample)/numPositiveIteractions,
                                  time.time()-start_time))

                sys.stderr.flush()

                start_time = time.time()





    def sampleUser(self):
        """
        Sample a user that has viewed at least one and not all items
        :return: user_id
        """
        while(True):

            user_id = np.random.randint(0, self.n_users)
            numSeenItems = self.URM_train[user_id].nnz

            if(numSeenItems >0 and numSeenItems<self.n_items):
                return user_id



    def sampleItemPair(self, user_id):
        """
        Returns for the given user a random seen item and a random not seen item
        :param user_id:
        :return: pos_item_id, neg_item_id
        """

        userSeenItems = self.URM_train[user_id].indices

        pos_item_id = userSeenItems[np.random.randint(0,len(userSeenItems))]

        while(True):

            neg_item_id = np.random.randint(0, self.n_items)

            if(neg_item_id not in userSeenItems):
                return pos_item_id, neg_item_id


    def sampleTriple(self):
        """
        Randomly samples a user and then samples randomly a seen and not seen item
        :return: user_id, pos_item_id, neg_item_id
        """

        user_id = self.sampleUser()
        pos_item_id, neg_item_id = self.sampleItemPair(user_id)

        return user_id, pos_item_id, neg_item_id

