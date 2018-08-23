#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 28 June 2017

@author: Maurizio Ferrari Dacrema
"""

import sys
import time

import numpy as np
import scipy.sparse as sps
from Base.Recommender_utils import similarityMatrixTopK
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
from scipy.special import expit

from SLIM_BPR.BPR_sampling import BPR_Sampling
from Base.Recommender import Recommender


def sigmoidFunction(x):
  return 1 / (1 + np.exp(-x))


class SLIM_BPR_Python(BPR_Sampling, SimilarityMatrixRecommender, Recommender):

    RECOMMENDER_NAME = "SLIM_BPR_Recommender"

    def __init__(self, URM_train, positive_threshold=4, sparse_weights = False):
        super(SLIM_BPR_Python, self).__init__()

        """
          Creates a new object for training and testing a Bayesian
          Personalised Ranking (BPR) SLIM

          This object uses the Theano library for training the model, meaning
          it can run on a GPU through CUDA. To make sure your Theano
          install is using the GPU, see:

            http://deeplearning.net/software/theano/tutorial/using_gpu.html

          When running on CPU, we recommend using OpenBLAS.

            http://www.openblas.net/
        """
        """
        if objective!='sigmoid' and objective != 'logsigmoid':
            raise ValueError("Objective not valid. Acceptable values are 'sigmoid' and 'logsigmoid'. Provided value was '{}'".format(objective))
        self.objective = objective
        """

        self.URM_train = URM_train
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]
        self.normalize = False
        self.sparse_weights = sparse_weights
        self.positive_threshold = positive_threshold


        self.URM_mask = self.URM_train.copy()

        self.URM_mask.data = self.URM_mask.data >= self.positive_threshold
        self.URM_mask.eliminate_zeros()







    def updateSimilarityMatrix(self):

        if self.topK != False:
            if self.sparse_weights:
                self.W_sparse = similarityMatrixTopK(self.S.T, k=self.topK, forceSparseOutput=True)
            else:
                self.W = similarityMatrixTopK(self.S.T, k=self.topK, forceSparseOutput=False)

        else:
            if self.sparse_weights:
                self.W_sparse = sps.csr_matrix(self.S.T)
            else:
                self.W = self.S.T



    def updateWeightsLoop(self, u, i, j):
        """
        Define the update rules to be used in the train phase and compile the train function
        :return:
        """

        x_ui = self.S[i]
        x_uj = self.S[j]

        # The difference is computed on the whole row not only on the user_seen items
        # The performance seems to be higher this way
        x_uij = x_ui - x_uj

        # Sigmoid whose argument is minus in order for the exponent of the exponential to be positive
        sigmoid = expit(-x_uij)

        delta_i = sigmoid-self.lambda_i*self.S[i]
        delta_j = -sigmoid-self.lambda_j*self.S[j]

        # Since a shared variable may be the target of only one update rule
        # All the required updates are chained inside a subtensor
        for sampleIndex in range(self.batch_size):

            user_id = u[sampleIndex]

            for item_id in self.userSeenItems[user_id]:
                # Do not update items i
                if item_id != i[sampleIndex]:
                    self.S[i] += self.learning_rate * delta_i

                # Do not update j
                if item_id != j[sampleIndex]:
                    self.S[j] += self.learning_rate * delta_j


    def updateWeightsBatch(self, u, i, j):
        """
        Define the update rules to be used in the train phase and compile the train function
        :return:
        """

        if self.batch_size==1:
            seenItems = self.userSeenItems[u[0]]

            x_ui = self.S[i, seenItems]
            x_uj = self.S[j, seenItems]

            # The difference is computed on the user_seen items
            x_uij = x_ui - x_uj

            #x_uij = x_uij[0,seenItems]
            x_uij = np.sum(x_uij)

            # log(sigm(+x_uij))
            gradient = 1 / (1 + np.exp(x_uij))

            # sigm(-x_uij)
            #exp = np.exp(x_uij)
            #gradient = exp/np.power(exp+1, 2)

        else:

            x_ui = self.S[i]
            x_uj = self.S[j]

            # The difference is computed on the user_seen items
            x_uij = x_ui - x_uj

            x_uij = self.URM_mask[u,:].dot(x_uij.T).diagonal()

            gradient = np.sum(1 / (1 + np.exp(x_uij))) / self.batch_size

        # Sigmoid whose argument is minus in order for the exponent of the exponential to be positive
        # Best performance with: gradient = np.sum(expit(-x_uij)) / self.batch_size
        #gradient = np.sum(x_uij) / self.batch_size
        #gradient = expit(-gradient)
        #gradient = np.sum(expit(-x_uij)) / self.batch_size
        #gradient = np.sum(np.log(expit(x_uij))) / self.batch_size
        #gradient = np.sum(1/(1+np.exp(x_uij))) / self.batch_size
        #gradient = min(10, max(-10, gradient))+10


        if self.batch_size==1:

            userSeenItems = self.userSeenItems[u[0]]

            self.S[i, userSeenItems] += self.learning_rate * gradient
            self.S[i, i] = 0

            self.S[j, userSeenItems] -= self.learning_rate * gradient
            self.S[j, j] = 0



        else:
            itemsToUpdate = np.array(self.URM_mask[u, :].sum(axis=0) > 0).ravel()

            # Do not update items i, set all user-posItem to false
            # itemsToUpdate[i] = False

            self.S[i] += self.learning_rate * gradient * itemsToUpdate
            self.S[i, i] = 0

            # Now update i, setting all user-posItem to true
            # Do not update j

            # itemsToUpdate[i] = True
            # itemsToUpdate[j] = False

            self.S[j] -= self.learning_rate * gradient * itemsToUpdate
            self.S[j, j] = 0



    def fit(self, epochs=30, logFile=None, URM_test=None, filterTopPop = False, minRatingsPerUser=1,
            batch_size = 1000, validate_every_N_epochs = 1, start_validation_after_N_epochs = 0,
            lambda_i = 0.0025, lambda_j = 0.00025, learning_rate = 0.05, topK = False):



        if self.sparse_weights:
            self.S = sps.csr_matrix((self.n_items, self.n_items), dtype=np.float32)
        else:
            self.S = np.zeros((self.n_items, self.n_items)).astype('float32')


        self.initializeFastSampling(positive_threshold=self.positive_threshold)



        if(topK != False and topK<1):
            raise ValueError("TopK not valid. Acceptable values are either False or a positive integer value. Provided value was '{}'".format(topK))
        self.topK = topK


        self.batch_size = batch_size
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate


        start_time_train = time.time()

        for currentEpoch in range(epochs):

            start_time_epoch = time.time()

            if self.batch_size>0:
                self.epochIteration()
            else:
                print("No batch not available")


            if (URM_test is not None) and ((currentEpoch +1 )% validate_every_N_epochs == 0) and \
                            currentEpoch+1 >= start_validation_after_N_epochs:

                print("Evaluation begins")

                self.updateSimilarityMatrix()

                results_run = self.evaluateRecommendations(URM_test, filterTopPop=filterTopPop,
                                                           minRatingsPerUser=minRatingsPerUser)

                self.writeCurrentConfig(currentEpoch+1, results_run, logFile)

                print("Epoch {} of {} complete in {:.2f} minutes".format(currentEpoch+1, epochs,
                                                                     float(time.time() - start_time_epoch) / 60))


            # Fit with no validation
            else:
                print("Epoch {} of {} complete in {:.2f} minutes".format(currentEpoch+1, epochs,
                                                                         float(time.time() - start_time_epoch) / 60))

        self.updateSimilarityMatrix()

        print("Fit completed in {:.2f} minutes".format(float(time.time() - start_time_train) / 60))

        sys.stdout.flush()



    def writeCurrentConfig(self, currentEpoch, results_run, logFile):

        current_config = {'lambda_i': self.lambda_i,
                          'lambda_j': self.lambda_j,
                          'batch_size': self.batch_size,
                          'learn_rate': self.learning_rate,
                          'topK_similarity': self.topK,
                          'epoch': currentEpoch}

        print("Test case: {}\nResults {}\n".format(current_config, results_run))
        # print("Weights: {}\n".format(str(list(self.weights))))

        sys.stdout.flush()

        if (logFile != None):
            logFile.write("Test case: {}, Results {}\n".format(current_config, results_run))
            # logFile.write("Weights: {}\n".format(str(list(self.weights))))
            logFile.flush()




    def epochIteration(self):

        # Get number of available interactions
        numPositiveIteractions = int(self.URM_mask.nnz*1)

        start_time_epoch = time.time()
        start_time_batch = time.time()

        totalNumberOfBatch = int(numPositiveIteractions/self.batch_size)+1

        # Uniform user sampling without replacement
        for numCurrentBatch in range(totalNumberOfBatch):

            sgd_users, sgd_pos_items, sgd_neg_items = self.sampleBatch()

            self.updateWeightsBatch(
                sgd_users,
                sgd_pos_items,
                sgd_neg_items
                )

            """
            self.updateWeightsLoop(
                sgd_users,
                sgd_pos_items,
                sgd_neg_items
                )
            """

            if(time.time() - start_time_batch >= 30 or numCurrentBatch==totalNumberOfBatch-1):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
                    numCurrentBatch*self.batch_size,
                    100.0* float(numCurrentBatch*self.batch_size)/numPositiveIteractions,
                    time.time() - start_time_batch,
                    float(numCurrentBatch*self.batch_size + 1) / (time.time() - start_time_epoch)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_batch = time.time()



        self.S[np.arange(0, self.n_items), np.arange(0, self.n_items)] = 0.0



