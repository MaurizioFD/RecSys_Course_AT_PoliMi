#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 28 June 2017

@author: Maurizio Ferrari Dacrema
"""

import sys
import time

import numpy as np
import theano
import theano.sparse
import theano.tensor as T
from Recommender_utils import similarityMatrixTopK

from BPR.BPR_Sampling import BPR_Sampling
from Base.Recommender import Recommender


class SLIM_BPR_Theano(BPR_Sampling, Recommender):

    def __init__(self, URM_train, lambda_i = 0.0025, lambda_j = 0.00025, learning_rate = 0.05,
                 topK = False):
        super(SLIM_BPR_Theano, self).__init__()

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

        if(topK != False and topK<1):
            raise ValueError("TopK not valid. Acceptable values are either False or a positive integer value. Provided value was '{}'".format(topK))
        self.topK = topK


        self.URM_train = URM_train
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate
        self.normalize = False
        self.sparse_weights = False

        self._configure_theano()


    def _configure_theano(self):
        """
          Configures Theano to run in fast mode
          and using 32-bit floats. 
        """

        # Compile highly optimized code
        theano.config.mode = 'FAST_RUN'

        # Default float
        theano.config.floatX = 'float32'

        # Enable multicore
        theano.config.openmp = 'true'

        #theano.config.exception_verbosity='high'
        #theano.config.optimizer = 'None'
        #theano.config.compute_test_value = 'off'
        #theano.config.cxx = ''
        theano.config.on_unused_input='ignore'


    def _clear_theano_train_data(self):
        """
        Delete unnecessary data structures required by the training phase
        :return: -
        """

        del self.S
        del self.URM_mask
        #del self.train_model


    def _generate_train_model_function_Batch(self):
        """
        Define the update rules to be used in the train phase and compile the train function
        :return:
        """

        u = theano.tensor.lvector('u')
        i = theano.tensor.lvector('i')
        j = theano.tensor.lvector('j')

        self.S = np.zeros((self.n_items, self.n_items)).astype('float32')
        self.S = theano.shared(self.S, name='S')

        # This URM is a boolean mask
        #self.URM_mask = theano.shared((self.URM_train.toarray()>0).astype('int8'), name='URM')
        #self.URM_mask = theano.shared((self.URM_train > 0).astype('int8'), name='URM')

        # The index refers to the rows, therefore matrix[vector] = vector.shape[0] x matrix.shape[1]
        # vector.shape[0] represents the size of the batch
        # No product with URM is required here

        x_ui = self.S[i]
        x_uj = self.S[j]

        # The difference is computed on the whole row not only on the user_seen items
        # The performance seems to be higher this way
        x_uij = x_ui - x_uj

        # Sigmoid whose argument is minus in order for the exponent of the exponential to be positive
        gradient = -x_uij.sum(axis=0) / self.batch_size
        gradient = T.nnet.sigmoid(gradient)


        # Select only items seen by the user
        #itemsToUpdate = self.URM_mask[u]


        # Do not update items i, set all user-posItem to false
        #itemsToUpdate = T.set_subtensor(itemsToUpdate[np.arange(0,self.batch_size),i], 0)

        delta_i = gradient#-self.lambda_i*self.S[i]
        delta_j = -gradient#-self.lambda_j*self.S[j]

        # Since a shared variable may be the target of only one update rule
        # All the required updates are chained inside a subtensor
        updateChain = self.S
        updateChain = T.inc_subtensor(updateChain[i], self.learning_rate * delta_i)

        # Now update i, setting all user-posItem to true
        # Do not update j
        #itemsToUpdate = T.set_subtensor(itemsToUpdate[np.arange(0,self.batch_size),i], 1)
        #itemsToUpdate = T.set_subtensor(itemsToUpdate[np.arange(0,self.batch_size),j], 0)

        updateChain = T.inc_subtensor(updateChain[j], self.learning_rate * delta_j)

        updates = [(self.S, updateChain)]

        # Create and compile the train function
        self.train_model = theano.function(inputs=[u, i, j], updates=updates)


    def _generate_train_model_function(self):
        """
        Define the update rules to be used in the train phase and compile the train function
        :return:
        """

        u = theano.tensor.lvector('u')
        i = theano.tensor.lvector('i')
        j = theano.tensor.lvector('j')

        localS = np.random.random((self.n_items, self.n_items)).astype('float32')
        localS[np.arange(0, self.n_items), np.arange(0, self.n_items)] = 0.0


        self.S = theano.shared(localS, name='S')

        # This URM is a boolean mask
        self.URM_mask = theano.shared((self.URM_train.toarray()>0).astype('int8'), name='URM')
        #self.URM_mask = theano.shared((self.URM_train > 0).astype('int8'), name='URM')

        # The index refers to the rows, therefore matrix[vector] = vector.shape[0] x matrix.shape[1]
        # vector.shape[0] represents the size of the batch
        # No product with URM is required here

        x_ui = self.S[i]
        x_uj = self.S[j]

        # The difference is computed on the whole row not only on the user_seen items
        # The performance seems to be higher this way
        x_uij = x_ui - x_uj

        # Sigmoid whose argument is minus in order for the exponent of the exponential to be positive
        sigmoid = T.nnet.sigmoid(-x_uij)

        # Select only items seen by the user
        itemsToUpdate = self.URM_mask[u]


        # Do not update items i, set all user-posItem to false
        itemsToUpdate = T.set_subtensor(itemsToUpdate[np.arange(0,self.batch_size),i], 0)

        delta_i = sigmoid-self.lambda_i*self.S[i]
        delta_j = -sigmoid-self.lambda_j*self.S[j]

        # Since a shared variable may be the target of only one update rule
        # All the required updates are chained inside a subtensor
        updateChain = self.S
        updateChain = T.inc_subtensor(updateChain[i], (self.learning_rate * delta_i) * itemsToUpdate)

        # Now update i, setting all user-posItem to true
        # Do not update j
        itemsToUpdate = T.set_subtensor(itemsToUpdate[np.arange(0,self.batch_size),i], 1)
        itemsToUpdate = T.set_subtensor(itemsToUpdate[np.arange(0,self.batch_size),j], 0)

        updateChain = T.inc_subtensor(updateChain[j], (self.learning_rate * delta_j) * itemsToUpdate)

        updates = [(self.S, updateChain)]

        # Create and compile the train function
        self.train_model = theano.function(inputs=[u, i, j], updates=updates)


    def _generate_train_model_function_NoBatch(self):
        """
        Define the update rules to be used in the train phase and compile the train function
        :return:
        """

        u = theano.tensor.scalar('u', dtype='int32')
        i = theano.tensor.scalar('i', dtype='int32')
        j = theano.tensor.scalar('j', dtype='int32')

        localS = np.random.random((self.n_items, self.n_items)).astype('float32')
        localS[np.arange(0, self.n_items), np.arange(0, self.n_items)] = 0.0


        self.S = theano.shared(localS, name='S')

        # This URM is a boolean mask
        #self.URM_mask = theano.shared((self.URM_train.toarray()>0).astype('int8'), name='URM')
        self.URM_mask = theano.shared((self.URM_train > 0).astype('int8'), name='URM')

        # The index refers to the rows, therefore matrix[vector] = vector.shape[0] x matrix.shape[1]
        # vector.shape[0] represents the size of the batch
        # No product with URM is required here

        x_ui = self.S[i,:]
        x_uj = self.S[j,:]

        # The difference is computed on the whole row not only on the user_seen items
        # The performance seems to be higher this way
        x_uij = x_ui - x_uj

        # Sigmoid whose argument is minus in order for the exponent of the exponential to be positive
        sigmoid = T.nnet.sigmoid(-x_uij)

        # Select only items seen by the user
        itemsToUpdate = self.URM_mask[u:u+1,0:self.n_items]

        itemsToUpdate = theano.sparse.dense_from_sparse(itemsToUpdate)
        itemsToUpdate = T.reshape(itemsToUpdate, [self.n_items])

        # Do not update items i, set all user-posItem to false
        itemsToUpdate = T.set_subtensor(itemsToUpdate[i], 0)

        delta_i = sigmoid-self.lambda_i*self.S[i]
        delta_j = -sigmoid-self.lambda_j*self.S[j]

        # Since a shared variable may be the target of only one update rule
        # All the required updates are chained inside a subtensor
        updateChain = self.S
        updateChain = T.inc_subtensor(updateChain[i], (self.learning_rate * delta_i) * itemsToUpdate)

        # Now update i, setting all user-posItem to true
        # Do not update j
        itemsToUpdate = T.set_subtensor(itemsToUpdate[i], 1)
        itemsToUpdate = T.set_subtensor(itemsToUpdate[j], 0)

        updateChain = T.inc_subtensor(updateChain[j], (self.learning_rate * delta_j) * itemsToUpdate)

        updates = [(self.S, updateChain)]

        # Create and compile the train function
        self.train_model_noBatch = theano.function(inputs=[u, i, j], updates=updates)



    def fit(self, epochs=30, batch_size=1000):
        """
        Train SLIM wit BPR. If the model was already trained, overwrites matrix S
        Training is performed via batch gradient descent
        :param epochs:
        :return: -
        """

        self.batch_size = batch_size

        if (self.batch_size == None):
            #self._generate_train_model_function_NoBatch()
            self._generate_train_model_function_NoBatch_Loops()
        else:
            self._generate_train_model_function_Batch()
            #self._generate_train_model_function()
            self.initializeFastSampling()


        start_time_train = time.time()

        for currentEpoch in range(epochs):

            start_time_epoch = time.time()

            if (self.batch_size == None):
                self.epochIterationNoBatch()
            else:
                self.epochIteration()

            print("Epoch {} of {} complete in {:.2f} minutes".format(currentEpoch+1, epochs, float(time.time()-start_time_epoch)/60))
            sys.stdout.flush()

        print("Train completed in {:.2f} minutes".format(float(time.time()-start_time_train)/60))

        # The similarity matrix is learnt row-wise
        # To be used in the product URM*S must be transposed to be column-wise


        self.W = self.S.get_value().T
        #self.W_sparse = sps.csr_matrix(self.S.get_value().T)

        self._clear_theano_train_data()


    def fitAndValidate(self, epochs=30, logFile=None, URM_test=None, filterTopPop = False, minRatingsPerUser=1,
            batch_size = 1000, validate_every_N_epochs = 1, start_validation_after_N_epochs = 0):
        """
        Fits the model performing a round of testing at the end of each epoch
        :param epochs:
        :param batch_size:
        :param logFile:
        :param URM_test:
        :return:
        """


        self.batch_size = batch_size

        if (self.batch_size == None):
            self._generate_train_model_function_NoBatch()
        else:
            self._generate_train_model_function_Batch()
            #self._generate_train_model_function()
            self.initializeFastSampling()


        start_time_train = time.time()

        for currentEpoch in range(epochs):

            start_time_epoch = time.time()

            if currentEpoch > 0:
                if self.batch_size>0:
                    self.epochIteration()
                else:
                    print("No batch not available")

            if self.topK != False:
                self.sparse_weights = True
                self.W_sparse = similarityMatrixTopK(self.S.get_value().T, k=self.topK)
            else:
                self.W = self.S.get_value().T

            if (URM_test is not None) and (currentEpoch % validate_every_N_epochs == 0) and \
                            currentEpoch >= start_validation_after_N_epochs:

                results_run = self.evaluateRecommendations(URM_test, filterTopPop=filterTopPop,
                                                           minRatingsPerUser=minRatingsPerUser)

                current_config = {'learn_rate': self.learning_rate,
                                  'batch': self.batch_size,
                                  'epoch': currentEpoch}

                print("Test case: {}\nResults {}\n".format(current_config, results_run))
                # print("Weights: {}\n".format(str(list(self.weights))))

                print("Epoch {} of {} complete in {:.2f} minutes".format(currentEpoch, epochs,
                                                                         float(time.time() - start_time_epoch) / 60))
                sys.stdout.flush()

                if (logFile != None):
                    logFile.write("Test case: {}, Results {}\n".format(current_config, results_run))
                    # logFile.write("Weights: {}\n".format(str(list(self.weights))))
                    logFile.flush()


            # Fit with no validation
            else:
                print("Epoch {} of {} complete in {:.2f} minutes".format(currentEpoch, epochs,
                                                                         float(time.time() - start_time_epoch) / 60))

        print("Fit completed in {:.2f} minutes".format(float(time.time() - start_time_train) / 60))


    def epochIteration(self):

        # Get number of available interactions
        numPositiveIteractions = self.URM_train.nnz

        start_time_epoch = time.time()
        start_time_batch = time.time()

        numBatch = int(numPositiveIteractions/self.batch_size)+1

        # Uniform user sampling without replacement
        for numSample in range(numBatch):

            sgd_users, sgd_pos_items, sgd_neg_items = self.sampleBatch()

            self.train_model(
                sgd_users,
                sgd_pos_items,
                sgd_neg_items
                )

            if(time.time() - start_time_batch >= 30 or numSample==numBatch-1):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
                    numSample*self.batch_size,
                    100.0* float(numSample*self.batch_size)/numPositiveIteractions,
                    time.time() - start_time_batch,
                    float(numSample*self.batch_size + 1) / (time.time() - start_time_epoch)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_batch = time.time()


    def epochIterationNoBatch(self):

        # Get number of available interactions
        numPositiveIteractions = self.URM_train.nnz

        start_time = time.time()

        # Uniform user sampling without replacement
        for numSample in range(numPositiveIteractions):

            user_id, pos_item_id, neg_item_id = self.sampleTriple()

            self.train_model_noBatch(
                user_id,
                pos_item_id,
                neg_item_id
                )

            if(numSample % 5000 == 0):
                print("Processed {} ( {:.2f}% ) in {:.4f} seconds".format(numSample,
                                  100.0* float(numSample)/numPositiveIteractions,
                                  time.time()-start_time))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time = time.time()
