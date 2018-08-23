"""
Created on 02/01/2018

@author: Maurizio Ferrari Dacrema
"""

#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: language_level=3
#cython: nonecheck=False
#cython: cdivision=True
#cython: unpack_method_calls=True
#cython: overflowcheck=False

from Base.Recommender_utils import check_matrix
import time, sys

import numpy as np
cimport numpy as np


from libc.math cimport exp, sqrt, log, pow
from libc.stdlib cimport rand, RAND_MAX





cdef struct BPR_sample:
    long user
    long pos_item
    long neg_item
    long seen_items_start_pos
    long seen_items_end_pos


cdef struct SGD_sample:
    long user
    long item
    double value
    long seen_items_start_pos
    long seen_items_end_pos



cdef class RP3beta_ML_Cython:

    cdef int [:] URM_indices, URM_indptr, W_sparse_indices, W_sparse_indptr
    cdef double [:] URM_data, W_sparse_data

    cdef long n_users, n_items



    def __init__(self, URM_train, W_sparse):

        super(RP3beta_ML_Cython, self).__init__()


        URM_train = check_matrix(URM_train, 'csr')
        W_sparse = check_matrix(W_sparse, 'csr')

        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]

        self.URM_indices = URM_train.indices
        self.URM_indptr = URM_train.indptr
        self.URM_data = np.array(URM_train.data, dtype=float)

        self.W_sparse_indices = W_sparse.indices
        self.W_sparse_indptr = W_sparse.indptr
        self.W_sparse_data = np.array(W_sparse.data, dtype=float)


    def fit(self, epochs=10, learn_rate = 1e-4,
            initialItemsDegree = None, useAdaGrad = False, objective = "BPR"):


        totSamples = len(self.URM_data)

        if RAND_MAX < totSamples:
            print("Warning: your C distribution has a RAND_MAX value of {} "
                  "while the number of samples is {}. Data points exceeding RAND_MAX "
                  "will not be sampled.".format(RAND_MAX, totSamples))


        if objective=="BPR":
            return self.fit_BPR(epochs=epochs, learn_rate = learn_rate,
                        initialItemsDegree = initialItemsDegree, useAdaGrad = useAdaGrad)


        elif objective=="RMSE":
            return self.fit_RMSE(epochs=epochs, learn_rate = learn_rate,
                        initialItemsDegree = initialItemsDegree, useAdaGrad = useAdaGrad)

        else:
            raise ValueError("Value for 'objective' not recognized. "
                             "Accepted values are 'BPR', 'RMSE'. Provided was '{}'".format(objective))



    cpdef fit_RMSE(self, int epochs=10, double learn_rate = 1e-4,
            initialItemsDegree = None, int useAdaGrad = False):

        cdef double [:] itemsDegree

        cdef long currentEpoch, numSample
        cdef long seenItemIndex, seenItem_id, weightItemIndex, weightItem_id
        cdef double error, loss, prediction, no_degree_prediction

        cdef double [:] sgd_cache
        cdef SGD_sample sample



        if initialItemsDegree == None:
            itemsDegree = np.ones(self.n_items)
        else:
            itemsDegree = initialItemsDegree

        if useAdaGrad:
            sgd_cache = np.zeros((self.n_items), dtype=float)



        cdef long totSamples = len(self.URM_data)



        for currentEpoch in range(epochs):

            loss = 0
            start_time = time.time()


            for numSample in range(totSamples):

                sample = self.sampleSGD_Cython()


                prediction = 0.0
                no_degree_prediction = 0.0

                # Get the first item in the user profile
                seenItemIndex = sample.seen_items_start_pos

                # Get the first non-zero weight
                weightItemIndex = self.W_sparse_indptr[sample.item]
                weightItem_id = self.W_sparse_indices[weightItemIndex]

                # For every element in the user profile
                while seenItemIndex < sample.seen_items_end_pos:

                    seenItem_id = self.URM_indices[seenItemIndex]

                    # Loop until the correct weight is found
                    # Indices are ordered, continue from the last position
                    while weightItem_id < seenItem_id and weightItemIndex < self.W_sparse_indptr[sample.item+1]-1:
                        weightItemIndex += 1
                        weightItem_id = self.W_sparse_indices[weightItemIndex]

                    # If weight is not null
                    if weightItem_id == seenItem_id:
                        prediction += self.URM_data[seenItemIndex]*self.W_sparse_data[weightItemIndex]*itemsDegree[seenItem_id]
                        no_degree_prediction += self.URM_data[seenItemIndex]*self.W_sparse_data[weightItemIndex]


                    seenItemIndex += 1



                error = sample.value - prediction
                loss += error**2

                gradient = error * no_degree_prediction



                # Get the first item in the user profile
                seenItemIndex = sample.seen_items_start_pos

                # For every element in the user profile
                while seenItemIndex < sample.seen_items_end_pos:

                    seenItem_id = self.URM_indices[seenItemIndex]

                    if useAdaGrad:

                        sgd_cache[seenItem_id] += gradient ** 2

                        itemsDegree[seenItem_id] += gradient * learn_rate / (sqrt(sgd_cache[seenItem_id]) + 1e-8)

                    else:
                        itemsDegree[seenItem_id] += gradient * learn_rate


                    if itemsDegree[seenItem_id] < 0:
                       itemsDegree[seenItem_id] = 0.0


                    seenItemIndex += 1





                if (numSample%5000000 == 0 and numSample!=0) or numSample==totSamples-1:
                    print("Samples processed {:.0f} ( {:.2f}%) in {:.2f} minutes. Loss is {:.2E}. Samples per second: {:.0f}".format(
                            numSample,
                            100*numSample/(totSamples-1),
                            (time.time() - start_time) / 60,
                            loss/numSample,
                            numSample / (time.time() - start_time)))

                    sys.stdout.flush()
                    sys.stderr.flush()


            print("Epoch {} complete in {:.2f} minutes, loss is {:.2E}".format(currentEpoch+1, (time.time() - start_time) / 60, loss/totSamples))


        return np.array(itemsDegree)






    cpdef fit_BPR(self, int epochs=10, double learn_rate = 1e-4,
            initialItemsDegree = None, int useAdaGrad = False):

        cdef double [:] itemsDegree

        cdef long currentEpoch, numSample
        cdef long seenItemIndex, seenItem_id
        cdef long weightItemIndex_pos, weightItem_id_pos, weightItemIndex_neg, weightItem_id_neg
        cdef double x_ij, no_degree_prediction, delta_weights, loss

        cdef double [:] sgd_cache
        cdef BPR_sample sample



        if initialItemsDegree == None:
            itemsDegree = np.ones(self.n_items)
        else:
            itemsDegree = initialItemsDegree

        if useAdaGrad:
            sgd_cache = np.zeros((self.n_items), dtype=float)



        cdef long totSamples = len(self.URM_data)


        for currentEpoch in range(epochs):

            loss = 0
            start_time = time.time()


            for numSample in range(totSamples):

                sample = self.sampleBPR_Cython()


                x_ij = 0.0
                no_degree_prediction = 0.0



                # Get the first item in the user profile
                seenItemIndex = sample.seen_items_start_pos

                # Get the first non-zero weight
                weightItemIndex_pos = self.W_sparse_indptr[sample.pos_item]
                weightItem_id_pos = self.W_sparse_indices[weightItemIndex_pos]

                weightItemIndex_neg = self.W_sparse_indptr[sample.neg_item]
                weightItem_id_neg = self.W_sparse_indices[weightItemIndex_neg]


                # For every element in the user profile
                while seenItemIndex <sample.seen_items_end_pos:

                    seenItem_id = self.URM_indices[seenItemIndex]

                    delta_weights = 0

                    # Loop until the correct weight is found
                    # Indices are ordered, continue from the last position

                    # Positive item
                    while weightItem_id_pos < seenItem_id and weightItemIndex_pos < self.W_sparse_indptr[sample.pos_item+1]-1:
                        weightItemIndex_pos += 1
                        weightItem_id_pos = self.W_sparse_indices[weightItemIndex_pos]

                    if weightItem_id_pos == seenItem_id:
                        delta_weights += self.W_sparse_data[weightItemIndex_pos]


                    # Negative item
                    while weightItem_id_neg < seenItem_id and weightItemIndex_neg < self.W_sparse_indptr[sample.neg_item+1]-1:
                        weightItemIndex_neg += 1
                        weightItem_id_neg = self.W_sparse_indices[weightItemIndex_neg]

                    if weightItem_id_neg == seenItem_id:
                        delta_weights -= self.W_sparse_data[weightItemIndex_neg]



                    # If delta_weights is not zero
                    if delta_weights != 0:

                        x_ij += self.URM_data[seenItemIndex] * delta_weights * itemsDegree[seenItem_id]
                        no_degree_prediction += self.URM_data[seenItemIndex] * delta_weights

                    seenItemIndex += 1


                x_uij = 1 / (1 + exp(x_ij))
                loss += x_uij**2

                gradient = x_uij * no_degree_prediction



                # Get the first item in the user profile
                seenItemIndex = sample.seen_items_start_pos

                # For every element in the user profile
                while seenItemIndex < sample.seen_items_end_pos:

                    seenItem_id = self.URM_indices[seenItemIndex]

                    if useAdaGrad:

                        sgd_cache[seenItem_id] += gradient ** 2

                        itemsDegree[seenItem_id] += gradient * learn_rate / (sqrt(sgd_cache[seenItem_id]) + 1e-8)

                    else:
                        itemsDegree[seenItem_id] += gradient * learn_rate


                    if itemsDegree[seenItem_id] < 0:
                       itemsDegree[seenItem_id] = 0.0


                    seenItemIndex += 1





                if (numSample%5000000 == 0 and numSample!=0) or numSample==totSamples-1:
                    print("Samples processed {:.0f} ( {:.2f}%) in {:.2f} minutes. BPR loss is {:.2E}. Samples per second: {:.0f}".format(
                            numSample,
                            100*numSample/(totSamples-1),
                            (time.time() - start_time) / 60,
                            loss/numSample,
                            numSample / (time.time() - start_time)))

                    sys.stdout.flush()
                    sys.stderr.flush()


            print("Epoch {} complete in {:.2f} minutes. BPR loss is {:.2E}.".format(currentEpoch+1, (time.time() - start_time) / 60, loss/numSample))



        return np.array(itemsDegree)






    cdef SGD_sample sampleSGD_Cython(self):

        cdef SGD_sample sample = SGD_sample(-1,-1,-1.0,-1,-1)
        cdef long index
        cdef int numSeenItems = 0

        # Skip users with no interactions
        while numSeenItems == 0:

            sample.user = rand() % self.n_users

            sample.seen_items_start_pos = self.URM_indptr[sample.user]
            sample.seen_items_end_pos = self.URM_indptr[sample.user+1]

            numSeenItems = sample.seen_items_end_pos - sample.seen_items_start_pos


        index = rand() % numSeenItems

        sample.item = self.URM_indices[sample.seen_items_start_pos + index]
        sample.value = self.URM_data[sample.seen_items_start_pos + index]


        return sample




    cdef BPR_sample sampleBPR_Cython(self):

        cdef BPR_sample sample = BPR_sample(-1,-1,-1,-1,-1)
        cdef long index

        cdef int negItemSelected, numSeenItems = 0

        # Skip users with no interactions or with no negative items
        while numSeenItems == 0 or numSeenItems == self.n_items:

            sample.user = rand() % self.n_users

            sample.seen_items_start_pos = self.URM_indptr[sample.user]
            sample.seen_items_end_pos = self.URM_indptr[sample.user+1]

            numSeenItems = sample.seen_items_end_pos - sample.seen_items_start_pos


        index = rand() % numSeenItems

        sample.pos_item = self.URM_indices[sample.seen_items_start_pos + index]



        negItemSelected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        # for every user
        while (not negItemSelected):

            sample.neg_item = rand() % self.n_items

            index = 0
            while index < numSeenItems and self.URM_indices[sample.seen_items_start_pos + index]!=sample.neg_item:
                index+=1

            if index == numSeenItems:
                negItemSelected = True


        return sample
