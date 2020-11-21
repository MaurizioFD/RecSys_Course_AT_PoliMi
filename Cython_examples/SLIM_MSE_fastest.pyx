"""
Created on 09/11/2020

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import time

from libc.stdlib cimport rand, srand, RAND_MAX

# These can be used to remove checks done at runtime (e.g. null pointers etc). Be careful as they can introduce errors
# For example cdivision performs the C division which can result in undesired integer divisions where
# floats are instead required
import cython
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def train_multiple_epochs(URM_train, learning_rate_input, n_epochs):

    URM_train_coo = URM_train.tocoo()
    cdef int n_items = URM_train.shape[1]
    cdef int n_interactions = URM_train.nnz
    cdef int[:] URM_train_coo_row = URM_train_coo.row
    cdef int[:] URM_train_coo_col = URM_train_coo.col
    cdef double[:] URM_train_coo_data = URM_train_coo.data
    cdef int[:] URM_train_indices = URM_train.indices
    cdef int[:] URM_train_indptr = URM_train.indptr
    cdef double[:] URM_train_data = URM_train.data

    cdef double[:,:] item_item_S = np.zeros((n_items, n_items), dtype = np.float)
    cdef double learning_rate = learning_rate_input
    cdef double loss = 0.0
    cdef long start_time
    cdef double true_rating, predicted_rating, prediction_error, profile_rating
    cdef int start_profile, end_profile
    cdef int index, sample_num, user_id, item_id, profile_item_id

    for n_epoch in range(n_epochs):

        loss = 0.0
        start_time = time.time()

        for sample_num in range(n_interactions):

            # Randomly pick sample
            index = rand() % n_interactions

            user_id = URM_train_coo_row[index]
            item_id = URM_train_coo_col[index]
            true_rating = URM_train_coo_data[index]

            # Compute prediction
            start_profile = URM_train_indptr[user_id]
            end_profile = URM_train_indptr[user_id + 1]
            predicted_rating = 0.0

            for index in range(start_profile, end_profile):
                profile_item_id = URM_train_indices[index]
                profile_rating = URM_train_data[index]
                predicted_rating += item_item_S[profile_item_id, item_id] * profile_rating

            # Compute prediction error, or gradient
            prediction_error = true_rating - predicted_rating
            loss += prediction_error ** 2

            # Update model, in this case the similarity
            for index in range(start_profile, end_profile):
                profile_item_id = URM_train_indices[index]
                profile_rating = URM_train_data[index]
                item_item_S[profile_item_id, item_id] += learning_rate * prediction_error * profile_rating

        #             # Print some stats
        #             if (sample_num +1)% 1000000 == 0:
        #                 elapsed_time = time.time() - start_time
        #                 samples_per_second = (sample_num+1)/elapsed_time
        #                 print("Iteration {} in {:.2f} seconds, loss is {:.2f}. Samples per second {:.2f}".format(sample_num+1, elapsed_time, loss/(sample_num+1), samples_per_second))

        elapsed_time = time.time() - start_time
        samples_per_second = (sample_num + 1) / elapsed_time

        print("Epoch {} complete in in {:.2f} seconds, loss is {:.3E}. Samples per second {:.2f}".format(n_epoch + 1,
                                                                                                         time.time() - start_time,
                                                                                                         loss / (sample_num + 1),
                                                                                                         samples_per_second))

    return np.array(item_item_S), loss / (sample_num + 1), samples_per_second