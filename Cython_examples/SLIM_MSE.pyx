"""
Created on 09/11/2020

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import time

def do_some_training(URM_train):

    URM_train_coo = URM_train.tocoo()
    n_items = URM_train.shape[1]

    cdef double[:,:] item_item_S = np.zeros((n_items, n_items), dtype = np.float)
    cdef double learning_rate = 1e-6
    cdef double loss = 0.0
    cdef long start_time = time.time()
    cdef double predicted_rating, prediction_error
    cdef int[:] items_in_user_profile
    cdef int index, sample_num

    for sample_num in range(100000):

        # Randomly pick sample
        sample_index = np.random.randint(URM_train_coo.nnz)

        user_id = URM_train_coo.row[sample_index]
        item_id = URM_train_coo.col[sample_index]
        rating = URM_train_coo.data[sample_index]

        # Compute prediction
        items_in_user_profile = URM_train.indices[URM_train.indptr[user_id]:URM_train.indptr[user_id+1]]
        predicted_rating = 0.0

        for index in items_in_user_profile:
            predicted_rating += item_item_S[index,item_id]

        # Compute prediction error, or gradient
        prediction_error = rating - predicted_rating
        loss += prediction_error**2

        # Update model, in this case the similarity
        for index in items_in_user_profile:
            item_item_S[index,item_id] += prediction_error * learning_rate

        # Print some stats
        if (sample_num +1)% 5000 == 0:
            elapsed_time = time.time() - start_time
            samples_per_second = sample_num/elapsed_time
            print("Iteration {} in {:.2f} seconds, loss is {:.2f}. Samples per second {:.2f}".format(sample_num+1, elapsed_time, loss/sample_num, samples_per_second))

    return loss, samples_per_second