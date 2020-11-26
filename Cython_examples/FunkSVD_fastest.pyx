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
    n_users, n_items = URM_train_coo.shape
    cdef int n_interactions = URM_train.nnz

    cdef int sample_num, sample_index, user_id, item_id, factor_index
    cdef double rating, predicted_rating, prediction_error

    cdef int num_factors = 10
    cdef double learning_rate = 1e-4
    cdef double regularization = 1e-5

    cdef int[:] URM_train_coo_row = URM_train_coo.row
    cdef int[:] URM_train_coo_col = URM_train_coo.col
    cdef double[:] URM_train_coo_data = URM_train_coo.data

    cdef double[:,:] user_factors = np.random.random((n_users, num_factors))
    cdef double[:,:] item_factors = np.random.random((n_items, num_factors))
    cdef double H_i, W_u
    cdef double item_factors_update, user_factors_update

    cdef double loss = 0.0
    cdef long start_time = time.time()

    for n_epoch in range(n_epochs):

        loss = 0.0
        start_time = time.time()

        for sample_num in range(n_interactions):

            # Randomly pick sample
            sample_index = rand() % n_interactions

            user_id = URM_train_coo_row[sample_index]
            item_id = URM_train_coo_col[sample_index]
            rating = URM_train_coo_data[sample_index]

            # Compute prediction
            predicted_rating = 0.0

            for factor_index in range(num_factors):
                predicted_rating += user_factors[user_id, factor_index] * item_factors[item_id, factor_index]

            # Compute prediction error, or gradient
            prediction_error = rating - predicted_rating
            loss += prediction_error**2

            # Copy original value to avoid messing up the updates
            for factor_index in range(num_factors):

                H_i = item_factors[item_id,factor_index]
                W_u = user_factors[user_id,factor_index]

                user_factors_update = prediction_error * H_i - regularization * W_u
                item_factors_update = prediction_error * W_u - regularization * H_i

                user_factors[user_id,factor_index] += learning_rate * user_factors_update
                item_factors[item_id,factor_index] += learning_rate * item_factors_update

#             # Print some stats
#             if (sample_num +1)% 1000000 == 0:
#                 elapsed_time = time.time() - start_time
#                 samples_per_second = (sample_num+1)/elapsed_time
#                 print("Iteration {} in {:.2f} seconds, loss is {:.2f}. Samples per second {:.2f}".format(sample_num+1, elapsed_time, loss/(sample_num+1), samples_per_second))


        elapsed_time = time.time() - start_time
        samples_per_second = (sample_num+1)/elapsed_time

        print("Epoch {} complete in in {:.2f} seconds, loss is {:.3E}. Samples per second {:.2f}".format(n_epoch+1, time.time() - start_time, loss/(sample_num+1), samples_per_second))

    return np.array(user_factors), np.array(item_factors), loss/(sample_num+1), samples_per_second