import numpy as np
import time

def do_some_training(URM_train):

    URM_train_coo = URM_train.tocoo()
    n_users, n_items = URM_train_coo.shape

    num_factors = 10
    learning_rate = 1e-4
    regularization = 1e-5

    user_factors = np.random.random((n_users, num_factors))
    item_factors = np.random.random((n_items, num_factors))

    loss = 0.0
    start_time = time.time()

    for sample_num in range(1000000):

        # Randomly pick sample
        sample_index = np.random.randint(URM_train_coo.nnz)

        user_id = URM_train_coo.row[sample_index]
        item_id = URM_train_coo.col[sample_index]
        rating = URM_train_coo.data[sample_index]

        # Compute prediction
        predicted_rating = np.dot(user_factors[user_id,:], item_factors[item_id,:])

        # Compute prediction error, or gradient
        prediction_error = rating - predicted_rating
        loss += prediction_error**2

        # Copy original value to avoid messing up the updates
        H_i = item_factors[item_id,:]
        W_u = user_factors[user_id,:]

        user_factors_update = prediction_error * H_i - regularization * W_u
        item_factors_update = prediction_error * W_u - regularization * H_i

        user_factors[user_id,:] += learning_rate * user_factors_update
        item_factors[item_id,:] += learning_rate * item_factors_update

        # Print some stats
        if (sample_num +1)% 100000 == 0:
            elapsed_time = time.time() - start_time
            samples_per_second = (sample_num+1)/elapsed_time
            print("Iteration {} in {:.2f} seconds, loss is {:.2f}. Samples per second {:.2f}".format(sample_num+1, elapsed_time, loss/(sample_num+1), samples_per_second))

    return loss/(sample_num+1), samples_per_second