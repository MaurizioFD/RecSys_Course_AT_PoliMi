"""
Created on 09/11/2020

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import time

def train_multiple_epochs(URM_train, learning_rate_input, n_epochs):

    URM_train_coo = URM_train.tocoo()
    n_items = URM_train.shape[1]
    n_interactions = URM_train.nnz

    item_item_S = np.zeros((n_items, n_items), dtype = np.float16)

    learning_rate = learning_rate_input


    for n_epoch in range(n_epochs):

        loss = 0.0
        start_time = time.time()

        for sample_num in range(n_interactions):

            # Randomly pick sample
            sample_index = np.random.randint(URM_train_coo.nnz)

            user_id = URM_train_coo.row[sample_index]
            item_id = URM_train_coo.col[sample_index]
            true_rating = URM_train_coo.data[sample_index]

            # Compute prediction
            predicted_rating = URM_train[user_id].dot(item_item_S[:,item_id])[0]

            # Compute prediction error, or gradient
            prediction_error = true_rating - predicted_rating
            loss += prediction_error**2

            # Update model, in this case the similarity
            items_in_user_profile = URM_train[user_id].indices
            ratings_in_user_profile = URM_train[user_id].data
            item_item_S[items_in_user_profile, item_id] += learning_rate * prediction_error * ratings_in_user_profile

            # Print some stats
            if (sample_num +1)% 5000 == 0:
                elapsed_time = time.time() - start_time
                samples_per_second = (sample_num+1)/elapsed_time
                print("Iteration {} in {:.2f} seconds, loss is {:.2f}. Samples per second {:.2f}".format(sample_num+1, elapsed_time, loss/(sample_num+1), samples_per_second))


        elapsed_time = time.time() - start_time
        samples_per_second = (sample_num+1)/elapsed_time

        print("Epoch {} complete in in {:.2f} seconds, loss is {:.3E}. Samples per second {:.2f}".format(n_epoch+1, time.time() - start_time, loss/(sample_num+1), samples_per_second))

    return np.array(item_item_S), loss/(sample_num+1), samples_per_second