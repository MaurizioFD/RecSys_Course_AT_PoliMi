#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 01/01/2023

@author: Maurizio Ferrari Dacrema
"""


import torch
import math

import numpy as np

import scipy.sparse as sps

# class InteractionIterator(object):
#     """
#     This Sampler samples among all the existing user-item interactions *uniformly at random*:
#     - One of the interactions in the dataset is sampled
#
#     The sample is: user_id, item_id, rating
#     """
#
#     def __init__(self, URM_train, batch_size = 1):
#         super(InteractionIterator, self).__init__()
#
#         self.URM_train = sps.coo_matrix(URM_train)
#         self.n_users, self.n_items = self.URM_train.shape
#         self.n_data_points = self.URM_train.nnz
#         self.n_sampled_points = 0
#         self.batch_size = batch_size
#
#     def __len__(self):
#         return math.ceil(self.n_data_points/self.batch_size)
#
#     def __iter__(self):
#         self.n_sampled_points = 0
#         return self
#
#     def __next__(self):
#
#         if self.n_sampled_points >= self.n_data_points:
#             raise StopIteration
#
#         this_batch_size = min(self.batch_size, self.n_data_points-self.n_sampled_points)
#         self.n_sampled_points += this_batch_size
#
#         index_batch = np.random.randint(self.n_data_points, size = this_batch_size)
#
#         return torch.from_numpy(self.URM_train.row[index_batch]).long(),\
#                torch.from_numpy(self.URM_train.col[index_batch]).long(), \
#                torch.from_numpy(self.URM_train.data[index_batch]).float()




class InteractionIterator(object):
    """
    This Sampler samples among all the existing user-item interactions *uniformly at random*:
    - One of the interactions in the dataset is sampled

    The sample is: user_id, item_id, rating
    """

    def __init__(self, URM_train, positive_quota, batch_size = 1, set_n_samples_to_draw = None):
        super(InteractionIterator, self).__init__()

        self.URM_train = sps.coo_matrix(URM_train)
        self.URM_train_row = torch.from_numpy(self.URM_train.row).long()
        self.URM_train_col = torch.from_numpy(self.URM_train.col).long()
        self.URM_train_data = torch.from_numpy(self.URM_train.data).float()

        self.URM_train = sps.csr_matrix(URM_train)
        self.URM_train = self.URM_train.sorted_indices()
        self.n_users, self.n_items = self.URM_train.shape
        self.batch_size = batch_size
        self.positive_quota = positive_quota

        self.n_samples_available = URM_train.nnz
        self.n_samples_to_draw = self.n_samples_available if set_n_samples_to_draw is None else set_n_samples_to_draw
        self.n_samples_drawn = 0

        self.batch_user = torch.empty((self.batch_size,), dtype=torch.long)
        self.batch_item = torch.empty((self.batch_size,), dtype=torch.long)
        self.batch_rating = torch.empty((self.batch_size,), dtype=torch.float)

    def __len__(self):
        return math.ceil(self.n_samples_to_draw/self.batch_size)

    def __iter__(self):
        self.n_sampled_points = 0
        return self

    def __next__(self):

        if self.n_sampled_points >= self.n_samples_to_draw:
            raise StopIteration

        this_batch_size = min(self.batch_size, self.n_samples_to_draw-self.n_sampled_points)
        self.n_sampled_points += this_batch_size

        index_batch = np.random.randint(self.n_samples_available, size = this_batch_size)

        self.batch_user[:this_batch_size] = self.URM_train_row[index_batch]
        self.batch_item[:this_batch_size] = self.URM_train_col[index_batch]
        self.batch_rating[:this_batch_size] = self.URM_train_data[index_batch]

        negative_interaction_flag = np.random.rand(self.batch_size) > self.positive_quota

        for i_batch in negative_interaction_flag.nonzero()[0]:

            user_id = self.batch_user[i_batch]

            start_pos_seen_items = self.URM_train.indptr[user_id]
            end_pos_seen_items = self.URM_train.indptr[user_id+1]
            n_seen_items = end_pos_seen_items - start_pos_seen_items

            negative_item_selected = False

            # It's faster to just try again then to build a mapping of the non-seen items for every user
            while not negative_item_selected:

                negative_item = np.random.randint(self.n_items)

                index = 0
                # Indices data is sorted, so I don't need to go to the end of the current row
                while index < n_seen_items and self.URM_train.indices[start_pos_seen_items + index] < negative_item:
                    index+=1

                # If the positive item in position 'index' is == sample.neg_item, negative not selected
                # If the positive item in position 'index' is > sample.neg_item or index == n_seen_items, negative selected
                if index == n_seen_items or self.URM_train.indices[start_pos_seen_items + index] > negative_item:
                    negative_item_selected = True

            self.batch_item[i_batch] = negative_item
            self.batch_rating[i_batch] = 0.0

        return self.batch_user[:i_batch+1], \
               self.batch_item[:i_batch+1], \
               self.batch_rating[:i_batch+1]


#
# class BPRIterator_set(object):
#     """
#     This Sampler performs BPR sampling *uniformly at random*:
#     - A user is sampled among the warm users (i.e., users who have at least an interaction in their user profile)
#     - An item the user interacted with
#     - An item the user did not interact with
#
#     The sample is: user_id, positive_item_id, negative_item_id
#     """
#
#     def __init__(self, URM_train, batch_size = 1):
#         super(BPRIterator_set, self).__init__()
#
#         self.URM_train = sps.csr_matrix(URM_train)
#         self.URM_train = self.URM_train.sorted_indices()
#         self.n_users, self.n_items = self.URM_train.shape
#         self.n_sampled_points = 0
#         self.batch_size = batch_size
#
#         self.user_to_profile_set = {user_id:set(self.URM_train[user_id].indices.tolist()) for user_id in range(self.n_users)}
#
#         self.warm_user_index_to_original_id = np.arange(0, self.n_users)[np.ediff1d(self.URM_train.indptr) > 0]
#         self.batch_user = torch.empty((self.batch_size,), dtype=torch.long)
#         self.batch_positive_item = torch.empty((self.batch_size,), dtype=torch.long)
#         self.batch_negative_item = torch.empty((self.batch_size,), dtype=torch.long)
#
#     def __len__(self):
#         return math.ceil(self.n_users/self.batch_size)
#
#     def __iter__(self):
#         self.n_sampled_points = 0
#         return self
#
#     def __next__(self):
#
#         if self.n_sampled_points >= self.n_users:
#             raise StopIteration
#
#         for i_batch in range(0, min(self.batch_size, self.n_users-self.n_sampled_points)):
#
#             self.n_sampled_points +=1
#             index = np.random.randint(self.n_users)
#             user_id = self.warm_user_index_to_original_id[index]
#
#             start_pos_seen_items = self.URM_train.indptr[user_id]
#             end_pos_seen_items = self.URM_train.indptr[user_id+1]
#             n_seen_items = end_pos_seen_items - start_pos_seen_items
#
#             index = np.random.randint(n_seen_items)
#             positive_item = self.URM_train.indices[start_pos_seen_items + index]
#
#             negative_item = np.random.randint(self.n_items)
#
#             while not negative_item in self.user_to_profile_set[user_id]:
#                 negative_item = np.random.randint(self.n_items)
#
#             self.batch_user[i_batch] = user_id
#             self.batch_positive_item[i_batch] = positive_item
#             self.batch_negative_item[i_batch] = negative_item
#
#
#         return self.batch_user[:i_batch+1], \
#                self.batch_positive_item[:i_batch+1], \
#                self.batch_negative_item[:i_batch+1]
#
#
#




class BPRIterator(object):
    """
    This Sampler performs BPR sampling *uniformly at random*:
    - A user is sampled among the warm users (i.e., users who have at least an interaction in their user profile)
    - An item the user interacted with
    - An item the user did not interact with

    The sample is: user_id, positive_item_id, negative_item_id
    """

    def __init__(self, URM_train, batch_size = 1, set_n_samples_to_draw = None, n_negatives_per_positive = 1):
        super(BPRIterator, self).__init__()

        self.URM_train = sps.csr_matrix(URM_train)
        self.URM_train = self.URM_train.sorted_indices()
        self.n_users, self.n_items = self.URM_train.shape
        self.n_negatives_per_positive = n_negatives_per_positive
        self.batch_size = batch_size

        self.warm_user_index_to_original_id = np.arange(0, self.n_users)[np.ediff1d(self.URM_train.indptr) > 0]
        self.n_samples_available = len(self.warm_user_index_to_original_id)
        self.n_samples_to_draw = self.n_samples_available if set_n_samples_to_draw is None else set_n_samples_to_draw
        self.n_samples_drawn = 0

        self.batch_user = torch.empty((self.batch_size,), dtype=torch.long)
        self.batch_positive_item = torch.empty((self.batch_size,), dtype=torch.long)

        if self.n_negatives_per_positive == 1:
            self.batch_negative_item = torch.empty((self.batch_size,), dtype=torch.long)
        else:
            self.batch_negative_item = torch.empty((self.batch_size,self.n_negatives_per_positive), dtype=torch.long)

    def __len__(self):
        return math.ceil(self.n_samples_to_draw/self.batch_size)

    def __iter__(self):
        self.n_samples_drawn = 0
        return self

    def __next__(self):

        if self.n_samples_drawn >= self.n_samples_to_draw:
            raise StopIteration

        for i_batch in range(0, min(self.batch_size, self.n_samples_to_draw-self.n_samples_drawn)):

            self.n_samples_drawn +=1
            index = np.random.randint(self.n_samples_available)
            user_id = self.warm_user_index_to_original_id[index]

            start_pos_seen_items = self.URM_train.indptr[user_id]
            end_pos_seen_items = self.URM_train.indptr[user_id+1]
            n_seen_items = end_pos_seen_items - start_pos_seen_items

            index = np.random.randint(n_seen_items)
            positive_item = self.URM_train.indices[start_pos_seen_items + index]

            self.batch_user[i_batch] = user_id
            self.batch_positive_item[i_batch] = positive_item

            for negative_item_batch_index in range(self.n_negatives_per_positive):

                negative_item_selected_flag = False

                # It's faster to just try again then to build a mapping of the non-seen items for every user
                while not negative_item_selected_flag:

                    negative_item = np.random.randint(self.n_items)

                    index = 0
                    # Indices data is sorted, so I don't need to go to the end of the current row
                    while index < n_seen_items and self.URM_train.indices[start_pos_seen_items + index] < negative_item:
                        index+=1

                    # If the positive item in position 'index' is == sample.neg_item, negative not selected
                    # If the positive item in position 'index' is > sample.neg_item or index == n_seen_items, negative selected
                    if index == n_seen_items or self.URM_train.indices[start_pos_seen_items + index] > negative_item:
                        negative_item_selected_flag = True


                if self.n_negatives_per_positive == 1:
                    self.batch_negative_item[i_batch] = negative_item
                else:
                    self.batch_negative_item[i_batch, negative_item_batch_index] = negative_item


        return self.batch_user[:i_batch+1], \
               self.batch_positive_item[:i_batch+1], \
               self.batch_negative_item[:i_batch+1]









class InteractionAndNegativeIterator(object):
    """
    This Sampler samples among all the existing user-item interactions *uniformly at random* and then adds a negative item:
    - One of the interactions in the dataset is sampled
    - Given the user associated to that interaction, it is also sampled an item the user did not interact with

    Note that this sampler is *NOT* BPR, this is because BPR samples the users at random and then, given the user,
    samples a positive and negative item. In this sampler the probability of selecting a user is proportional to the number
    of interaction in their user profile.

    The sample is: user_id, positive_item_id, negative_item_id

    """

    def __init__(self, URM_train, batch_size = 1, set_n_samples_to_draw = None, n_negatives_per_positive = 1):
        super(InteractionAndNegativeIterator, self).__init__()

        self.URM_train = sps.coo_matrix(URM_train)
        self.URM_train_row = torch.from_numpy(self.URM_train.row).long()
        self.URM_train_col = torch.from_numpy(self.URM_train.col).long()

        self.URM_train = sps.csr_matrix(URM_train)
        self.URM_train = self.URM_train.sorted_indices()
        self.n_users, self.n_items = self.URM_train.shape
        self.n_negatives_per_positive = n_negatives_per_positive
        self.batch_size = batch_size

        self.n_samples_available = URM_train.nnz
        self.n_samples_to_draw = self.n_samples_available if set_n_samples_to_draw is None else set_n_samples_to_draw
        self.n_samples_drawn = 0

        self.batch_user = torch.empty((self.batch_size,), dtype=torch.long)
        self.batch_positive_item = torch.empty((self.batch_size,), dtype=torch.long)

        if self.n_negatives_per_positive == 1:
            self.batch_negative_item = torch.empty((self.batch_size,), dtype=torch.long)
        else:
            self.batch_negative_item = torch.empty((self.batch_size,self.n_negatives_per_positive), dtype=torch.long)

    def __len__(self):
        return math.ceil(self.n_samples_to_draw/self.batch_size)

    def __iter__(self):
        self.n_samples_drawn = 0
        return self

    def __next__(self):

        if self.n_samples_drawn >= self.n_samples_to_draw:
            raise StopIteration

        this_batch_size = min(self.batch_size, self.n_samples_to_draw - self.n_samples_drawn)
        self.n_samples_drawn += this_batch_size

        index_batch = np.random.randint(self.n_samples_available, size = this_batch_size)

        self.batch_user[:this_batch_size] = self.URM_train_row[index_batch]
        self.batch_positive_item[:this_batch_size] = self.URM_train_col[index_batch]

        for i_batch in range(0, this_batch_size):

            user_id = self.batch_user[i_batch]

            start_pos_seen_items = self.URM_train.indptr[user_id]
            end_pos_seen_items = self.URM_train.indptr[user_id+1]
            n_seen_items = end_pos_seen_items - start_pos_seen_items

            for negative_item_batch_index in range(self.n_negatives_per_positive):

                negative_item_selected_flag = False

                # It's faster to just try again then to build a mapping of the non-seen items for every user
                while not negative_item_selected_flag:

                    negative_item = np.random.randint(self.n_items)

                    index = 0
                    # Indices data is sorted, so I don't need to go to the end of the current row
                    while index < n_seen_items and self.URM_train.indices[start_pos_seen_items + index] < negative_item:
                        index+=1

                    # If the positive item in position 'index' is == sample.neg_item, negative not selected
                    # If the positive item in position 'index' is > sample.neg_item or index == n_seen_items, negative selected
                    if index == n_seen_items or self.URM_train.indices[start_pos_seen_items + index] > negative_item:
                        negative_item_selected_flag = True

                if self.n_negatives_per_positive == 1:
                    self.batch_negative_item[i_batch] = negative_item
                else:
                    self.batch_negative_item[i_batch, negative_item_batch_index] = negative_item


        return self.batch_user[:i_batch+1], \
               self.batch_positive_item[:i_batch+1], \
               self.batch_negative_item[:i_batch+1]
