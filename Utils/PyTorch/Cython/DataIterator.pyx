#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 01/01/2023

@author: Maurizio Ferrari Dacrema
"""

#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: unpack_method_calls=True
#cython: overflowcheck=False

#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import cython
import torch
import math

import numpy as np
cimport numpy as np

import scipy.sparse as sps
from libc.stdlib cimport rand, srand, RAND_MAX

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.initializedcheck(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
# @cython.overflowcheck(False)
# cdef class InteractionIterator:
#     """
#     This Sampler samples among all the existing user-item interactions *uniformly at random*:
#     - One of the interactions in the dataset is sampled
#
#     The sample is: user_id, item_id, rating
#     """
#
#     cdef int n_users, n_items, n_data_points, batch_size, n_samples_drawn
#
#     cdef int[:] URM_train_row, URM_train_col
#     cdef double[:] URM_train_data
#
#     cdef int[:] batch_user, batch_col
#     cdef double[:] batch_rating
#
#     def __init__(self, URM_train,
#                        batch_size = 1):
#         super().__init__()
#
#         URM_train = sps.coo_matrix(URM_train)
#
#         self.n_users, self.n_items = URM_train.shape
#         self.n_data_points = URM_train.nnz
#         self.n_samples_drawn = 0
#
#         self.URM_train_row = np.array(URM_train.row, dtype=np.int32)
#         self.URM_train_col = np.array(URM_train.col, dtype=np.int32)
#         self.URM_train_data = np.array(URM_train.data, dtype=np.float64)
#
#         self.batch_size = batch_size
#         self.batch_user = np.zeros(self.batch_size, dtype=np.int32)
#         self.batch_col = np.zeros(self.batch_size, dtype=np.int32)
#         self.batch_rating = np.zeros(self.batch_size, dtype=np.float64)
#
#     def __len__(self):
#         return math.ceil(self.n_data_points/self.batch_size)
#
#     def __iter__(self):
#         self.n_samples_drawn = 0
#         return self
#
#     def __next__(self):
#
#         cdef int i_batch, i_data
#
#         if self.n_samples_drawn >= self.n_data_points:
#             raise StopIteration
#
#         # for i_batch in range(0, self.batch_size):#min(self.batch_size, self.n_data_points-self.n_samples_drawn)):
#         for i_batch in range(0, min(self.batch_size, self.n_data_points-self.n_samples_drawn)):
#
#             self.n_samples_drawn +=1
#             i_data = int(rand()*1.0/RAND_MAX*(self.n_data_points-1))
#
#             self.batch_user[i_batch] = self.URM_train_row[i_data]
#             self.batch_col[i_batch] = self.URM_train_col[i_data]
#             self.batch_rating[i_batch] = self.URM_train_data[i_data]
#
#         return torch.from_numpy(np.array(self.batch_user[:i_batch+1], dtype=np.int64)),\
#                torch.from_numpy(np.array(self.batch_col[:i_batch+1], dtype=np.int64)), \
#                torch.from_numpy(np.array(self.batch_rating[:i_batch+1], dtype=np.float64))
#

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
cdef class InteractionIterator:
    """
    This Sampler samples among all the existing user-item interactions *uniformly at random*:
    - One of the interactions in the dataset is sampled

    The sample is: user_id, item_id, rating
    """

    cdef int n_users, n_items, batch_size, n_samples_available, n_samples_drawn, n_samples_to_draw
    cdef double positive_quota

    cdef int[:] URM_train_row, URM_train_col, URM_train_indices, URM_train_indptr
    cdef double[:] URM_train_data

    cdef int[:] batch_user, batch_item
    cdef double[:] batch_rating

    def __init__(self, URM_train, positive_quota, batch_size = 1, set_n_samples_to_draw = None):
        super().__init__()

        URM_train = sps.coo_matrix(URM_train)
        self.n_users, self.n_items = URM_train.shape
        self.positive_quota = positive_quota

        self.n_samples_available = URM_train.nnz
        self.n_samples_to_draw = self.n_samples_available if set_n_samples_to_draw is None else set_n_samples_to_draw
        self.n_samples_drawn = 0

        self.URM_train_row = np.array(URM_train.row, dtype=np.int32)
        self.URM_train_col = np.array(URM_train.col, dtype=np.int32)
        self.URM_train_data = np.array(URM_train.data, dtype=np.float64)

        URM_train = sps.csr_matrix(URM_train)
        URM_train = URM_train.sorted_indices()
        self.URM_train_indices = np.array(URM_train.indices, dtype=np.int32)
        self.URM_train_indptr = np.array(URM_train.indptr, dtype=np.int32)

        self.batch_size = batch_size
        self.batch_user = np.zeros(self.batch_size, dtype=np.int32)
        self.batch_item = np.zeros(self.batch_size, dtype=np.int32)
        self.batch_rating = np.zeros(self.batch_size, dtype=np.float64)

    def __len__(self):
        return math.ceil(self.n_samples_to_draw/self.batch_size)

    def __iter__(self):
        self.n_samples_drawn = 0
        return self

    def __next__(self):

        cdef int i_batch, i_data
        cdef int user_id, index, n_seen_items, positive_item, negative_item
        cdef int start_pos_seen_items, end_pos_seen_items, negative_item_selected

        if self.n_samples_drawn >= self.n_samples_to_draw:
            raise StopIteration

        for i_batch in range(0, min(self.batch_size, self.n_samples_to_draw-self.n_samples_drawn)):

            self.n_samples_drawn +=1
            i_data = int(rand()*1.0/RAND_MAX*(self.n_samples_available-1))

            self.batch_user[i_batch] = self.URM_train_row[i_data]

            if rand()*1.0/RAND_MAX <= self.positive_quota:
                self.batch_item[i_batch] = self.URM_train_col[i_data]
                self.batch_rating[i_batch] = self.URM_train_data[i_data]
            else:

                start_pos_seen_items = self.URM_train_indptr[self.batch_user[i_batch]]
                end_pos_seen_items = self.URM_train_indptr[self.batch_user[i_batch]+1]
                n_seen_items = end_pos_seen_items - start_pos_seen_items

                negative_item_selected = False

                # It's faster to just try again then to build a mapping of the non-seen items for every user
                while not negative_item_selected:

                    negative_item = int(rand()*1.0/RAND_MAX*(self.n_items-1))

                    index = 0
                    # Indices data is sorted, so I don't need to go to the end of the current row
                    while index < n_seen_items and self.URM_train_indices[start_pos_seen_items + index] < negative_item:
                        index+=1

                    # If the positive item in position 'index' is == sample.neg_item, negative not selected
                    # If the positive item in position 'index' is > sample.neg_item or index == n_seen_items, negative selected
                    if index == n_seen_items or self.URM_train_indices[start_pos_seen_items + index] > negative_item:
                        negative_item_selected = True

                self.batch_item[i_batch] = negative_item
                self.batch_rating[i_batch] = 0.0

        return torch.from_numpy(np.array(self.batch_user[:i_batch+1], dtype=np.int64)),\
               torch.from_numpy(np.array(self.batch_item[:i_batch+1], dtype=np.int64)), \
               torch.from_numpy(np.array(self.batch_rating[:i_batch+1], dtype=np.float64))





@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
cdef class BPRIterator:
    """
    This Sampler performs BPR sampling *uniformly at random*:
    - A user is sampled among the warm users (i.e., users who have at least an interaction in their user profile)
    - An item the user interacted with
    - An item the user did not interact with

    The sample is: user_id, positive_item_id, negative_item_id
    """

    cdef int n_users, n_items, batch_size, n_samples_to_draw, n_samples_available, n_samples_drawn, n_negatives_per_positive
    cdef double rescaling_factor_rnd
    cdef int[:] URM_train_indices, URM_train_indptr, warm_user_index_to_original_id

    cdef int[:] batch_user, batch_positive_item, batch_negative_item
    cdef int[:,:] batch_multiple_negative_item

    def __init__(self, URM_train, batch_size = 1, set_n_samples_to_draw = None, n_negatives_per_positive = 1):
        super().__init__()

        self.n_users, self.n_items = URM_train.shape
        self.n_negatives_per_positive = n_negatives_per_positive

        self.warm_user_index_to_original_id = np.arange(0, self.n_users, dtype=np.int32)[np.ediff1d(URM_train.indptr) > 0]
        self.n_samples_available = len(self.warm_user_index_to_original_id)
        self.n_samples_to_draw = self.n_samples_available if set_n_samples_to_draw is None else set_n_samples_to_draw
        self.n_samples_drawn = 0

        URM_train = sps.csr_matrix(URM_train)
        URM_train = URM_train.sorted_indices()
        self.URM_train_indices = np.array(URM_train.indices, dtype=np.int32)
        self.URM_train_indptr = np.array(URM_train.indptr, dtype=np.int32)

        self.batch_size = batch_size
        self.batch_user = np.zeros(self.batch_size, dtype=np.int32)
        self.batch_positive_item = np.zeros(self.batch_size, dtype=np.int32)

        if self.n_negatives_per_positive == 1:
            self.batch_negative_item = np.zeros(self.batch_size, dtype=np.int32)
        else:
            self.batch_multiple_negative_item = np.zeros((self.batch_size, self.n_negatives_per_positive), dtype=np.int32)

    def __len__(self):
        return math.ceil(self.n_samples_to_draw/self.batch_size)

    def __iter__(self):
        self.n_samples_drawn = 0
        return self

    def __next__(self):

        cdef int i_batch, user_id, index, n_seen_items, positive_item, negative_item
        cdef int start_pos_seen_items, end_pos_seen_items, negative_item_selected_flag, negative_item_batch_index

        if self.n_samples_drawn >= self.n_samples_to_draw:
            raise StopIteration

        for i_batch in range(0, min(self.batch_size, self.n_samples_to_draw-self.n_samples_drawn)):

            self.n_samples_drawn +=1
            index = int(rand()*1.0/RAND_MAX*(self.n_samples_available-1))
            user_id = self.warm_user_index_to_original_id[index]

            start_pos_seen_items = self.URM_train_indptr[user_id]
            end_pos_seen_items = self.URM_train_indptr[user_id+1]
            n_seen_items = end_pos_seen_items - start_pos_seen_items

            index = int(rand()*1.0/RAND_MAX*(n_seen_items-1))
            positive_item = self.URM_train_indices[start_pos_seen_items + index]

            self.batch_user[i_batch] = user_id
            self.batch_positive_item[i_batch] = positive_item

            for negative_item_batch_index in range(self.n_negatives_per_positive):

                negative_item_selected_flag = False

                # It's faster to just try again then to build a mapping of the non-seen items for every user
                while not negative_item_selected_flag:

                    negative_item = int(rand()*1.0/RAND_MAX*(self.n_items-1))

                    index = 0
                    # Indices data is sorted, so I don't need to go to the end of the current row
                    while index < n_seen_items and self.URM_train_indices[start_pos_seen_items + index] < negative_item:
                        index+=1

                    # If the positive item in position 'index' is == sample.neg_item, negative not selected
                    # If the positive item in position 'index' is > sample.neg_item or index == n_seen_items, negative selected
                    if index == n_seen_items or self.URM_train_indices[start_pos_seen_items + index] > negative_item:
                        negative_item_selected_flag = True

                if self.n_negatives_per_positive == 1:
                    self.batch_negative_item[i_batch] = negative_item
                else:
                    self.batch_multiple_negative_item[i_batch, negative_item_batch_index] = negative_item


        if self.n_negatives_per_positive == 1:
            return torch.from_numpy(np.array(self.batch_user[:i_batch+1], dtype=np.int64)),\
                   torch.from_numpy(np.array(self.batch_positive_item[:i_batch+1], dtype=np.int64)), \
                   torch.from_numpy(np.array(self.batch_negative_item[:i_batch+1], dtype=np.int64))
        else:
            return torch.from_numpy(np.array(self.batch_user[:i_batch+1], dtype=np.int64)),\
                   torch.from_numpy(np.array(self.batch_positive_item[:i_batch+1], dtype=np.int64)), \
                   torch.from_numpy(np.array(self.batch_multiple_negative_item[:i_batch+1,:], dtype=np.int64))







@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
cdef class InteractionAndNegativeIterator:
    """
    This Sampler samples among all the existing user-item interactions *uniformly at random* and then adds a negative item:
    - One of the interactions in the dataset is sampled
    - Given the user associated to that interaction, it is also sampled an item the user did not interact with

    Note that this sampler is *NOT* BPR, this is because BPR samples the users at random and then, given the user,
    samples a positive and negative item. In this sampler the probability of selecting a user is proportional to the number
    of interaction in their user profile.

    The sample is: user_id, positive_item_id, negative_item_id

    """

    cdef int n_users, n_items, batch_size, n_samples_available, n_samples_to_draw, n_samples_drawn, n_negatives_per_positive

    cdef int[:] URM_train_row, URM_train_col
    cdef int[:] URM_train_indices, URM_train_indptr

    cdef double[:] URM_train_data

    cdef int[:] batch_user, batch_positive_item, batch_negative_item
    cdef int[:,:] batch_multiple_negative_item

    def __init__(self, URM_train, batch_size = 1, set_n_samples_to_draw = None, n_negatives_per_positive = 1):
        super().__init__()

        self.n_users, self.n_items = URM_train.shape
        self.n_negatives_per_positive = n_negatives_per_positive

        self.n_samples_available = URM_train.nnz
        self.n_samples_to_draw = self.n_samples_available if set_n_samples_to_draw is None else set_n_samples_to_draw
        self.n_samples_drawn = 0

        URM_train = sps.csr_matrix(URM_train)
        URM_train = URM_train.sorted_indices()
        self.URM_train_indices = np.array(URM_train.indices, dtype=np.int32)
        self.URM_train_indptr = np.array(URM_train.indptr, dtype=np.int32)

        URM_train = sps.coo_matrix(URM_train)
        self.URM_train_row = np.array(URM_train.row, dtype=np.int32)
        self.URM_train_col = np.array(URM_train.col, dtype=np.int32)

        self.batch_size = batch_size
        self.batch_user = np.zeros(self.batch_size, dtype=np.int32)
        self.batch_positive_item = np.zeros(self.batch_size, dtype=np.int32)

        if self.n_negatives_per_positive == 1:
            self.batch_negative_item = np.zeros(self.batch_size, dtype=np.int32)
        else:
            self.batch_multiple_negative_item = np.zeros((self.batch_size, self.n_negatives_per_positive), dtype=np.int32)

    def __len__(self):
        return math.ceil(self.n_samples_to_draw/self.batch_size)

    def __iter__(self):
        self.n_samples_drawn = 0
        return self

    def __next__(self):

        cdef int i_batch, user_id, index, n_seen_items, positive_item, negative_item
        cdef int start_pos_seen_items, end_pos_seen_items, negative_item_selected_flag, negative_item_batch_index

        if self.n_samples_drawn >= self.n_samples_to_draw:
            raise StopIteration

        for i_batch in range(0, min(self.batch_size, self.n_samples_to_draw-self.n_samples_drawn)):

            self.n_samples_drawn +=1
            index = int(rand()*1.0/RAND_MAX*(self.n_samples_available-1))

            user_id = self.URM_train_row[index]
            positive_item = self.URM_train_col[index]

            start_pos_seen_items = self.URM_train_indptr[user_id]
            end_pos_seen_items = self.URM_train_indptr[user_id+1]
            n_seen_items = end_pos_seen_items - start_pos_seen_items

            self.batch_user[i_batch] = user_id
            self.batch_positive_item[i_batch] = positive_item

            for negative_item_batch_index in range(self.n_negatives_per_positive):

                negative_item_selected_flag = False

                # It's faster to just try again then to build a mapping of the non-seen items for every user
                while not negative_item_selected_flag:

                    negative_item = int(rand()*1.0/RAND_MAX*(self.n_items-1))

                    index = 0
                    # Indices data is sorted, so I don't need to go to the end of the current row
                    while index < n_seen_items and self.URM_train_indices[start_pos_seen_items + index] < negative_item:
                        index+=1

                    # If the positive item in position 'index' is == sample.neg_item, negative not selected
                    # If the positive item in position 'index' is > sample.neg_item or index == n_seen_items, negative selected
                    if index == n_seen_items or self.URM_train_indices[start_pos_seen_items + index] > negative_item:
                        negative_item_selected_flag = True


                if self.n_negatives_per_positive == 1:
                    self.batch_negative_item[i_batch] = negative_item
                else:
                    self.batch_multiple_negative_item[i_batch, negative_item_batch_index] = negative_item


        if self.n_negatives_per_positive == 1:
            return torch.from_numpy(np.array(self.batch_user[:i_batch+1], dtype=np.int64)),\
                   torch.from_numpy(np.array(self.batch_positive_item[:i_batch+1], dtype=np.int64)), \
                   torch.from_numpy(np.array(self.batch_negative_item[:i_batch+1], dtype=np.int64))
        else:
            return torch.from_numpy(np.array(self.batch_user[:i_batch+1], dtype=np.int64)),\
                   torch.from_numpy(np.array(self.batch_positive_item[:i_batch+1], dtype=np.int64)), \
                   torch.from_numpy(np.array(self.batch_multiple_negative_item[:i_batch+1,:], dtype=np.int64))

