#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 24/12/2022

@author: Maurizio Ferrari Dacrema
"""



import scipy.sparse as sps
import numpy as np
import torch
from torch.utils.data import Dataset



class InteractionSampler(Dataset):
    """
    This Sampler samples among all the existing user-item interactions *uniformly at random*:
    - One of the interactions in the dataset is sampled

    The sample is: user_id, item_id, rating
    """

    def __init__(self, URM_train):
        super().__init__()
        URM_train = sps.coo_matrix(URM_train)

        self._row = torch.tensor(URM_train.row).type(torch.LongTensor)
        self._col = torch.tensor(URM_train.col).type(torch.LongTensor)
        self._data = torch.tensor(URM_train.data).type(torch.FloatTensor)


    def __len__(self):
        return len(self._row)

    def __getitem__(self, index):
        return self._row[index], self._col[index], self._data[index]


class BPRSampler(Dataset):
    """
    This Sampler performs BPR sampling *uniformly at random*:
    - A user is sampled among the warm users (i.e., users who have at least an interaction in their user profile)
    - An item the user interacted with
    - An item the user did not interact with

    The sample is: user_id, positive_item_id, negative_item_id
    """

    def __init__(self, URM_train):
        super().__init__()
        self._URM_train = sps.csr_matrix(URM_train)
        self.n_users, self.n_items = self._URM_train.shape

        self.warm_user_index_to_original_id = np.arange(0, self.n_users, dtype=np.int64)[np.ediff1d(sps.csr_matrix(self._URM_train).indptr) > 0]

    def __len__(self):
        return len(self.warm_user_index_to_original_id)

    def __getitem__(self, user_index):

        user_id = self.warm_user_index_to_original_id[user_index]

        seen_items = self._URM_train.indices[self._URM_train.indptr[user_id]:self._URM_train.indptr[user_id+1]]
        item_positive = np.random.choice(seen_items)

        # seen_items = set(list(seen_items))
        negative_selected = False

        while not negative_selected:
            negative_candidate = np.random.randint(low=0, high=self.n_items, size=1)[0]

            if negative_candidate not in seen_items:
                item_negative = negative_candidate
                negative_selected = True

        return user_id, item_positive.astype(np.int64), item_negative.astype(np.int64)


class InteractionAndNegativeSampler(Dataset):
    """
    This Sampler samples among all the existing user-item interactions *uniformly at random* and then adds a negative item:
    - One of the interactions in the dataset is sampled
    - Given the user associated to that interaction, it is also sampled an item the user did not interact with

    Note that this sampler is *NOT* BPR, this is because BPR samples the users at random and then, given the user,
    samples a positive and negative item. In this sampler the probability of selecting a user is proportional to the number
    of interaction in their user profile.

    The sample is: user_id, positive_item_id, negative_item_id

    """

    def __init__(self, URM_train):
        super().__init__()
        self._URM_train = sps.csr_matrix(URM_train)
        self.n_users, self.n_items = self._URM_train.shape

        URM_train_coo = sps.coo_matrix(URM_train)
        self._row = torch.tensor(URM_train_coo.row).type(torch.LongTensor)
        self._col = torch.tensor(URM_train_coo.col).type(torch.LongTensor)

    def __len__(self):
        return self._URM_train.nnz

    def __getitem__(self, index):

        user_id = self._row[index]
        item_positive = self._col[index]

        seen_items = self._URM_train.indices[self._URM_train.indptr[user_id]:self._URM_train.indptr[user_id+1]]

        negative_selected = False

        while not negative_selected:
            negative_candidate = np.random.randint(low=0, high=self.n_items, size=1)[0]

            if negative_candidate not in seen_items:
                item_negative = negative_candidate
                negative_selected = True

        return user_id, item_positive, item_negative.astype(np.int64)


