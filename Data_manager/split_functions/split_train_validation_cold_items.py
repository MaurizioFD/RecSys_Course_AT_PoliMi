#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/04/2019

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import scipy.sparse as sps
from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix




def _select_train_warm_items(URM_all, train_item_percentage, train_interaction_percentage = None):
    """
    Selects a certain percentage of the URM_all WARM items and splits the URM in two
    IMPORTANT: the number of items to be sampled is not computed with respect to the shape of the URM but with respect
    to the number of WARM items it contains. Cold items don't count.
    :param URM_all:
    :param train_item_percentage:
    :param train_interaction_percentage:
    :return:
    """

    sample_successful = False
    terminate = False

    n_interactions = URM_all.nnz

    URM = sps.csc_matrix(URM_all)
    item_interactions = np.ediff1d(URM.indptr)

    n_warm_items = np.sum(item_interactions>0)

    n_train_items = int(n_warm_items * train_item_percentage)

    indices_for_sampling = np.arange(0, URM_all.shape[1], dtype=np.int)[item_interactions>0]
    np.random.shuffle(indices_for_sampling)


    while not terminate:

        if n_train_items == n_warm_items and n_train_items > 1:
            n_train_items -= 1

        train_items = indices_for_sampling[0:n_train_items]

        # check if enough interactions are in train
        if train_interaction_percentage is not None:

            train_interactions = np.sum(item_interactions[train_items])

            current_train_interaction_percentage = train_interactions/n_interactions

            if current_train_interaction_percentage < train_interaction_percentage*0.9:
                # Too few interactions in train, add items
                if n_train_items == n_warm_items:
                    terminate = True
                    sample_successful = False
                else:
                    n_train_items += 1

            elif current_train_interaction_percentage > train_interaction_percentage*1.1:
                # Too many interactions in train, remove items
                if n_train_items == 1:
                    terminate = True
                    sample_successful = False
                else:
                    n_train_items -= 1

            else:
                terminate = True
                sample_successful = True

        else:
            terminate = True
            sample_successful = True


    assert sample_successful, "Unable to select the train items with the desired specifications"



    return train_items



def _zero_out_values(sparse_matrix, columns_to_zero = None, rows_to_zero = None):

    if rows_to_zero is not None:
        sparse_matrix = sps.csr_matrix(sparse_matrix)

        for n_row in rows_to_zero:
            start_pos = sparse_matrix.indptr[n_row]
            end_pos = sparse_matrix.indptr[n_row+1]

            sparse_matrix.data[start_pos:end_pos] = np.zeros_like(sparse_matrix.data[start_pos:end_pos])

        sparse_matrix.eliminate_zeros()

    if columns_to_zero is not None:
        sparse_matrix = _zero_out_values(sparse_matrix.T, rows_to_zero = columns_to_zero).T

    sparse_matrix = sps.csr_matrix(sparse_matrix)

    return sparse_matrix



def split_train_in_two_cold_items(URM_all, ICM_list = None, train_item_percentage = 0.1, train_interaction_percentage = None):
    """
    The function splits an URM in two matrices selecting the number of interactions one user at a time
    :param URM_train:
    :param train_percentage:
    :param verbose:
    :return:
    """

    assert train_item_percentage >= 0.0 and train_item_percentage<=1.0, "train_item_percentage must be a value between 0.0 and 1.0, provided was '{}'".format(train_item_percentage)

    # Use CSC for item-wise split
    URM_all = sps.csc_matrix(URM_all)

    n_users, n_items = URM_all.shape

    train_items = _select_train_warm_items(URM_all, train_item_percentage, train_interaction_percentage = train_interaction_percentage)

    validation_items_mask = np.ones(n_items, dtype=np.bool)
    validation_items_mask[train_items] = False
    validation_items = np.arange(0, n_items, dtype = np.int)[validation_items_mask]

    URM_train = _zero_out_values(URM_all.copy(), columns_to_zero = validation_items)
    URM_validation = _zero_out_values(URM_all.copy(), columns_to_zero = train_items)

    if ICM_list is not None:

        ICM_train_list = []
        ICM_valiation_list = []

        for ICM_object in ICM_list:
            ICM_object_train = _zero_out_values(ICM_object.copy(), rows_to_zero = validation_items)
            ICM_object_validation = _zero_out_values(ICM_object.copy(), rows_to_zero = train_items)

            ICM_train_list.append(ICM_object_train)
            ICM_valiation_list.append(ICM_object_validation)

        return URM_train, URM_validation, ICM_train_list, ICM_valiation_list, train_items

    else:
        return URM_train, URM_validation, train_items


