#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13/01/2018

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.Recommender_utils import check_matrix

from Data_manager.DataPostprocessing import DataPostprocessing
from Data_manager.DataReader_utils import remove_empty_rows_and_cols
import numpy as np


class DataPostprocessing_K_Cores(DataPostprocessing):
    """
    This class selects a dense partition of URM such that all items and users have at least K interactions.
    The algorithm is recursive and might not converge until the graph is empty.
    https://www.geeksforgeeks.org/find-k-cores-graph/
    """


    def __init__(self, dataReader_object, k_cores_value):

        assert k_cores_value >= 1,\
            "DataReaderPostprocessing_K_Cores: k_cores_value must be a positive value >= 1, provided value was {}".format(k_cores_value)

        super(DataPostprocessing_K_Cores, self).__init__(dataReader_object)

        self.k_cores_value = k_cores_value



    def _get_dataset_name_data_subfolder(self):
        """
        Returns the subfolder inside the dataset folder tree which contains the specific data to be loaded
        This method must be overridden by any data post processing object like k-cores / user sampling / interaction sampling etc
        to be applied before the data split

        :return: original or k_cores etc...
        """

        subfolder_name = "{}_cores/".format(self.k_cores_value)

        inner_subfolder_name = self.dataReader_object._get_dataset_name_data_subfolder()

        # Avoid concatenating the original/ part
        if inner_subfolder_name != self.DATASET_SUBFOLDER_ORIGINAL:
            subfolder_name += inner_subfolder_name

        return subfolder_name



    def _load_from_original_file(self):
        """
        _load_from_original_file will call the load of the dataset and then apply on it the k-cores
        :return:
        """

        loaded_dataset = self.dataReader_object.load_data()
        URM_all = loaded_dataset.AVAILABLE_URM["URM_all"]

         # Apply required K - core on zero-core data from ORIGINAL split
        _, users_to_remove, items_to_remove = select_k_cores(URM_all, k_value = self.k_cores_value)

        loaded_dataset._remove_items_and_users(items_to_remove = items_to_remove, users_to_remove = users_to_remove)

        return loaded_dataset


def select_k_cores(URM, k_value = 5, reshape = False):
    """

    :param URM:
    :param k_value:
    :param reshape:
    :return: URM, removedUsers, removedItems
    """

    print("DataDenseSplit_K_Cores: k-cores extraction will zero out some users and items without changing URM shape")

    URM.eliminate_zeros()
    n_users, n_items = URM.shape

    removed_users = set()
    removed_items = set()

    print("DataDenseSplit_K_Cores: Initial URM desity is {:.2E}".format(URM.nnz/(n_users*n_items)))

    convergence = False
    num_iterations = 0

    while not convergence:

        convergence_user = False

        URM = check_matrix(URM, 'csr')

        user_degree = np.ediff1d(URM.indptr)

        to_be_removed = user_degree < k_value
        to_be_removed[np.array(list(removed_users), dtype=np.int32)] = False

        if not np.any(to_be_removed):
            convergence_user = True

        else:

            for user in range(n_users):

                if to_be_removed[user] and user not in removed_users:
                    URM.data[URM.indptr[user]:URM.indptr[user+1]] = 0
                    removed_users.add(user)

            URM.eliminate_zeros()



        convergence_item = False

        URM = check_matrix(URM, 'csc')

        items_degree = np.ediff1d(URM.indptr)

        to_be_removed = items_degree < k_value
        to_be_removed[np.array(list(removed_items), dtype=np.int32)] = False

        if not np.any(to_be_removed):
            convergence_item = True

        else:

            for item in range(n_items):

                if to_be_removed[item] and item not in removed_items:
                    URM.data[URM.indptr[item]:URM.indptr[item+1]] = 0
                    removed_items.add(item)

            URM.eliminate_zeros()




        num_iterations += 1
        convergence = convergence_item and convergence_user


        if URM.nnz == 0:
            convergence = True
            print("DataDenseSplit_K_Cores: WARNING on iteration {}. URM is empty.".format(num_iterations))

        else:
            n_selected_users = n_users - len(removed_users)
            n_selected_items = n_items - len(removed_items)

            print("DataDenseSplit_K_Cores: K_Cores {}. Iteration {}. URM density without zeroed-out nodes is {:.2E}.\n"
                  "Selected users are {} ({:4.1f}%), Selected items are {} ({:4.1f}%)".format(
                k_value, num_iterations,
                URM.nnz/((n_selected_users)*(n_selected_items)),
                n_selected_users, n_selected_users/n_users*100,
                n_selected_items, n_selected_items/n_items*100))


    print("DataDenseSplit_K_Cores: split complete")

    URM.eliminate_zeros()

    if reshape:
        # Remove all columns and rows with no interactions
        return remove_empty_rows_and_cols(URM)


    return URM.copy(), np.array(list(removed_users)), np.array(list(removed_items))
