#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13/01/2018

@author: Maurizio Ferrari Dacrema
"""

from Base.Recommender_utils import check_matrix

from data.DataReader import removeZeroRatingRowAndCol
import numpy as np


# 
# class URM_Dense_K_Cores(object):
#     """
#     This class selects a dense partition of URM such that all items and users have at least K interactions.
#     The algorithm is recursive and might not converge until the graph is empty.
#     https://www.geeksforgeeks.org/find-k-cores-graph/
#     """
# 
#     def __init__( URM):
# 
#         super(URM_Dense_K_Cores, .__init__()
# 
#         URM = URM.copy()
# 
# 

def select_k_cores(URM, k_value = 5, reshape = False):
    """

    :param URM:
    :param k_value:
    :param reshape:
    :return: URM, removedUsers, removedItems
    """

    print("DataDenseSplit_K_Cores: k-cores extraction will zero out some users and items without changing URM shape")

    URM.eliminate_zeros()

    n_users = URM.shape[0]
    n_items = URM.shape[1]

    removed_users = set()
    removed_items = set()

    print("DataDenseSplit_K_Cores: Initial URM desity is {:.2E}".format(URM.nnz/(n_users*n_items)))

    convergence = False
    numIterations = 0

    while not convergence:

        convergence_user = False

        URM = check_matrix(URM, 'csr')

        user_degree = np.ediff1d(URM.indptr)

        to_be_removed = user_degree < k_value
        to_be_removed[np.array(list(removed_users), dtype=np.int)] = False

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
        to_be_removed[np.array(list(removed_items), dtype=np.int)] = False

        if not np.any(to_be_removed):
            convergence_item = True

        else:

            for item in range(n_items):

                if to_be_removed[item] and item not in removed_items:
                    URM.data[URM.indptr[item]:URM.indptr[item+1]] = 0
                    removed_items.add(item)

            URM.eliminate_zeros()




        numIterations += 1
        convergence = convergence_item and convergence_user


        if URM.data.sum() == 0:
            convergence = True
            print("DataDenseSplit_K_Cores: WARNING on iteration {}. URM is empty.".format(numIterations))

        else:
             print("DataDenseSplit_K_Cores: Iteration {}. URM desity without zeroed-out nodes is {:.2E}.\n"
                  "Users with less than {} interactions are {} ( {:.2f}%), Items are {} ( {:.2f}%)".format(
                numIterations,
                sum(URM.data)/((n_users-len(removed_users))*(n_items-len(removed_items))),
                k_value,
                len(removed_users), len(removed_users)/n_users*100,
                len(removed_items), len(removed_items)/n_items*100))


    print("DataDenseSplit_K_Cores: split complete")

    URM.eliminate_zeros()

    if reshape:
        # Remove all columns and rows with no interactions
        return removeZeroRatingRowAndCol(URM)


    return URM.copy(), np.array(list(removed_users)), np.array(list(removed_items))
