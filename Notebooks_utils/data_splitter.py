#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21/10/2018

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import scipy.sparse as sps



def train_test_holdout(URM_all, train_perc = 0.8):


    numInteractions = URM_all.nnz
    URM_all = URM_all.tocoo()
    shape = URM_all.shape


    train_mask = np.random.choice([True,False], numInteractions, p=[train_perc, 1-train_perc])


    URM_train = sps.coo_matrix((URM_all.data[train_mask], (URM_all.row[train_mask], URM_all.col[train_mask])), shape=shape)
    URM_train = URM_train.tocsr()

    test_mask = np.logical_not(train_mask)

    URM_test = sps.coo_matrix((URM_all.data[test_mask], (URM_all.row[test_mask], URM_all.col[test_mask])), shape=shape)
    URM_test = URM_test.tocsr()

    return URM_train, URM_test