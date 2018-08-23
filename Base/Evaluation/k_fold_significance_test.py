#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11/07/18

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
from scipy import stats

def compute_k_fold_significance(list_1, *other_lists):

    print("List 1: {:.4f} ± {:.4f}".format(np.mean(list_1), np.std(list_1)))


    for other_list_index in range(len(other_lists)):

        other_list = other_lists[other_list_index]

        assert isinstance(other_list, list) or isinstance(other_list, np.ndarray), "The provided lists must be either Python lists or numpy.ndarray"
        assert len(list_1) == len(other_list), "The provided lists have different length, list 1: {}, list 2: {}".format(len(list_1), len(other_list))

        print("List {}: {:.4f} ± {:.4f}".format(other_list_index+2, np.mean(other_list), np.std(other_list)))

        t_statistic, p_value = stats.ttest_ind(list_1, other_list)

        print("List {} t_statistic: {:.4f}, p_value: {:.4f}".format(other_list_index+2, t_statistic, p_value))
