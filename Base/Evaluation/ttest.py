#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/07/18

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
from scipy import stats

np.random.seed(12345678)

rvs1 = stats.norm.rvs(loc=5,scale=10,size=500)
rvs2 = stats.norm.rvs(loc=5,scale=10,size=500)

from Base.Evaluation.k_fold_significance_test import compute_k_fold_significance


compute_k_fold_significance(rvs1, rvs2)



#
# t_statistic, p_value = stats.ttest_ind(rvs1,rvs2)
#
# print("t_statistic: {:.4f}, p_value: {:.4f}".format(t_statistic, p_value))
#
# t_statistic, p_value = stats.ttest_ind(rvs1,rvs2, equal_var = False)
#
# print("t_statistic: {:.4f}, p_value: {:.4f}".format(t_statistic, p_value))