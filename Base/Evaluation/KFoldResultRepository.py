#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11/07/18

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
from scipy import stats
#
# import random
# from statsmodels.sandbox.stats.multicomp import multipletests
#
# # as example, all null hypotheses are true
# pvals = [random.random() for _ in range(10)]
# is_reject, corrected_pvals, _, _ = multipletests(pvals, alpha=0.1, method='fdr_bh')


def compute_k_fold_significance(list_1, alpha, *other_lists):
    """
    Type 1 Errors: we identify as significant somenthing which is not, due to random chance. Lower alpha values reduce this error rate.
    Bonferroni correction is VERY conservative and also reduces the true positives rate.
    http://www.nonlinear.com/support/progenesis/comet/faq/v2.0/pq-values.aspx


    https://multithreaded.stitchfix.com/blog/2015/10/15/multiple-hypothesis-testing/
    https://www.scipy-lectures.org/packages/statistics/index.html

    :param list_1:
    :param alpha:
    :param other_lists:
    :return:
    """

    print("List 1: {:.4f} ± {:.4f}".format(np.mean(list_1), np.std(list_1)))

    if len(other_lists) > 1:
        original_alpha = alpha
        alpha = alpha/len(other_lists)
        print("Applying Bonferroni correction for {} lists, original alpha is {}, corrected alpha is {}".format(len(other_lists), original_alpha, alpha))


    for other_list_index in range(len(other_lists)):

        other_list = other_lists[other_list_index]

        assert isinstance(other_list, list) or isinstance(other_list, np.ndarray), "The provided lists must be either Python lists or numpy.ndarray"
        assert len(list_1) == len(other_list), "The provided lists have different length, list 1: {}, list 2: {}".format(len(list_1), len(other_list))

        print("List {}: {:.4f} ± {:.4f}".format(other_list_index+2, np.mean(other_list), np.std(other_list)))

        # Test difference between populations
        t_statistic, p_value = stats.ttest_ind(list_1, other_list)
        #t_statistic, p_value = stats.mannwhitneyu(list_1, other_list)

        if p_value < alpha:
            significance = "IS significant."
        else:
            significance = "Is NOT significant."

        print("List {} t_statistic: {:.4f}, p_value: {:.4f}, alpha: {:.4f}. {}".format(other_list_index+2, t_statistic, p_value, alpha, significance))

        # Test difference between two observations of the same "individual" or data with a paired test
        # Equivalent to test whether (list_1 - other_list) has an average of 0
        t_statistic, p_value = stats.ttest_rel(list_1, other_list)
        #t_statistic, p_value = stats.wilcoxon(list_1, other_list)

        if p_value < alpha:
            significance = "IS significant."
        else:
            significance = "Is NOT significant."

        print("List {} paired t_statistic: {:.4f}, p_value: {:.4f}, alpha: {:.4f}. {}\n".format(other_list_index+2, t_statistic, p_value, alpha, significance))




class KFoldResultRepository(object):
    """KFoldResultRepository"""

    def __init__(self, n_folds, allow_overwrite = False):
        super(KFoldResultRepository, self).__init__()

        assert n_folds>0, "KFoldResultRepository: n_folds cannot be negative"

        self._result_list = [None]*n_folds
        self._n_folds = n_folds
        self._allow_overwrite = allow_overwrite


    def set_results_in_fold(self, fold_index, result_dict):

        if self._result_list[fold_index] is not None and not self._allow_overwrite:
            raise Exception("KFoldResultRepository: set_results_in_fold {} would overite previously set value".format(fold_index))

        self._result_list[fold_index] = result_dict.copy()


    def get_results(self):
        return self._result_list.copy()

    def get_fold_number(self):
        return self._n_folds


    def run_significance_test(self, other_result_repository, metric = None, alpha = 0.005):

        assert isinstance(other_result_repository, KFoldResultRepository), "KFoldResultRepository: run_significance_test must receive another repository as parameter"
        assert other_result_repository.get_fold_number()== self.get_fold_number(), "KFoldResultRepository: run_significance_test other repository must have the same number of folds"

        result_list_other = other_result_repository.get_results()

        if metric is None:
            metric_list = list(result_list_other[0].keys())
        else:
            metric_list = [metric]


        for metric in metric_list:

            print("Significance test on metric: {}".format(metric))

            list_this = []
            list_other = []

            for fold_index in range(self._n_folds):

                list_this.append(self._result_list[fold_index][metric])
                list_other.append(result_list_other[fold_index][metric])

            compute_k_fold_significance(list_this, alpha, list_other)