#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26/06/18

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import scipy.sparse as sps
import time, sys, copy
import pandas as pd

from enum import Enum
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit

from Evaluation.metrics import precision, precision_recall_min_denominator, recall, MAP, MAP_MIN_DEN, MRR, HIT_RATE, ndcg, arhr_all_hits, \
    Novelty, Coverage_Item, Coverage_Item_HIT, Items_In_GT, _Metrics_Object, Coverage_User, Coverage_User_HIT, Users_In_GT, Gini_Diversity, Shannon_Entropy, Diversity_MeanInterList,\
    Diversity_Herfindahl, AveragePopularity, Ratio_Diversity_Gini, Ratio_Diversity_Herfindahl, Ratio_Shannon_Entropy, Ratio_AveragePopularity, Ratio_Novelty


class EvaluatorMetrics(Enum):

    PRECISION = "PRECISION"
    PRECISION_RECALL_MIN_DEN = "PRECISION_RECALL_MIN_DEN"
    RECALL = "RECALL"
    MAP = "MAP"
    MAP_MIN_DEN = "MAP_MIN_DEN"
    MRR = "MRR"
    NDCG = "NDCG"
    F1 = "F1"
    HIT_RATE = "HIT_RATE"
    ARHR = "ARHR_ALL_HITS"
    NOVELTY = "NOVELTY"
    AVERAGE_POPULARITY = "AVERAGE_POPULARITY"
    DIVERSITY_SIMILARITY = "DIVERSITY_SIMILARITY"
    DIVERSITY_MEAN_INTER_LIST = "DIVERSITY_MEAN_INTER_LIST"
    DIVERSITY_HERFINDAHL = "DIVERSITY_HERFINDAHL"
    COVERAGE_ITEM = "COVERAGE_ITEM"
    COVERAGE_ITEM_HIT = "COVERAGE_ITEM_HIT"
    ITEMS_IN_GT = "ITEMS_IN_GT"
    COVERAGE_USER = "COVERAGE_USER"
    COVERAGE_USER_HIT = "COVERAGE_USER_HIT"
    USERS_IN_GT = "USERS_IN_GT"
    DIVERSITY_GINI = "DIVERSITY_GINI"
    SHANNON_ENTROPY = "SHANNON_ENTROPY"

    RATIO_DIVERSITY_HERFINDAHL = "RATIO_DIVERSITY_HERFINDAHL"
    RATIO_DIVERSITY_GINI = "RATIO_DIVERSITY_GINI"
    RATIO_SHANNON_ENTROPY = "RATIO_SHANNON_ENTROPY"
    RATIO_AVERAGE_POPULARITY = "RATIO_AVERAGE_POPULARITY"
    RATIO_NOVELTY = "RATIO_NOVELTY"


def _create_empty_metrics_dict(cutoff_list, n_items, n_users, URM_train, URM_test, ignore_items, ignore_users, diversity_similarity_object):

    empty_dict = {}

    for cutoff in cutoff_list:

        cutoff_dict = {}

        for metric in EvaluatorMetrics:
            if metric == EvaluatorMetrics.COVERAGE_ITEM:
                cutoff_dict[metric.value] = Coverage_Item(n_items, ignore_items)

            elif metric == EvaluatorMetrics.COVERAGE_ITEM_HIT:
                cutoff_dict[metric.value] = Coverage_Item_HIT(n_items, ignore_items)

            elif metric == EvaluatorMetrics.ITEMS_IN_GT:
                cutoff_dict[metric.value] = Items_In_GT(URM_test, ignore_items)

            elif metric == EvaluatorMetrics.DIVERSITY_GINI:
                cutoff_dict[metric.value] = Gini_Diversity(n_items, ignore_items)

            elif metric == EvaluatorMetrics.SHANNON_ENTROPY:
                cutoff_dict[metric.value] = Shannon_Entropy(n_items, ignore_items)

            elif metric == EvaluatorMetrics.COVERAGE_USER:
                cutoff_dict[metric.value] = Coverage_User(n_users, ignore_users)

            elif metric == EvaluatorMetrics.COVERAGE_USER_HIT:
                cutoff_dict[metric.value] = Coverage_User_HIT(n_users, ignore_users)

            elif metric == EvaluatorMetrics.USERS_IN_GT:
                cutoff_dict[metric.value] = Users_In_GT(URM_test, ignore_users)

            elif metric == EvaluatorMetrics.DIVERSITY_MEAN_INTER_LIST:
                cutoff_dict[metric.value] = Diversity_MeanInterList(n_items, cutoff)

            elif metric == EvaluatorMetrics.DIVERSITY_HERFINDAHL:
                cutoff_dict[metric.value] = Diversity_Herfindahl(n_items, ignore_items)

            elif metric == EvaluatorMetrics.NOVELTY:
                cutoff_dict[metric.value] = Novelty(URM_train)

            elif metric == EvaluatorMetrics.AVERAGE_POPULARITY:
                cutoff_dict[metric.value] = AveragePopularity(URM_train)

            elif metric == EvaluatorMetrics.MAP:
                cutoff_dict[metric.value] = MAP()

            elif metric == EvaluatorMetrics.MAP_MIN_DEN:
                cutoff_dict[metric.value] = MAP_MIN_DEN()

            elif metric == EvaluatorMetrics.MRR:
                cutoff_dict[metric.value] = MRR()

            elif metric == EvaluatorMetrics.HIT_RATE:
                cutoff_dict[metric.value] = HIT_RATE()

            elif metric == EvaluatorMetrics.RATIO_DIVERSITY_GINI:
                cutoff_dict[metric.value] = Ratio_Diversity_Gini(URM_train, ignore_items)

            elif metric == EvaluatorMetrics.RATIO_DIVERSITY_HERFINDAHL:
                cutoff_dict[metric.value] = Ratio_Diversity_Herfindahl(URM_train, ignore_items)

            elif metric == EvaluatorMetrics.RATIO_SHANNON_ENTROPY:
                cutoff_dict[metric.value] = Ratio_Shannon_Entropy(URM_train, ignore_items)

            elif metric == EvaluatorMetrics.RATIO_AVERAGE_POPULARITY:
                cutoff_dict[metric.value] = Ratio_AveragePopularity(URM_train)

            elif metric == EvaluatorMetrics.RATIO_NOVELTY:
                cutoff_dict[metric.value] = Ratio_Novelty(URM_train)

            elif metric == EvaluatorMetrics.DIVERSITY_SIMILARITY:
                    if diversity_similarity_object is not None:
                        cutoff_dict[metric.value] = copy.deepcopy(diversity_similarity_object)
            else:
                cutoff_dict[metric.value] = 0.0


        empty_dict[cutoff] = cutoff_dict

    return  empty_dict






def get_result_string_df(results_run_df, n_decimals=7):

    output_str = ""

    for cutoff in results_run_df.index:

        output_str += "CUTOFF: {} - ".format(cutoff)

        for metric in results_run_df.columns:
            output_str += "{}: {:.{n_decimals}f}, ".format(metric, results_run_df.loc[cutoff, metric], n_decimals = n_decimals)

        output_str += "\n"

    return output_str

def _remove_item_interactions(URM, item_list):

    URM = sps.csc_matrix(URM.copy())

    for item_index in item_list:

        start_pos = URM.indptr[item_index]
        end_pos = URM.indptr[item_index+1]

        URM.data[start_pos:end_pos] = np.zeros_like(URM.data[start_pos:end_pos])

    URM.eliminate_zeros()
    URM = sps.csr_matrix(URM)

    return URM



def _prune_users(URM_test, ignore_items_ID, min_ratings_per_user):
    """
    Remove users with a number of ratings lower than min_ratings_per_user, excluding the items to be ignored in the evaluation
    :param URM_test:
    :param ignore_items_ID:
    :param min_ratings_per_user:
    :return:
    """

    users_to_evaluate_mask = np.zeros(URM_test.shape[0], dtype=np.bool)

    URM_test = _remove_item_interactions(URM_test, ignore_items_ID)
    URM_test = sps.csr_matrix(URM_test)

    rows = URM_test.indptr
    n_user_ratings = np.ediff1d(rows)
    new_mask = n_user_ratings >= min_ratings_per_user

    users_to_evaluate_mask = np.logical_or(users_to_evaluate_mask, new_mask)

    return URM_test, users_to_evaluate_mask


class Evaluator(object):
    """Abstract Evaluator"""

    EVALUATOR_NAME = "Evaluator_Base_Class"

    def __init__(self, URM_test, cutoff_list, min_ratings_per_user=1, exclude_seen=True,
                 diversity_object = None,
                 ignore_items = None,
                 ignore_users = None,
                 verbose=True):

        super(Evaluator, self).__init__()

        self.verbose = verbose

        if ignore_items is None:
            self.ignore_items_flag = False
            self.ignore_items_ID = np.array([])
        else:
            self._print("Ignoring {} Items".format(len(ignore_items)))
            self.ignore_items_flag = True
            self.ignore_items_ID = np.array(ignore_items)

        self.cutoff_list = cutoff_list.copy()
        self.max_cutoff = max(self.cutoff_list)

        self.min_ratings_per_user = min_ratings_per_user
        self.exclude_seen = exclude_seen
        self.diversity_object = diversity_object
        self.n_users, self.n_items = URM_test.shape

        # Prune users with an insufficient number of ratings
        self.URM_test, users_to_evaluate_mask = _prune_users(URM_test, self.ignore_items_ID, min_ratings_per_user)

        if not np.all(users_to_evaluate_mask):
            self._print("Ignoring {} ({:4.1f}%) Users that have less than {} test interactions".format(len(users_to_evaluate_mask) - np.sum(users_to_evaluate_mask),
                                                                                                     100*np.sum(np.logical_not(users_to_evaluate_mask))/len(users_to_evaluate_mask), min_ratings_per_user))

        self.users_to_evaluate = np.arange(self.n_users)[users_to_evaluate_mask]

        if ignore_users is not None:
            self._print("Ignoring {} Users".format(len(ignore_users)))
            self.ignore_users_ID = np.array(ignore_users)
            self.users_to_evaluate = set(self.users_to_evaluate) - set(ignore_users)
        else:
            self.ignore_users_ID = np.array([])

        self.users_to_evaluate = list(self.users_to_evaluate)

        # Those will be set at each new evaluation
        self._start_time = np.nan
        self._start_time_print = np.nan
        self._n_users_evaluated = np.nan


    def _print(self, string):

        if self.verbose:
            print("{}: {}".format(self.EVALUATOR_NAME, string))


    def evaluateRecommender(self, recommender_object):
        """
        :param recommender_object: the trained recommender object, a BaseRecommender subclass
        :param URM_test_list: list of URMs to test the recommender against, or a single URM object
        :param cutoff_list: list of cutoffs to be use to report the scores, or a single cutoff
        :return results_df: dataframe with index the cutoff and columns the metric
        :return results_run_string: printable result string
        """

        if self.ignore_items_flag:
            recommender_object.set_items_to_ignore(self.ignore_items_ID)

        self._start_time = time.time()
        self._start_time_print = time.time()
        self._n_users_evaluated = 0

        results_dict = self._run_evaluation_on_selected_users(recommender_object, self.users_to_evaluate)


        if self._n_users_evaluated > 0:

            for cutoff in self.cutoff_list:
                results_current_cutoff = results_dict[cutoff]

                for key in results_current_cutoff.keys():
                    value = results_current_cutoff[key]

                    if isinstance(value, _Metrics_Object):
                        results_current_cutoff[key] = value.get_metric_value()
                    else:
                        results_current_cutoff[key] = value/self._n_users_evaluated

                if EvaluatorMetrics.F1.value in results_current_cutoff:
                    precision_ = results_current_cutoff[EvaluatorMetrics.PRECISION.value]
                    recall_ = results_current_cutoff[EvaluatorMetrics.RECALL.value]

                    if precision_ + recall_ != 0:
                        # F1 micro averaged: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.8244&rep=rep1&type=pdf
                        results_current_cutoff[EvaluatorMetrics.F1.value] = 2 * (precision_ * recall_) / (precision_ + recall_)


        else:
            self._print("WARNING: No users had a sufficient number of relevant items")

        if self.ignore_items_flag:
            recommender_object.reset_items_to_ignore()

        results_df = pd.DataFrame(columns=results_dict[self.cutoff_list[0]].keys(),
                                  index=self.cutoff_list)
        results_df.index.rename("cutoff", inplace = True)

        for cutoff in results_dict.keys():
            results_df.loc[cutoff] = results_dict[cutoff]

        results_run_string = get_result_string_df(results_df)

        return results_df, results_run_string



    def get_user_relevant_items(self, user_id):

        assert self.URM_test.getformat() == "csr", "Evaluator_Base_Class: URM_test is not CSR, this will cause errors in getting relevant items"
        return self.URM_test.indices[self.URM_test.indptr[user_id]:self.URM_test.indptr[user_id+1]]


    def get_user_test_ratings(self, user_id):

        assert self.URM_test.getformat() == "csr", "Evaluator_Base_Class: URM_test is not CSR, this will cause errors in relevant items ratings"
        return self.URM_test.data[self.URM_test.indptr[user_id]:self.URM_test.indptr[user_id+1]]



    def _compute_metrics_on_recommendation_list(self, test_user_batch_array, recommended_items_batch_list, scores_batch, results_dict):

        assert len(recommended_items_batch_list) == len(test_user_batch_array), "{}: recommended_items_batch_list contained recommendations for {} users, expected was {}".format(
            self.EVALUATOR_NAME, len(recommended_items_batch_list), len(test_user_batch_array))

        assert scores_batch.shape[0] == len(test_user_batch_array), "{}: scores_batch contained scores for {} users, expected was {}".format(
            self.EVALUATOR_NAME, scores_batch.shape[0], len(test_user_batch_array))

        assert scores_batch.shape[1] == self.n_items, "{}: scores_batch contained scores for {} items, expected was {}".format(
            self.EVALUATOR_NAME, scores_batch.shape[1], self.n_items)


        # Compute recommendation quality for each user in batch
        for batch_user_index in range(len(recommended_items_batch_list)):

            test_user = test_user_batch_array[batch_user_index]

            relevant_items = self.get_user_relevant_items(test_user)

            # Being the URM CSR, the indices are the non-zero column indexes
            recommended_items = recommended_items_batch_list[batch_user_index]
            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            self._n_users_evaluated += 1

            for cutoff in self.cutoff_list:

                results_current_cutoff = results_dict[cutoff]

                is_relevant_current_cutoff = is_relevant[0:cutoff]
                recommended_items_current_cutoff = recommended_items[0:cutoff]

                results_current_cutoff[EvaluatorMetrics.PRECISION.value]            += precision(is_relevant_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.PRECISION_RECALL_MIN_DEN.value]   += precision_recall_min_denominator(is_relevant_current_cutoff, len(relevant_items))
                results_current_cutoff[EvaluatorMetrics.RECALL.value]               += recall(is_relevant_current_cutoff, relevant_items)
                results_current_cutoff[EvaluatorMetrics.NDCG.value]                 += ndcg(recommended_items_current_cutoff, relevant_items, relevance=self.get_user_test_ratings(test_user), at=cutoff)
                results_current_cutoff[EvaluatorMetrics.ARHR.value]                 += arhr_all_hits(is_relevant_current_cutoff)

                results_current_cutoff[EvaluatorMetrics.MRR.value].add_recommendations(is_relevant_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.MAP.value].add_recommendations(is_relevant_current_cutoff, relevant_items)
                results_current_cutoff[EvaluatorMetrics.MAP_MIN_DEN.value].add_recommendations(is_relevant_current_cutoff, relevant_items)
                results_current_cutoff[EvaluatorMetrics.HIT_RATE.value].add_recommendations(is_relevant_current_cutoff)

                results_current_cutoff[EvaluatorMetrics.NOVELTY.value].add_recommendations(recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.AVERAGE_POPULARITY.value].add_recommendations(recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.DIVERSITY_GINI.value].add_recommendations(recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.SHANNON_ENTROPY.value].add_recommendations(recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.COVERAGE_ITEM.value].add_recommendations(recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.COVERAGE_ITEM_HIT.value].add_recommendations(recommended_items_current_cutoff, is_relevant_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.COVERAGE_USER.value].add_recommendations(recommended_items_current_cutoff, test_user)
                results_current_cutoff[EvaluatorMetrics.COVERAGE_USER_HIT.value].add_recommendations(is_relevant_current_cutoff, test_user)
                results_current_cutoff[EvaluatorMetrics.DIVERSITY_MEAN_INTER_LIST.value].add_recommendations(recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.DIVERSITY_HERFINDAHL.value].add_recommendations(recommended_items_current_cutoff)

                results_current_cutoff[EvaluatorMetrics.RATIO_SHANNON_ENTROPY.value].add_recommendations(recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.RATIO_DIVERSITY_HERFINDAHL.value].add_recommendations(recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.RATIO_DIVERSITY_GINI.value].add_recommendations(recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.RATIO_NOVELTY.value].add_recommendations(recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.RATIO_AVERAGE_POPULARITY.value].add_recommendations(recommended_items_current_cutoff)

                if EvaluatorMetrics.DIVERSITY_SIMILARITY.value in results_current_cutoff:
                    results_current_cutoff[EvaluatorMetrics.DIVERSITY_SIMILARITY.value].add_recommendations(recommended_items_current_cutoff)


        if time.time() - self._start_time_print > 300 or self._n_users_evaluated==len(self.users_to_evaluate):

            elapsed_time = time.time()-self._start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

            self._print("Processed {} ({:4.1f}%) in {:.2f} {}. Users per second: {:.0f}".format(
                          self._n_users_evaluated,
                          100.0* float(self._n_users_evaluated)/len(self.users_to_evaluate),
                          new_time_value, new_time_unit,
                          float(self._n_users_evaluated)/elapsed_time if elapsed_time>0.0 else np.nan))

            sys.stdout.flush()
            sys.stderr.flush()

            self._start_time_print = time.time()


        return results_dict







class EvaluatorHoldout(Evaluator):
    """EvaluatorHoldout"""

    EVALUATOR_NAME = "EvaluatorHoldout"

    def __init__(self, URM_test_list, cutoff_list, min_ratings_per_user=1, exclude_seen=True,
                 diversity_object = None,
                 ignore_items = None,
                 ignore_users = None,
                 verbose=True):


        super(EvaluatorHoldout, self).__init__(URM_test_list, cutoff_list,
                                               diversity_object = diversity_object,
                                               min_ratings_per_user =min_ratings_per_user, exclude_seen=exclude_seen,
                                               ignore_items = ignore_items, ignore_users = ignore_users,
                                               verbose = verbose)





    def _run_evaluation_on_selected_users(self, recommender_object, users_to_evaluate, block_size = None):

        if block_size is None:
            # Reduce block size if estimated memory requirement exceeds 4 GB
            block_size = min([1000, int(4*1e9*8/64/self.n_items), len(users_to_evaluate)])


        results_dict = _create_empty_metrics_dict(self.cutoff_list,
                                                  self.n_items, self.n_users,
                                                  recommender_object.get_URM_train(),
                                                  self.URM_test,
                                                  self.ignore_items_ID,
                                                  self.ignore_users_ID,
                                                  self.diversity_object)


        if self.ignore_items_flag:
            recommender_object.set_items_to_ignore(self.ignore_items_ID)

        # Start from -block_size to ensure it to be 0 at the first block
        user_batch_start = 0
        user_batch_end = 0

        while user_batch_start < len(users_to_evaluate):

            user_batch_end = user_batch_start + block_size
            user_batch_end = min(user_batch_end, len(users_to_evaluate))

            test_user_batch_array = np.array(users_to_evaluate[user_batch_start:user_batch_end])
            user_batch_start = user_batch_end

            # Compute predictions for a batch of users using vectorization, much more efficient than computing it one at a time
            recommended_items_batch_list, scores_batch = recommender_object.recommend(test_user_batch_array,
                                                                      remove_seen_flag=self.exclude_seen,
                                                                      cutoff = self.max_cutoff,
                                                                      remove_top_pop_flag=False,
                                                                      remove_custom_items_flag=self.ignore_items_flag,
                                                                      return_scores = True
                                                                     )

            results_dict = self._compute_metrics_on_recommendation_list(test_user_batch_array = test_user_batch_array,
                                                         recommended_items_batch_list = recommended_items_batch_list,
                                                         scores_batch = scores_batch,
                                                         results_dict = results_dict)


        return results_dict




class EvaluatorNegativeItemSample(Evaluator):
    """EvaluatorNegativeItemSample"""

    EVALUATOR_NAME = "EvaluatorNegativeItemSample"

    def __init__(self, URM_test_list, URM_test_negative, cutoff_list, min_ratings_per_user=1, exclude_seen=True,
                 diversity_object = None,
                 ignore_items = None,
                 ignore_users = None):
        """

        The EvaluatorNegativeItemSample computes the recommendations by sorting the test items as well as the test_negative items
        It ensures that each item appears only once even if it is listed in both matrices

        :param URM_test_list:
        :param URM_test_negative: Items to rank together with the test items
        :param cutoff_list:
        :param min_ratings_per_user:
        :param exclude_seen:
        :param diversity_object:
        :param ignore_items:
        :param ignore_users:
        """
        super(EvaluatorNegativeItemSample, self).__init__(URM_test_list, cutoff_list,
                                                          diversity_object = diversity_object,
                                                          min_ratings_per_user = min_ratings_per_user, exclude_seen=exclude_seen,
                                                          ignore_items = ignore_items, ignore_users = ignore_users)


        self.URM_items_to_rank = sps.csr_matrix(self.URM_test.copy().astype(np.bool)) + sps.csr_matrix(URM_test_negative.copy().astype(np.bool))
        self.URM_items_to_rank.eliminate_zeros()
        self.URM_items_to_rank.data = np.ones_like(self.URM_items_to_rank.data)



    def _get_user_specific_items_to_compute(self, user_id):

        start_pos = self.URM_items_to_rank.indptr[user_id]
        end_pos = self.URM_items_to_rank.indptr[user_id+1]

        items_to_compute = self.URM_items_to_rank.indices[start_pos:end_pos]

        return items_to_compute



    def _run_evaluation_on_selected_users(self, recommender_object, users_to_evaluate, block_size = None):



        results_dict = _create_empty_metrics_dict(self.cutoff_list,
                                                  self.n_items, self.n_users,
                                                  recommender_object.get_URM_train(),
                                                  self.URM_test,
                                                  self.ignore_items_ID,
                                                  self.ignore_users_ID,
                                                  self.diversity_object)


        if self.ignore_items_flag:
            recommender_object.set_items_to_ignore(self.ignore_items_ID)



        for test_user in users_to_evaluate:

            items_to_compute = self._get_user_specific_items_to_compute(test_user)

            recommended_items, all_items_predicted_ratings = recommender_object.recommend(np.atleast_1d(test_user),
                                                              remove_seen_flag=self.exclude_seen,
                                                              cutoff = self.max_cutoff,
                                                              remove_top_pop_flag=False,
                                                              items_to_compute = items_to_compute,
                                                              remove_custom_items_flag=self.ignore_items_flag,
                                                              return_scores = True
                                                             )


            results_dict = self._compute_metrics_on_recommendation_list(test_user_batch_array = [test_user],
                                                         recommended_items_batch_list = recommended_items,
                                                         scores_batch = all_items_predicted_ratings,
                                                         results_dict = results_dict)


        return results_dict

