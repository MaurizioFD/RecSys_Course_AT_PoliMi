#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/09/17

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import unittest



class MyTestCase(unittest.TestCase):

    def test_Gini_Index(self):

        from Base.Evaluation.metrics import Gini_Index

        n_items = 1000

        gini_index = Gini_Index(n_items, ignore_items=np.array([]))

        gini_index.recommended_counter = np.ones(n_items)
        assert np.isclose(0.0, gini_index.get_metric_value(), atol=1e-3), "Gini_Index metric incorrect"

        gini_index.recommended_counter = np.ones(n_items)*1e-12
        gini_index.recommended_counter[0] = 1.0
        assert np.isclose(1.0, gini_index.get_metric_value(), atol=1e-3), "Gini_Index metric incorrect"

        gini_index.recommended_counter = np.random.uniform(0, 1, n_items)
        assert  np.isclose(0.3, gini_index.get_metric_value(), atol=1e-1), "Gini_Index metric incorrect"



    def test_Shannon_Entropy(self):

        from Base.Evaluation.metrics import Shannon_Entropy

        n_items = 1000

        shannon_entropy = Shannon_Entropy(n_items, ignore_items=np.array([]))

        shannon_entropy.recommended_counter = np.ones(n_items)
        assert np.isclose(9.96, shannon_entropy.get_metric_value(), atol=1e-2), "metric incorrect"

        shannon_entropy.recommended_counter = np.zeros(n_items)
        shannon_entropy.recommended_counter[0] = 1.0
        assert np.isclose(0.0, shannon_entropy.get_metric_value(), atol=1e-3), "metric incorrect"

        shannon_entropy.recommended_counter = np.random.uniform(0, 100, n_items).astype(np.int)
        assert  np.isclose(9.6, shannon_entropy.get_metric_value(), atol=1e-1), "metric incorrect"

        # n_items = 10000
        #
        # shannon_entropy.recommended_counter = np.random.normal(0, 50, n_items).astype(np.int)
        # shannon_entropy.recommended_counter += abs(min(shannon_entropy.recommended_counter))
        # assert  np.isclose(9.8, shannon_entropy.get_metric_value(), atol=1e-1), "metric incorrect"




    def test_Diversity_list_all_equals(self):

        from Base.Evaluation.metrics import Diversity_MeanInterList
        import scipy.sparse as sps

        n_items = 3
        n_users = 10
        cutoff = min(5, n_items)

        # create recommendation list
        URM_predicted_row = []
        URM_predicted_col = []

        diversity_list = Diversity_MeanInterList(n_items, cutoff)
        item_id_list = np.arange(0, n_items, dtype=np.int)

        for n_user in range(n_users):

            np.random.shuffle(item_id_list)
            recommended = item_id_list[:cutoff]
            URM_predicted_row.extend([n_user]*cutoff)
            URM_predicted_col.extend(recommended)

            diversity_list.add_recommendations(recommended)


        object_diversity = diversity_list.get_metric_value()

        URM_predicted_data = np.ones_like(URM_predicted_row)

        URM_predicted_sparse = sps.csr_matrix((URM_predicted_data, (URM_predicted_row, URM_predicted_col)), dtype=np.int)

        co_counts = URM_predicted_sparse.dot(URM_predicted_sparse.T).toarray()
        np.fill_diagonal(co_counts, 0)

        all_user_couples_count = n_users**2 - n_users

        diversity_cumulative = 1 - co_counts/cutoff
        np.fill_diagonal(diversity_cumulative, 0)

        diversity_cooccurrence = diversity_cumulative.sum()/all_user_couples_count

        assert  np.isclose(diversity_cooccurrence, object_diversity, atol=1e-4), "metric incorrect"






    def test_Diversity_list(self):

        from Base.Evaluation.metrics import Diversity_MeanInterList
        import scipy.sparse as sps

        n_items = 500
        n_users = 1000
        cutoff = 10

        # create recommendation list
        URM_predicted_row = []
        URM_predicted_col = []

        diversity_list = Diversity_MeanInterList(n_items, cutoff)
        item_id_list = np.arange(0, n_items, dtype=np.int)

        for n_user in range(n_users):

            np.random.shuffle(item_id_list)
            recommended = item_id_list[:cutoff]
            URM_predicted_row.extend([n_user]*cutoff)
            URM_predicted_col.extend(recommended)

            diversity_list.add_recommendations(recommended)


        object_diversity = diversity_list.get_metric_value()

        URM_predicted_data = np.ones_like(URM_predicted_row)

        URM_predicted_sparse = sps.csr_matrix((URM_predicted_data, (URM_predicted_row, URM_predicted_col)), dtype=np.int)

        co_counts = URM_predicted_sparse.dot(URM_predicted_sparse.T).toarray()
        np.fill_diagonal(co_counts, 0)

        all_user_couples_count = n_users**2 - n_users

        diversity_cumulative = 1 - co_counts/cutoff
        np.fill_diagonal(diversity_cumulative, 0)

        diversity_cooccurrence = diversity_cumulative.sum()/all_user_couples_count

        assert  np.isclose(diversity_cooccurrence, object_diversity, atol=1e-4), "metric incorrect"




if __name__ == '__main__':

    unittest.main()

