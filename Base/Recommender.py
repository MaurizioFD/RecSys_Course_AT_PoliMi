#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import pickle


class Recommender(object):
    """Abstract Recommender"""

    RECOMMENDER_NAME = "Recommender_Base_Class"

    def __init__(self):

        super(Recommender, self).__init__()

        self.URM_train = None
        self.sparse_weights = True
        self.normalize = False

        self.filterTopPop = False
        self.filterTopPop_ItemsID = np.array([], dtype=np.int)

        self.items_to_ignore_flag = False
        self.items_to_ignore_ID = np.array([], dtype=np.int)


    def fit(self):
        pass

    def get_URM_train(self):
        return self.URM_train.copy()


    def set_items_to_ignore(self, items_to_ignore):

        self.items_to_ignore_flag = True
        self.items_to_ignore_ID = np.array(items_to_ignore, dtype=np.int)

    def reset_items_to_ignore(self):

        self.items_to_ignore_flag = False
        self.items_to_ignore_ID = np.array([], dtype=np.int)


    def _remove_TopPop_on_scores(self, scores_batch):
        scores_batch[:, self.filterTopPop_ItemsID] = -np.inf
        return scores_batch


    def _remove_CustomItems_on_scores(self, scores_batch):
        scores_batch[:, self.items_to_ignore_ID] = -np.inf
        return scores_batch


    def _remove_seen_on_scores(self, user_id, scores):

        assert self.URM_train.getformat() == "csr", "Recommender_Base_Class: URM_train is not CSR, this will cause errors in filtering seen items"

        seen = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

        scores[seen] = -np.inf
        return scores







    def compute_item_score(self, user_id):
        raise NotImplementedError("Recommender: compute_item_score not assigned for current recommender, unable to compute prediction scores")






    def recommend(self, user_id_array, cutoff = None, remove_seen_flag=True, remove_top_pop_flag = False, remove_CustomItems_flag = False):

        # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False


        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        # Compute the scores using the model-specific function
        # Vectorize over all users in user_id_array
        scores_batch = self.compute_item_score(user_id_array)


        # if self.normalize:
        #     # normalization will keep the scores in the same range
        #     # of value of the ratings in dataset
        #     user_profile = self.URM_train[user_id]
        #
        #     rated = user_profile.copy()
        #     rated.data = np.ones_like(rated.data)
        #     if self.sparse_weights:
        #         den = rated.dot(self.W_sparse).toarray().ravel()
        #     else:
        #         den = rated.dot(self.W).ravel()
        #     den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
        #     scores /= den


        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]

            if remove_seen_flag:
                scores_batch[user_index,:] = self._remove_seen_on_scores(user_id, scores_batch[user_index, :])

            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index
            # relevant_items_partition = (-scores_user).argpartition(cutoff)[0:cutoff]
            # relevant_items_partition_sorting = np.argsort(-scores_user[relevant_items_partition])
            # ranking = relevant_items_partition[relevant_items_partition_sorting]
            #
            # ranking_list.append(ranking)


        if remove_top_pop_flag:
            scores_batch = self._remove_TopPop_on_scores(scores_batch)

        if remove_CustomItems_flag:
            scores_batch = self._remove_CustomItems_on_scores(scores_batch)

        # scores_batch = np.arange(0,3260).reshape((1, -1))
        # scores_batch = np.repeat(scores_batch, 1000, axis = 0)

        # relevant_items_partition is block_size x cutoff
        relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:,0:cutoff]

        # Get original value and sort it
        # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
        # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
        relevant_items_partition_original_value = scores_batch[np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        ranking = relevant_items_partition[np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        ranking_list = ranking.tolist()


        # Return single list for one user, instead of list of lists
        if single_user:
            ranking_list = ranking_list[0]

        return ranking_list






    def evaluateRecommendations(self, URM_test, at=5, minRatingsPerUser=1, exclude_seen=True,
                                filterCustomItems = np.array([], dtype=np.int),
                                filterCustomUsers = np.array([], dtype=np.int)):
        """
        Speed info:
        - Sparse weighgs: batch mode is 2x faster than sequential
        - Dense weighgts: batch and sequential speed are equivalent


        :param URM_test:            URM to be used for testing
        :param at: 5                    Length of the recommended items
        :param minRatingsPerUser: 1     Users with less than this number of interactions will not be evaluated
        :param exclude_seen: True       Whether to remove already seen items from the recommended items

        :param mode: 'sequential', 'parallel', 'batch'
        :param filterTopPop: False or decimal number        Percentage of items to be removed from recommended list and testing interactions
        :param filterCustomItems: Array, default empty           Items ID to NOT take into account when recommending
        :param filterCustomUsers: Array, default empty           Users ID to NOT take into account when recommending
        :return:
        """

        import warnings

        warnings.warn("DEPRECATED! Use Base.Evaluation.SequentialEvaluator.evaluateRecommendations()", DeprecationWarning)


        from Base.Evaluation.Evaluator import SequentialEvaluator

        evaluator = SequentialEvaluator(URM_test, [at], exclude_seen= exclude_seen,
                                        minRatingsPerUser=minRatingsPerUser,
                                        ignore_items=filterCustomItems, ignore_users=filterCustomUsers)

        results_run, results_run_string = evaluator.evaluateRecommender(self)

        results_run = results_run[at]

        results_run_lowercase = {}

        for key in results_run.keys():
            results_run_lowercase[key.lower()] = results_run[key]


        return results_run_lowercase













    def saveModel(self, folder_path, file_name = None):
        raise NotImplementedError("Recommender: saveModel not implemented")




    def loadModel(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Loading model from file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))


        data_dict = pickle.load(open(folder_path + file_name, "rb"))

        for attrib_name in data_dict.keys():
             self.__setattr__(attrib_name, data_dict[attrib_name])


        print("{}: Loading complete".format(self.RECOMMENDER_NAME))

