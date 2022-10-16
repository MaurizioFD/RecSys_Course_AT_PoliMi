#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/2018

@author: Maurizio Ferrari Dacrema
"""

import traceback, os, shutil

from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender

from Evaluation.Evaluator import EvaluatorHoldout, EvaluatorNegativeItemSample
from Data_manager.Movielens.Movielens1MReader import Movielens1MReader
from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping


def write_log_string(log_file, string):
    log_file.write(string)
    log_file.flush()


def _get_instance(recommender_class, URM_train, ICM_all, UCM_all):

    if issubclass(recommender_class, BaseItemCBFRecommender):
        recommender_object = recommender_class(URM_train, ICM_all)
    elif issubclass(recommender_class, BaseUserCBFRecommender):
        recommender_object = recommender_class(URM_train, UCM_all)
    else:
        recommender_object = recommender_class(URM_train)

    return recommender_object


def run_recommender(recommender_class):
    temp_save_file_folder = "./result_experiments/__temp_model/"

    if not os.path.isdir(temp_save_file_folder):
        os.makedirs(temp_save_file_folder)

    try:
        dataset_object = Movielens1MReader()

        dataSplitter = DataSplitter_leave_k_out(dataset_object, k_out_value=2)

        dataSplitter.load_data()
        URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()
        ICM_all = dataSplitter.get_loaded_ICM_dict()["ICM_genres"]
        UCM_all = dataSplitter.get_loaded_UCM_dict()["UCM_all"]

        write_log_string(log_file, "On Recommender {}\n".format(recommender_class))

        recommender_object = _get_instance(recommender_class, URM_train, ICM_all, UCM_all)

        if isinstance(recommender_object, Incremental_Training_Early_Stopping):
            fit_params = {"epochs": 15}
        else:
            fit_params = {}

        recommender_object.fit(**fit_params)

        write_log_string(log_file, "Fit OK, ")



        evaluator = EvaluatorHoldout(URM_test, [5], exclude_seen = True)
        results_df, results_run_string = evaluator.evaluateRecommender(recommender_object)

        write_log_string(log_file, "EvaluatorHoldout OK, ")



        evaluator = EvaluatorNegativeItemSample(URM_test, URM_train, [5], exclude_seen = True)
        _, _ = evaluator.evaluateRecommender(recommender_object)

        write_log_string(log_file, "EvaluatorNegativeItemSample OK, ")


        items_to_compute_not_sorted = np.random.randint(0,URM_train.shape[1], size = 300)
        items_to_compute_sorted = np.sort(items_to_compute_not_sorted)
        for user_id in range(URM_train.shape[0]):
            recommendations_sorted, scores_sorted = recommender_object.recommend(user_id, cutoff = 50, items_to_compute = items_to_compute_sorted, return_scores = True)
            recommendations_not_sorted, scores_not_sorted = recommender_object.recommend(user_id, cutoff = 50, items_to_compute = items_to_compute_not_sorted, return_scores = True)

            # try:
            assert np.equal(recommendations_sorted, recommendations_not_sorted).all()
            assert np.allclose(scores_sorted, scores_not_sorted, atol=1e-5)

            scores_sorted[0,items_to_compute_sorted] = -np.inf
            assert np.isinf(scores_sorted).all()
            # except:
            #     # np.where(np.logical_not(scores_sorted == scores_not_sorted))[1]
            #     pass

        write_log_string(log_file, "items_to_compute in the right order OK, ")




        recommender_object.save_model(temp_save_file_folder, file_name="temp_model")

        write_log_string(log_file, "save_model OK, ")



        recommender_object = _get_instance(recommender_class, URM_train, ICM_all, UCM_all)
        recommender_object.load_model(temp_save_file_folder, file_name="temp_model")

        evaluator = EvaluatorHoldout(URM_test, [5], exclude_seen = True)
        result_df_load, results_run_string_2 = evaluator.evaluateRecommender(recommender_object)

        print(results_run_string)
        print(results_run_string_2)
        assert results_df.equals(result_df_load), "The results of the original model should be equal to that of the loaded one"

        write_log_string(log_file, "load_model OK, ")




        from Recommenders.DataIO import DataIO
        dataIO = DataIO(temp_save_file_folder)
        data = dataIO.load_data("temp_model.zip")

        shutil.rmtree(temp_save_file_folder, ignore_errors = True)

        write_log_string(log_file, " PASS\n")
        write_log_string(log_file, results_run_string + "\n\n")



    except Exception as e:

        print("On Recommender {} Exception {}".format(recommender_class, str(e)))
        log_file.write("On Recommender {} Exception {}\n\n\n".format(recommender_class, str(e)))
        log_file.flush()

        traceback.print_exc()


from Recommenders.Recommender_import_list import *


if __name__ == '__main__':


    log_file_name = "./result_experiments/run_test_recommender.txt"


    recommender_list = [
        Random,
        TopPop,
        GlobalEffects,
        UserKNNCFRecommender,
        ItemKNNCFRecommender,
        ItemKNNCBFRecommender,
        P3alphaRecommender,
        RP3betaRecommender,
        SLIM_BPR_Cython,
        SLIMElasticNetRecommender,
        MatrixFactorization_BPR_Cython,
        MatrixFactorization_FunkSVD_Cython,
        MatrixFactorization_AsySVD_Cython,
        PureSVDRecommender,
        IALSRecommender,
        EASE_R_Recommender,
    ]

    log_file = open(log_file_name, "w")



    for recommender_class in recommender_list:
        run_recommender(recommender_class)
    #
    # pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    # resultList = pool.map(run_dataset, dataset_list)

