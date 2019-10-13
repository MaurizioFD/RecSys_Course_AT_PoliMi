#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/2018

@author: Maurizio Ferrari Dacrema
"""

import traceback, os, shutil


from Base.Evaluation.Evaluator import EvaluatorHoldout, EvaluatorNegativeItemSample
from Data_manager.Movielens1M.Movielens1MReader import Movielens1MReader
from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping


def write_log_string(log_file, string):
    log_file.write(string)
    log_file.flush()



def run_recommender(recommender_class):



    temp_save_file_folder = "./result_experiments/__temp_model/"

    if not os.path.isdir(temp_save_file_folder):
        os.makedirs(temp_save_file_folder)

    try:
        dataset_object = Movielens1MReader()

        dataSplitter = DataSplitter_leave_k_out(dataset_object, k_out_value=2)

        dataSplitter.load_data()
        URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

        write_log_string(log_file, "On Recommender {}\n".format(recommender_class))



        recommender_object = recommender_class(URM_train)

        if isinstance(recommender_object, Incremental_Training_Early_Stopping):
            fit_params = {"epochs": 15}
        else:
            fit_params = {}

        recommender_object.fit(**fit_params)

        write_log_string(log_file, "Fit OK, ")



        evaluator = EvaluatorHoldout(URM_test, [5], exclude_seen=True)
        _, results_run_string = evaluator.evaluateRecommender(recommender_object)

        write_log_string(log_file, "EvaluatorHoldout OK, ")



        evaluator = EvaluatorNegativeItemSample(URM_test, URM_train, [5], exclude_seen=True)
        _, _ = evaluator.evaluateRecommender(recommender_object)

        write_log_string(log_file, "EvaluatorNegativeItemSample OK, ")



        recommender_object.saveModel(temp_save_file_folder, file_name="temp_model")

        write_log_string(log_file, "saveModel OK, ")



        recommender_object = recommender_class(URM_train)
        recommender_object.loadModel(temp_save_file_folder, file_name="temp_model")

        evaluator = EvaluatorHoldout(URM_test, [5], exclude_seen=True)
        _, results_run_string_2 = evaluator.evaluateRecommender(recommender_object)

        write_log_string(log_file, "loadModel OK, ")



        shutil.rmtree(temp_save_file_folder, ignore_errors=True)

        write_log_string(log_file, " PASS\n")
        write_log_string(log_file, results_run_string + "\n\n")



    except Exception as e:

        print("On Recommender {} Exception {}".format(recommender_class, str(e)))
        log_file.write("On Recommender {} Exception {}\n\n\n".format(recommender_class, str(e)))
        log_file.flush()

        traceback.print_exc()



import multiprocessing
from Base.NonPersonalizedRecommender import Random, TopPop, GlobalEffects
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender

from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender

from MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from MatrixFactorization.IALSRecommender import IALSRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython

from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender


if __name__ == '__main__':


    log_file_name = "./result_experiments/run_test_recommender.txt"


    recommender_list = [
        Random,
        TopPop,
        GlobalEffects,
        UserKNNCFRecommender,
        ItemKNNCFRecommender,
        P3alphaRecommender,
        RP3betaRecommender,
        SLIM_BPR_Cython,
        SLIMElasticNetRecommender,
        MatrixFactorization_BPR_Cython,
        MatrixFactorization_FunkSVD_Cython,
        MatrixFactorization_AsySVD_Cython,
        PureSVDRecommender,
        IALSRecommender,
    ]

    log_file = open(log_file_name, "w")



    for recommender_class in recommender_list:
        run_recommender(recommender_class)
    #
    # pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    # resultList = pool.map(run_dataset, dataset_list)

