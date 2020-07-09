#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/2018

@author: Maurizio Ferrari Dacrema
"""

import traceback, os, shutil


from Base.Evaluation.Evaluator import EvaluatorHoldout, EvaluatorNegativeItemSample
from Data_manager.Movielens.Movielens1MReader import Movielens1MReader
from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping


def write_log_string(log_file, string):
    log_file.write(string)
    log_file.flush()

def _get_recommender_instance(recommender_class, URM_train, ICM_train):
    
    if recommender_class is ItemKNNCBFRecommender:
        recommender_object = recommender_class(URM_train, ICM_train)
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
        dataSplitter.load_data(save_folder_path= output_folder_path + dataset_object._get_dataset_name() + "_data/")

        URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()
        ICM_name = dataSplitter.get_all_available_ICM_names()[0]
        ICM_train = dataSplitter.get_ICM_from_name(ICM_name)
        
        write_log_string(log_file, "On Recommender {}\n".format(recommender_class))


        recommender_object = _get_recommender_instance(recommender_class, URM_train, ICM_train)


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



        recommender_object.save_model(temp_save_file_folder, file_name="temp_model")

        write_log_string(log_file, "save_model OK, ")


        recommender_object = _get_recommender_instance(recommender_class, URM_train, ICM_train)

        recommender_object.load_model(temp_save_file_folder, file_name= "temp_model")

        evaluator = EvaluatorHoldout(URM_test, [5], exclude_seen=True)
        _, results_run_string_2 = evaluator.evaluateRecommender(recommender_object)

        write_log_string(log_file, "load_model OK, ")



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
from EASE_R.EASE_R_Recommender import EASE_R_Recommender

from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender


if __name__ == '__main__':

    output_folder_path = "./result_experiments/rec_test/"
    log_file_name = "run_test_recommender.txt"


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

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    log_file = open(output_folder_path + log_file_name, "w")



    for recommender_class in recommender_list:
        run_recommender(recommender_class)
    #
    # pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    # resultList = pool.map(run_dataset, dataset_list)

