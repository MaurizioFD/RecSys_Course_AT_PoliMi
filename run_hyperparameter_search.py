#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.Recommender_import_list import *

import traceback

import os, multiprocessing
from functools import partial



from Data_manager.Movielens.Movielens1MReader import Movielens1MReader
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative, runHyperparameterSearch_Content, runHyperparameterSearch_Hybrid


def read_data_split_and_search():
    """
    This function provides a simple example on how to tune parameters of a given algorithm

    The BayesianSearch object will save:
        - A .txt file with all the cases explored and the recommendation quality
        - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
        - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
        - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
        - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    """



    dataReader = Movielens1MReader()
    dataset = dataReader.load_data()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(dataset.get_URM_all(), train_percentage = 0.80)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage = 0.80)

    output_folder_path = "result_experiments/"


    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)


    collaborative_algorithm_list = [
        Random,
        TopPop,
        P3alphaRecommender,
        RP3betaRecommender,
        ItemKNNCFRecommender,
        UserKNNCFRecommender,
        MatrixFactorization_BPR_Cython,
        MatrixFactorization_FunkSVD_Cython,
        PureSVDRecommender,
        SLIM_BPR_Cython,
        SLIMElasticNetRecommender
    ]




    from Evaluation.Evaluator import EvaluatorHoldout

    cutoff_list = [5, 10, 20]
    metric_to_optimize = "MAP"
    cutoff_to_optimize = 10

    n_cases = 10
    n_random_starts = int(n_cases/3)

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list = cutoff_list)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list = cutoff_list)


    runParameterSearch_Collaborative_partial = partial(runHyperparameterSearch_Collaborative,
                                                       URM_train = URM_train,
                                                       metric_to_optimize = metric_to_optimize,
                                                       cutoff_to_optimize = cutoff_to_optimize,
                                                       n_cases = n_cases,
                                                       n_random_starts = n_random_starts,
                                                       evaluator_validation_earlystopping = evaluator_validation,
                                                       evaluator_validation = evaluator_validation,
                                                       evaluator_test = evaluator_test,
                                                       output_folder_path = output_folder_path,
                                                       resume_from_saved = True,
                                                       similarity_type_list = ["cosine"],
                                                       parallelizeKNN = False)





    pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)

    #
    #
    # for recommender_class in collaborative_algorithm_list:
    #
    #     try:
    #
    #         runParameterSearch_Collaborative_partial(recommender_class)
    #
    #     except Exception as e:
    #
    #         print("On recommender {} Exception {}".format(recommender_class, str(e)))
    #         traceback.print_exc()
    #




    ################################################################################################
    ###### Content Baselines

    for ICM_name, ICM_object in dataset.get_loaded_ICM_dict().items():

        try:

            runHyperparameterSearch_Content(ItemKNNCBFRecommender,
                                        URM_train = URM_train,
                                        URM_train_last_test = URM_train + URM_validation,
                                        metric_to_optimize = metric_to_optimize,
                                        cutoff_to_optimize = cutoff_to_optimize,
                                        evaluator_validation = evaluator_validation,
                                        evaluator_test = evaluator_test,
                                        output_folder_path = output_folder_path,
                                        parallelizeKNN = True,
                                        allow_weighting = True,
                                        resume_from_saved = True,
                                        similarity_type_list = ["cosine"],
                                        ICM_name = ICM_name,
                                        ICM_object = ICM_object.copy(),
                                        n_cases = n_cases,
                                        n_random_starts = n_random_starts)

        except Exception as e:

            print("On CBF recommender for ICM {} Exception {}".format(ICM_name, str(e)))
            traceback.print_exc()


        try:

            runHyperparameterSearch_Hybrid(ItemKNN_CFCBF_Hybrid_Recommender,
                                        URM_train = URM_train,
                                        URM_train_last_test = URM_train + URM_validation,
                                        metric_to_optimize = metric_to_optimize,
                                        cutoff_to_optimize = cutoff_to_optimize,
                                        evaluator_validation = evaluator_validation,
                                        evaluator_test = evaluator_test,
                                        output_folder_path = output_folder_path,
                                        parallelizeKNN = True,
                                        allow_weighting = True,
                                        resume_from_saved = True,
                                        similarity_type_list = ["cosine"],
                                        ICM_name = ICM_name,
                                        ICM_object = ICM_object.copy(),
                                        n_cases = n_cases,
                                        n_random_starts = n_random_starts)


        except Exception as e:

            print("On recommender {} Exception {}".format(ItemKNN_CFCBF_Hybrid_Recommender, str(e)))
            traceback.print_exc()





if __name__ == '__main__':


    read_data_split_and_search()
