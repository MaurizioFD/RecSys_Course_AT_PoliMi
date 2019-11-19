
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from MatrixFactorization.IALSRecommender import IALSRecommender

from Base.NonPersonalizedRecommender import TopPop, Random, GlobalEffects

from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender

from Data_manager.Movielens10M.Movielens10MReader import Movielens10MReader
from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out

import traceback, os


if __name__ == '__main__':


    dataset_object = Movielens10MReader()

    dataSplitter = DataSplitter_leave_k_out(dataset_object, k_out_value=2)

    dataSplitter.load_data()
    URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

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


    from Base.Evaluation.Evaluator import EvaluatorHoldout

    evaluator = EvaluatorHoldout(URM_test, [5, 20], exclude_seen=True)


    output_root_path = "./result_experiments/"

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)


    logFile = open(output_root_path + "result_all_algorithms.txt", "a")


    for recommender_class in recommender_list:

        try:

            print("Algorithm: {}".format(recommender_class))


            recommender = recommender_class(URM_train)
            recommender.fit()

            results_run, results_run_string = evaluator.evaluateRecommender(recommender)

            print("Algorithm: {}, results: \n{}".format(recommender.__class__, results_run_string))
            logFile.write("Algorithm: {}, results: \n{}\n".format(recommender.__class__, results_run_string))
            logFile.flush()

        except Exception as e:
            traceback.print_exc()
            logFile.write("Algorithm: {} - Exception: {}\n".format(recommender_class, str(e)))
            logFile.flush()
