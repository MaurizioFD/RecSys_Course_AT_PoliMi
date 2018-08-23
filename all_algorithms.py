
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython
from MatrixFactorization.PureSVD import PureSVDRecommender

from Base.NonPersonalizedRecommender import TopPop, Random

from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

from data.Movielens10MReader import Movielens10MReader

import traceback


if __name__ == '__main__':


    dataReader = Movielens10MReader()

    URM_train = dataReader.get_URM_train()
    URM_validation = dataReader.get_URM_validation()
    URM_test = dataReader.get_URM_test()

    recommender_list = [
        Random(URM_train),
        TopPop(URM_train),
        ItemKNNCFRecommender(URM_train),
        UserKNNCFRecommender(URM_train),
        MatrixFactorization_BPR_Cython(URM_train),
        MatrixFactorization_FunkSVD_Cython(URM_train),
        PureSVDRecommender(URM_train),
        SLIM_BPR_Cython(URM_train),
        SLIMElasticNetRecommender(URM_train)
        ]

    from Base.Evaluation.Evaluator import SequentialEvaluator

    evaluator = SequentialEvaluator(URM_test, [5, 20], exclude_seen=True)



    logFile = open("result_all_algorithms.txt", "a")


    for recommender in recommender_list:

        try:

            print("Algorithm: {}".format(recommender.__class__))

            recommender.fit()

            results_run, results_run_string = evaluator.evaluateRecommender(recommender)

            print("Algorithm: {}, results: \n{}".format(recommender.__class__, results_run_string))
            logFile.write("Algorithm: {}, results: \n{}".format(recommender.__class__, results_run_string))
            logFile.flush()

        except Exception as e:
            traceback.print_exc()
            logFile.write("Algorithm: {} - Exception: {}\n".format(recommender.__class__, str(e)))
