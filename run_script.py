

from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from MatrixFactorization.PureSVD import PureSVDRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

from data.Movielens10MReader import Movielens10MReader


if __name__ == '__main__':


    dataReader = Movielens10MReader()

    URM_train = dataReader.get_URM_train()
    URM_validation = dataReader.get_URM_validation()
    URM_test = dataReader.get_URM_test()


    recommender = PureSVDRecommender(URM_train)

    recommender.fit()


    from Base.Evaluation.Evaluator import SequentialEvaluator

    evaluator = SequentialEvaluator(URM_test, [5])

    results_run, results_run_string = evaluator.evaluateRecommender(recommender)

    print("Block:\n" + results_run_string)
