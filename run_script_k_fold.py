
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from SLIM_BPR.SLIM_BPR import SLIM_BPR
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from MatrixFactorization.Cython.MF_BPR_Cython import MF_BPR_Cython
#from MatrixFactorization.MatrixFactorization_RMSE import FunkSVD
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from SLIM_KarypisLab.SLIM_KarypisLab import SLIM_KarypisLab
from SLIM_ElasticNet.SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet
from Base.NonPersonalizedRecommender import TopPop, Random, GlobalEffects
#from LatentFactorSimilarity.SIMCFRecommender import SIMCFRecommender

from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
#from GraphBased.RP3beta_ML import RP3betaRecommender_ML

from MatrixFactorization.PureSVD import PureSVDRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from SLIM_ElasticNet.Cython.SLIM_Structure_Cython import SLIM_Structure_BPR_Cython

#from FW_Rating.Cython.FBSM_Rating_Cython import FBSM_Rating_Cython

from data.NetflixEnhanced.NetflixEnhancedReader import NetflixEnhancedReader
from data.Epinions.EpinionsReader import EpinionsReader
from data.Movielens_20m.Movielens20MReader import Movielens20MReader
from data.Movielens_10m.Movielens10MReader import Movielens10MReader
from data.Movielens_1m.Movielens1MReader import Movielens1MReader
from data.BookCrossing.BookCrossingReader import BookCrossingReader
from data.TheMoviesDataset.TheMoviesDatasetReader import TheMoviesDatasetReader
from data.AmazonReviewData.AmazonElectronics.AmazonElectronicsReader import AmazonElectronicsReader
from data.AmazonReviewData.AmazonAutomotive.AmazonAutomotiveReader import AmazonAutomotiveReader
from data.XingChallenge2016.XingChallenge2016Reader import XingChallenge2016Reader
from data.XingChallenge2017.XingChallenge2017Reader import XingChallenge2017Reader
from data.NetflixPrize.NetflixPrizeReader import NetflixPrizeReader
from data.ThirtyMusic.ThirtyMusicReader import ThirtyMusicReader

from data.URM_Dense_K_Cores import select_k_cores
import numpy as np

from data.DataSplitter import DataSplitter_ColdItems_WarmValidation, DataSplitter_Warm, DataSplitter_ColdItems_ColdValidation
from data.DataSplitter_k_fold import DataSplitter_Warm_k_fold, DataSplitter_ColdItems_k_fold

from Base.Evaluation.KFoldResultRepository import KFoldResultRepository

if __name__ == '__main__':


    dataSplitter = DataSplitter_Warm_k_fold(Movielens1MReader, allow_cold_users=True, ICM_to_load=None, apply_k_cores=1, force_new_split=False)
    #dataSplitter = DataSplitter_ColdItems_k_fold(XingChallenge2017Reader, ICM_to_load=None, apply_k_cores=1, force_new_split=False, forbid_new_split=False)

    ICM_name = "ICM_all"

    result_repo_alg1 = KFoldResultRepository(len(dataSplitter))
    result_repo_alg2 = KFoldResultRepository(len(dataSplitter))


    from Base.Evaluation.Evaluator import SequentialEvaluator

    for fold_index, (URM_train, URM_test) in dataSplitter:

        print("Processing fold {}".format(fold_index))

        evaluator = SequentialEvaluator(URM_test, [10])
        #ICM_train = dataSplitter.get_ICM(ICM_to_load = ICM_name)

        recommender = TopPop(URM_train)
        recommender.fit()
        results_run, results_run_string = evaluator.evaluateRecommender(recommender)

        print(results_run_string)
        result_repo_alg1.set_results_in_fold(fold_index, results_run[10])


        recommender = ItemKNNCFRecommender(URM_train)
        recommender.fit()
        results_run, results_run_string = evaluator.evaluateRecommender(recommender)

        print(results_run_string)
        result_repo_alg2.set_results_in_fold(fold_index, results_run[10])


    result_repo_alg1.run_significance_test(result_repo_alg2)