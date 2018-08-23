
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from MatrixFactorization.Cython.MF_BPR_Cython import MF_BPR_Cython
from data.Movielens10MReader import Movielens10MReader


def run_SLIM():

    dataReader = Movielens10MReader()

    URM_train = dataReader.get_URM_train()
    URM_test = dataReader.get_URM_test()

    recommender = SLIM_BPR_Cython(URM_train, recompile_cython=False, positive_threshold=4, sparse_weights=True)
    #recommender = MF_BPR_Cython(URM_train, recompile_cython=False, positive_threshold=4)

    logFile = open("Result_log.txt", "a")


    recommender.fit(epochs=2, validate_every_N_epochs=1, URM_test=URM_test,
                    logFile=logFile, batch_size=1, sgd_mode='rmsprop', learning_rate=1e-4)


    results_run = recommender.evaluateRecommendations(URM_test, at=5)
    print(results_run)


run_SLIM()