#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/09/17

@author: Maurizio Ferrari Dacrema
"""

from Base.Recommender import Recommender
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Base.Evaluation.Evaluator import SequentialEvaluator

import subprocess
import os, sys
import time, pickle
import numpy as np




class MatrixFactorization_Cython(Recommender, Incremental_Training_Early_Stopping):

    RECOMMENDER_NAME = "MatrixFactorization_Cython_Recommender"


    def __init__(self, URM_train, positive_threshold=4, URM_validation = None, recompile_cython = False, algorithm = "MF_BPR"):


        super(MatrixFactorization_Cython, self).__init__()


        self.URM_train = URM_train
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]
        self.normalize = False

        self.algorithm = algorithm

        self.positive_threshold = positive_threshold

        if URM_validation is not None:
            self.URM_validation = URM_validation.copy()
        else:
            self.URM_validation = None

        self.compute_item_score = self.compute_score_MF


        if recompile_cython:
            print("Compiling in Cython")
            self.runCompilationScript()
            print("Compilation Complete")



    def compute_score_MF(self, user_id):

        scores_array = np.dot(self.W[user_id], self.H.T)

        return scores_array




    def fit(self, epochs=300, batch_size = 1000, num_factors=10,
            learning_rate = 0.01, sgd_mode='sgd', user_reg = 0.0, positive_reg = 0.0, negative_reg = 0.0,
            stop_on_validation = False, lower_validatons_allowed = 5, validation_metric = "MAP",
            evaluator_object = None, validation_every_n = 5):



        self.num_factors = num_factors
        self.sgd_mode = sgd_mode
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if evaluator_object is None and stop_on_validation:
            evaluator_object = SequentialEvaluator(self.URM_validation, [5])


        # Import compiled module
        from MatrixFactorization.Cython.MatrixFactorization_Cython_Epoch import MatrixFactorization_Cython_Epoch


        if self.algorithm == "FUNK_SVD":


            self.cythonEpoch = MatrixFactorization_Cython_Epoch(self.URM_train,
                                                     algorithm = self.algorithm,
                                                     n_factors = self.num_factors,
                                                     learning_rate = learning_rate,
                                                     batch_size = 1,
                                                     sgd_mode = sgd_mode,
                                                     user_reg = user_reg,
                                                     positive_reg = positive_reg,
                                                     negative_reg = 0.0)

        elif self.algorithm == "ASY_SVD":


            self.cythonEpoch = MatrixFactorization_Cython_Epoch(self.URM_train,
                                                     algorithm = self.algorithm,
                                                     n_factors = self.num_factors,
                                                     learning_rate = learning_rate,
                                                     batch_size = 1,
                                                     sgd_mode = sgd_mode,
                                                     user_reg = user_reg,
                                                     positive_reg = positive_reg,
                                                     negative_reg = 0.0)

        elif self.algorithm == "MF_BPR":

            # Select only positive interactions
            URM_train_positive = self.URM_train.copy()

            URM_train_positive.data = URM_train_positive.data >= self.positive_threshold
            URM_train_positive.eliminate_zeros()

            assert URM_train_positive.nnz > 0, "MatrixFactorization_Cython: URM_train_positive is empty, positive threshold is too high"

            self.cythonEpoch = MatrixFactorization_Cython_Epoch(URM_train_positive,
                                                     algorithm = self.algorithm,
                                                     n_factors = self.num_factors,
                                                     learning_rate=learning_rate,
                                                     batch_size=1,
                                                     sgd_mode = sgd_mode,
                                                     user_reg=user_reg,
                                                     positive_reg=positive_reg,
                                                     negative_reg=negative_reg)






        self._train_with_early_stopping(epochs, validation_every_n, stop_on_validation,
                                    validation_metric, lower_validatons_allowed, evaluator_object,
                                    algorithm_name = self.algorithm)





        self.W = self.W_best
        self.H = self.H_best

        sys.stdout.flush()






    def _initialize_incremental_model(self):

        self.W_incremental = self.cythonEpoch.get_W()
        self.W_best = self.W_incremental.copy()

        self.H_incremental = self.cythonEpoch.get_H()
        self.H_best = self.H_incremental.copy()



    def _update_incremental_model(self):

        self.W_incremental = self.cythonEpoch.get_W()
        self.H_incremental = self.cythonEpoch.get_H()

        self.W = self.W_incremental
        self.H = self.H_incremental


    def _update_best_model(self):

        self.W_best = self.W_incremental.copy()
        self.H_best = self.H_incremental.copy()



    def _run_epoch(self, num_epoch):
       self.cythonEpoch.epochIteration_Cython()


























    def runCompilationScript(self):

        # Run compile script setting the working directory to ensure the compiled file are contained in the
        # appropriate subfolder and not the project root

        compiledModuleSubfolder = "/MatrixFactorization/Cython"
        fileToCompile_list = ['MatrixFactorization_Cython_Epoch.pyx']

        for fileToCompile in fileToCompile_list:

            command = ['python',
                       'compileCython.py',
                       fileToCompile,
                       'build_ext',
                       '--inplace'
                       ]


            output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)

            try:

                command = ['cython',
                           fileToCompile,
                           '-a'
                           ]

                output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)

            except:
                pass


        print("Compiled module saved in subfolder: {}".format(compiledModuleSubfolder))

        # Command to run compilation script
        # python compileCython.py MatrixFactorization_Cython_Epoch.pyx build_ext --inplace

        # Command to generate html report
        # cython -a MatrixFactorization_Cython_Epoch.pyx






    def writeCurrentConfig(self, currentEpoch, results_run, logFile):

        current_config = {'learn_rate': self.learning_rate,
                          'num_factors': self.num_factors,
                          'batch_size': 1,
                          'epoch': currentEpoch}

        print("Test case: {}\nResults {}\n".format(current_config, results_run))

        sys.stdout.flush()

        if (logFile != None):
            logFile.write("Test case: {}, Results {}\n".format(current_config, results_run))
            logFile.flush()






    def saveModel(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        dictionary_to_save = {"W": self.W,
                              "H": self.H}


        pickle.dump(dictionary_to_save,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)


        print("{}: Saving complete")







class MatrixFactorization_BPR_Cython(MatrixFactorization_Cython):
    """
    Subclas allowing only for MF BPR
    """

    RECOMMENDER_NAME = "MatrixFactorization_BPR_Cython_Recommender"

    def __init__(self, *pos_args, **key_args):
        super(MatrixFactorization_BPR_Cython, self).__init__(*pos_args, algorithm="MF_BPR", **key_args)

    def fit(self, **key_args):
        super(MatrixFactorization_BPR_Cython, self).fit(**key_args)





class MatrixFactorization_FunkSVD_Cython(MatrixFactorization_Cython):
    """
    Subclas allowing only for FunkSVD
    """

    RECOMMENDER_NAME = "MatrixFactorization_FunkSVD_Cython_Recommender"

    def __init__(self, *pos_args, **key_args):
        super(MatrixFactorization_FunkSVD_Cython, self).__init__(*pos_args, algorithm="FUNK_SVD", **key_args)


    def fit(self, **key_args):

        if "reg" in key_args:
            key_args["positive_reg"] = key_args["reg"]
            del key_args["reg"]

        super(MatrixFactorization_FunkSVD_Cython, self).fit(**key_args)




class MatrixFactorization_AsySVD_Cython(MatrixFactorization_Cython):
    """
    Subclas allowing only for AsySVD
    """

    RECOMMENDER_NAME = "MatrixFactorization_AsySVD_Cython_Recommender"

    def __init__(self, *pos_args, **key_args):
        super(MatrixFactorization_AsySVD_Cython, self).__init__(*pos_args, algorithm="ASY_SVD", **key_args)

    def fit(self, **key_args):
        super(MatrixFactorization_AsySVD_Cython, self).fit(**key_args)
