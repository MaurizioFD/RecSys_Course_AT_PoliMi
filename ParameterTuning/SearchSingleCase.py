#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/12/18

@author: Emanuele Chioso, Maurizio Ferrari Dacrema
"""

import os
from Base.DataIO import DataIO
from ParameterTuning.SearchAbstractClass import writeLog
from ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt



class SearchSingleCase(SearchBayesianSkopt):

    ALGORITHM_NAME = "SearchSingleCase"

    def __init__(self, recommender_class, evaluator_validation = None, evaluator_test = None):

        super(SearchSingleCase, self).__init__(recommender_class,
                                               evaluator_validation= evaluator_validation,
                                               evaluator_test=evaluator_test)



    def search(self, recommender_input_args,
               fit_parameters_values = None,
               metric_to_optimize = "MAP",
               output_folder_path = None,
               output_file_name_root = None,
               save_metadata = True,
               recommender_input_args_last_test = None,
               ):


        assert fit_parameters_values is not None, "{}: fit_parameters_values must contain a dictionary".format(self.ALGORITHM_NAME)

        self.recommender_input_args = recommender_input_args
        self.recommender_input_args_last_test = recommender_input_args_last_test
        self.metric_to_optimize = metric_to_optimize
        self.output_folder_path = output_folder_path
        self.output_file_name_root = output_file_name_root
        self.best_solution_val = None

        # If directory does not exist, create
        if not os.path.exists(self.output_folder_path):
            os.makedirs(self.output_folder_path)


        self.log_file = open(self.output_folder_path + self.output_file_name_root + "_{}.txt".format(self.ALGORITHM_NAME), "a")

        self.save_metadata = save_metadata
        self.n_calls = 1
        self.model_counter = 0
        self.best_solution_counter = 0
        self.save_model = "best"

        self.hyperparams_names = {}
        self.hyperparams_single_value = {}

        # In case of earlystopping the best_solution_parameters will contain also the number of epochs
        self.best_solution_parameters = fit_parameters_values.copy()


        if self.save_metadata:
            self._init_metadata_dict()
            self.dataIO = DataIO(folder_path = self.output_folder_path)


        self._objective_function(fit_parameters_values)

        writeLog("{}: Search complete. Best config is {}: {}\n".format(self.ALGORITHM_NAME,
                                                                       self.best_solution_counter,
                                                                       self.best_solution_parameters), self.log_file)


        if self.recommender_input_args_last_test is not None:
            self._evaluate_on_test_with_data_last()



