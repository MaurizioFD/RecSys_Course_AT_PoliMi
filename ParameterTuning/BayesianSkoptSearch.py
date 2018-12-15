#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/12/18

@author: Emanuele Chioso, Maurizio Ferrari Dacrema
"""

import pickle

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

from ParameterTuning.AbstractClassSearch import AbstractClassSearch, DictionaryKeys, writeLog



class BayesianSkoptSearch(AbstractClassSearch):

    ALGORITHM_NAME = "BayesianSkoptSearch"


    def __init__(self, recommender_class, evaluator_validation=None, evaluator_test = None):


        super(BayesianSkoptSearch, self).__init__(recommender_class,
                                                  evaluator_validation= evaluator_validation,
                                                  evaluator_test=evaluator_test)



    def set_skopt_params(self, n_calls = 70,
                         n_random_starts = 20,
                         n_points = 10000,
                         n_jobs = 1,
                         noise = 'gaussian',
                         acq_func = 'gp_hedge',
                         acq_optimizer = 'auto',
                         random_state = None,
                         verbose = True,
                         n_restarts_optimizer = 10,
                         xi = 0.01,
                         kappa = 1.96,
                         x0 = None,
                         y0 = None):
        """
        wrapper to change the params of the bayesian optimizator.
        for further details:
        https://scikit-optimize.github.io/#skopt.gp_minimize

        """
        self.n_point = n_points
        self.n_calls = n_calls
        self.n_random_starts = n_random_starts
        self.n_jobs = n_jobs
        self.acq_func = acq_func
        self.acq_optimizer = acq_optimizer
        self.random_state = random_state
        self.n_restarts_optimizer = n_restarts_optimizer
        self.verbose = verbose
        self.xi = xi
        self.kappa = kappa
        self.noise = noise
        self.x0 = x0
        self.y0 = y0



    def search(self, recommender_constructor_dict,
               metric_to_optimize = "MAP",
               n_cases = 20,
               n_random_starts = 5,
               output_folder_path = None,
               output_file_name_root = None,
               parallelize = False,
               save_model = "best"
               ):

        assert save_model in ["no", "all", "best"], "BayesianSkoptSearch: parameter save_model must be in '['no', 'all', 'best']', provided was '{}'.".format(save_model)
        self.save_model = save_model



        self.set_skopt_params()    ### default parameters are set here

        # Associate the params that will be returned by BayesianOpt object to those you want to save
        # E.g. with early stopping you know which is the optimal number of epochs only afterwards
        # but you might want to save it as well
        self.from_fit_params_to_saved_params = {}

        self.recommender_constructor_dict = recommender_constructor_dict.copy()
        self.metric_to_optimize = metric_to_optimize
        self.output_root_path = output_folder_path
        self.output_file_name_root = output_file_name_root
        self.n_random_starts = n_random_starts
        self.n_calls = n_cases


        self.log_file = open(self.output_root_path + self.output_file_name_root + "_BayesianSkoptSearch.txt", "a")
        self.model_counter = 0
        self.best_solution_val = None

        if parallelize:
            self.n_jobs = parallelize



        self.hyperparams = dict()
        self.hyperparams_names = list()
        self.hyperparams_values = list()
        self.hyperparams_single_value = dict()

        skopt_types = [Real, Integer, Categorical]

        for name, hyperparam in self.recommender_constructor_dict[DictionaryKeys.FIT_RANGE_KEYWORD_ARGS].items():

            if any(isinstance(hyperparam, sko_type) for sko_type in skopt_types):
                self.hyperparams_names.append(name)
                self.hyperparams_values.append(hyperparam)
                self.hyperparams[name] = hyperparam

            elif(isinstance(hyperparam, str) or isinstance(hyperparam, int) or isinstance(hyperparam, bool)):
                self.hyperparams_single_value[name] = hyperparam

            else:
                raise ValueError("BayesianSkoptSearch: Unexpected paramether type:"+" "+str(name)+" "+str(hyperparam))



        self.result = gp_minimize(self.__objective_function,
                                  self.hyperparams_values,
                                  base_estimator=None,
                                  n_calls=self.n_calls,
                                  n_random_starts=self.n_random_starts,
                                  acq_func=self.acq_func,
                                  acq_optimizer=self.acq_optimizer,
                                  x0=self.x0,
                                  y0=self.y0,
                                  random_state=self.random_state,
                                  verbose=self.verbose,
                                  callback=None,
                                  n_points=self.n_point,
                                  n_restarts_optimizer=self.n_restarts_optimizer,
                                  xi=self.xi,
                                  kappa=self.kappa,
                                  noise=self.noise,
                                  n_jobs=self.n_jobs)



    def __evaluate(self, current_fit_parameters, evaluator):


        # Construct a new recommender instance
        recommender_instance = self.recommender_class(*self.recommender_constructor_dict[DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS],
                                                      **self.recommender_constructor_dict[DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS])


        print("BayesianSkoptSearch: Testing config:", current_fit_parameters)


        recommender_instance.fit(*self.recommender_constructor_dict[DictionaryKeys.FIT_POSITIONAL_ARGS],
                                **self.recommender_constructor_dict[DictionaryKeys.FIT_KEYWORD_ARGS],
                                **current_fit_parameters,
                                **self.hyperparams_single_value)

        # Evaluate recommender and get results for the first cutoff
        result_dict, result_string = evaluator.evaluateRecommender(recommender_instance, self.recommender_constructor_dict)
        result_dict = result_dict[list(result_dict.keys())[0]]

        return result_dict, result_string, recommender_instance







    def __evaluate_on_test(self, recommender_instance):

        # Evaluate recommender and get results for the first cutoff
        result_dict, result_string = self.evaluator_test.evaluateRecommender(recommender_instance, self.recommender_constructor_dict)
        result_dict = result_dict[list(result_dict.keys())[0]]

        writeLog(self.ALGORITHM_NAME + ": Best result evaluated on URM_test. Config: {} - results:\n{}\n".format(self.best_solution_parameters, result_string), self.log_file)

        return result_dict




    def __objective_function(self, current_fit_parameters_values):


        current_fit_parameters = dict(zip(self.hyperparams_names, current_fit_parameters_values))


        result_dict, _, recommender_instance = self.__evaluate(current_fit_parameters, self.evaluator_validation)

        current_result = - result_dict[self.metric_to_optimize]


        paramether_dictionary_to_save = self.from_fit_params_to_saved_params_function(recommender_instance, current_fit_parameters)

        self.from_fit_params_to_saved_params[frozenset(current_fit_parameters.items())] = paramether_dictionary_to_save





        # Always save best model separately
        if self.save_model == "all":

            print(self.ALGORITHM_NAME + ": Saving model in {}\n".format(self.output_root_path + self.output_file_name_root))

            recommender_instance.saveModel(self.output_root_path, file_name = self.output_file_name_root + "_model_{}".format(self.model_counter))

            pickle.dump(paramether_dictionary_to_save.copy(),
                        open(self.output_root_path + self.output_file_name_root + "_parameters_{}".format(self.model_counter), "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)



        if self.best_solution_val == None or self.best_solution_val < result_dict[self.metric_to_optimize]:

            writeLog("BayesianSearch: New best config found. Config {}: {} - results: {}\n".format(self.model_counter, paramether_dictionary_to_save, result_dict), self.log_file)

            pickle.dump(paramether_dictionary_to_save.copy(),
                        open(self.output_root_path + self.output_file_name_root + "_best_parameters", "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)

            pickle.dump(result_dict.copy(),
                        open(self.output_root_path + self.output_file_name_root + "_best_result_validation", "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)

            self.best_solution_val = result_dict[self.metric_to_optimize]
            self.best_solution_parameters = paramether_dictionary_to_save.copy()

            if self.save_model != "no":
                print("BayesianSearch: Saving model in {}\n".format(self.output_root_path + self.output_file_name_root))
                recommender_instance.saveModel(self.output_root_path, file_name = self.output_file_name_root + "_best_model")


            if self.evaluator_test is not None:
                result_dict_test = self.__evaluate_on_test(recommender_instance)

                pickle.dump(result_dict_test,
                            open(self.output_root_path + self.output_file_name_root + "_best_result_test", "wb"),
                            protocol=pickle.HIGHEST_PROTOCOL)


        else:
            writeLog("BayesianSearch: Config {} is suboptimal. Config: {} - results: {}\n".format(self.model_counter, paramether_dictionary_to_save, result_dict), self.log_file)



        self.model_counter += 1


        return current_result
