#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/03/2018

@author: Maurizio Ferrari Dacrema
"""

from ParameterTuning.AbstractClassSearch import AbstractClassSearch, DictionaryKeys
from functools import partial
import traceback, pickle
import numpy as np

try:
    #from bayes_opt import BayesianOptimization
    from ParameterTuning.BayesianOptimization_master.bayes_opt.bayesian_optimization import BayesianOptimization
except ImportError as importError:
    print("Unable to load BayesianOptimization module. Please install it using 'pip install bayesian-optimization' "
          "or download it from 'https://github.com/fmfn/BayesianOptimization'")

    raise importError



def writeLog(string, logFile):

    print(string)

    if logFile!=None:
        logFile.write(string)
        logFile.flush()




class BayesianSearch(AbstractClassSearch):

    ALGORITHM_NAME = "BayesianSearch"

    """
    This class applies Bayesian parameter tuning using this package:
    https://github.com/fmfn/BayesianOptimization

    pip install bayesian-optimization
    """

    def __init__(self, recommender_class, evaluator_validation=None, evaluator_test = None):

        super(BayesianSearch, self).__init__(recommender_class,
                                             evaluator_validation= evaluator_validation, evaluator_test=evaluator_test)




    def search(self, dictionary, metric ="MAP", init_points = 5, n_cases = 30, output_root_path = None, parallelPoolSize = 2, parallelize = True,
               save_model = "best"):

        # Associate the params that will be returned by BayesianOpt object to those you want to save
        # E.g. with early stopping you know which is the optimal number of epochs only afterwards
        # but you might want to save it as well
        self.from_fit_params_to_saved_params = {}

        self.dictionary_input = dictionary.copy()


        hyperparamethers_range_dictionary = dictionary[DictionaryKeys.FIT_RANGE_KEYWORD_ARGS].copy()

        self.output_root_path = output_root_path
        self.logFile = open(self.output_root_path + "_BayesianSearch.txt", "a")
        self.save_model = save_model
        self.model_counter = 0

        self.categorical_mapper_dict_case_to_index = {}
        self.categorical_mapper_dict_index_to_case = {}

        # Transform range element in a list of two elements: min, max
        for key in hyperparamethers_range_dictionary.keys():

            # Get the extremes for every range
            current_range = hyperparamethers_range_dictionary[key]

            if type(current_range) is range:
                min_val = current_range.start
                max_val = current_range.stop

            elif type(current_range) is list:

                categorical_mapper_dict_case_to_index_current = {}
                categorical_mapper_dict_index_to_case_current = {}

                for current_single_case in current_range:
                    num_vaues = len(categorical_mapper_dict_case_to_index_current)
                    categorical_mapper_dict_case_to_index_current[current_single_case] = num_vaues
                    categorical_mapper_dict_index_to_case_current[num_vaues] = current_single_case

                num_vaues = len(categorical_mapper_dict_case_to_index_current)

                min_val = 0
                max_val = num_vaues-1

                self.categorical_mapper_dict_case_to_index[key] = categorical_mapper_dict_case_to_index_current.copy()
                self.categorical_mapper_dict_index_to_case[key] = categorical_mapper_dict_index_to_case_current.copy()

            else:
                raise TypeError("BayesianSearch: for every parameter a range may be specified either by a 'range' object or by a list."
                                "Provided object type for parameter '{}' was '{}'".format(key, type(current_range)))


            hyperparamethers_range_dictionary[key] = [min_val, max_val]



        self.runSingleCase_partial = partial(self.runSingleCase,
                                             dictionary = dictionary,
                                             metric = metric)






        self.bayesian_optimizer = BayesianOptimization(self.runSingleCase_partial, hyperparamethers_range_dictionary)

        self.best_solution_val = None
        self.best_solution_parameters = None
        #self.best_solution_object = None


        self.bayesian_optimizer.maximize(init_points=init_points, n_iter=n_cases, kappa=2)

        best_solution = self.bayesian_optimizer.res['max']

        self.best_solution_val = best_solution["max_val"]
        self.best_solution_parameters = best_solution["max_params"].copy()
        self.best_solution_parameters = self.parameter_bayesian_to_token(self.best_solution_parameters)
        self.best_solution_parameters = self.from_fit_params_to_saved_params[frozenset(self.best_solution_parameters.items())]


        writeLog("BayesianSearch: Best config is: Config {}, {} value is {:.4f}\n".format(
            self.best_solution_parameters, metric, self.best_solution_val), self.logFile)

        #
        #
        # if folderPath != None:
        #
        #     writeLog("BayesianSearch: Saving model in {}\n".format(folderPath), self.logFile)
        #     self.runSingleCase_param_parsed(dictionary, metric, self.best_solution_parameters, folderPath = folderPath, namePrefix = namePrefix)


        return self.best_solution_parameters.copy()


    #
    # def evaluate_on_test(self):
    #
    #     # Create an object of the same class of the imput
    #     # Passing the paramether as a dictionary
    #     recommender = self.recommender_class(*self.dictionary_input[DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS],
    #                                          **self.dictionary_input[DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS])
    #
    #
    #     recommender.fit(*self.dictionary_input[DictionaryKeys.FIT_POSITIONAL_ARGS],
    #                     **self.dictionary_input[DictionaryKeys.FIT_KEYWORD_ARGS],
    #                     **self.best_solution_parameters)
    #
    #
    #     result_dict = self.evaluator_test.evaluateRecommender(recommender, self.best_solution_parameters)
    #
    #
    #     writeLog("ParameterSearch: Best result evaluated on URM_test. Config: {} - results: {}\n".format(self.best_solution_parameters, result_dict), self.logFile)
    #
    #     return result_dict
    #



    def parameter_bayesian_to_token(self, paramether_dictionary):
        """
        The function takes the random values from BayesianSearch and transforms them in the corresponding categorical
        tokens
        :param paramether_dictionary:
        :return:
        """

        # Convert categorical values
        for key in paramether_dictionary.keys():

            if key in self.categorical_mapper_dict_index_to_case:

                float_value = paramether_dictionary[key]
                index = int(round(float_value, 0))

                categorical = self.categorical_mapper_dict_index_to_case[key][index]

                paramether_dictionary[key] = categorical


        return paramether_dictionary





    def runSingleCase(self, dictionary, metric, **paramether_dictionary_input):


        paramether_dictionary = self.parameter_bayesian_to_token(paramether_dictionary_input)

        return self.runSingleCase_param_parsed(dictionary, metric, paramether_dictionary)




    def runSingleCase_param_parsed(self, dictionary, metric, paramether_dictionary):


        try:

            # Create an object of the same class of the imput
            # Passing the paramether as a dictionary
            recommender = self.recommender_class(*dictionary[DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS],
                                                 **dictionary[DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS])


            print("BayesianSearch: Testing config: {}".format(paramether_dictionary))

            recommender.fit(*dictionary[DictionaryKeys.FIT_POSITIONAL_ARGS],
                            **dictionary[DictionaryKeys.FIT_KEYWORD_ARGS],
                            **paramether_dictionary)

            #return recommender.evaluateRecommendations(self.URM_validation, at=5, mode="sequential")
            result_dict, _  = self.evaluator_validation.evaluateRecommender(recommender, paramether_dictionary)
            result_dict = result_dict[list(result_dict.keys())[0]]


            paramether_dictionary_to_save = self.from_fit_params_to_saved_params_function(recommender, paramether_dictionary)

            self.from_fit_params_to_saved_params[frozenset(paramether_dictionary.items())] = paramether_dictionary_to_save

            self.model_counter += 1


            # Always save best model separately
            if self.save_model == "all":
                print(self.ALGORITHM_NAME + ": Saving model in {}\n".format(self.output_root_path))
                recommender.saveModel(self.output_root_path, file_name= "_model_{}".format(self.model_counter))

                pickle.dump(paramether_dictionary_to_save.copy(),
                            open(self.output_root_path + "_parameters_{}".format(self.model_counter), "wb"),
                            protocol=pickle.HIGHEST_PROTOCOL)


            if self.best_solution_val == None or self.best_solution_val<result_dict[metric]:

                writeLog("BayesianSearch: New best config found. Config: {} - results: {}\n".format(paramether_dictionary_to_save, result_dict), self.logFile)

                pickle.dump(paramether_dictionary_to_save.copy(),
                            open(self.output_root_path + "_best_parameters", "wb"),
                            protocol=pickle.HIGHEST_PROTOCOL)

                pickle.dump(result_dict.copy(),
                            open(self.output_root_path + "_best_result_validation", "wb"),
                            protocol=pickle.HIGHEST_PROTOCOL)

                self.best_solution_val = result_dict[metric]
                self.best_solution_parameters = paramether_dictionary_to_save.copy()
                #self.best_solution_object = recommender

                if self.save_model != "no":
                    print("BayesianSearch: Saving model in {}\n".format(self.output_root_path))
                    recommender.saveModel(self.output_root_path, file_name = "_best_model")

                if self.evaluator_test is not None:
                    self.evaluate_on_test()

            else:
                writeLog("BayesianSearch: Config is suboptimal. Config: {} - results: {}\n".format(paramether_dictionary_to_save, result_dict), self.logFile)


            return result_dict[metric]


        except Exception as e:

            writeLog("BayesianSearch: Testing config: {} - Exception {}\n".format(paramether_dictionary, str(e)), self.logFile)
            traceback.print_exc()

            return - np.inf


















def function_interface(x, y):

    return -x ** 2 - (y - 1) ** 2 + 1



if __name__ == '__main__':

    # Lets find the maximum of a simple quadratic function of two variables
    # We create the bayes_opt object and pass the function to be maximized
    # together with the parameters names and their bounds.
    bo = BayesianOptimization(function_interface,
                              {'x': (-4, 4), 'y': (-3, 3)})

    # One of the things we can do with this object is pass points
    # which we want the algorithm to probe. A dictionary with the
    # parameters names and a list of values to include in the search
    # must be given.
    bo.explore({'x': [-1, 3], 'y': [-2, 2]})

    # Additionally, if we have any prior knowledge of the behaviour of
    # the target function (even if not totally accurate) we can also
    # tell that to the optimizer.
    # Here we pass a dictionary with 'target' and parameter names as keys and a
    # list of corresponding values
    bo.initialize(
        {
            'target': [-1, -1],
            'x': [1, 1],
            'y': [0, 2]
        }
    )

    # Once we are satisfied with the initialization conditions
    # we let the algorithm do its magic by calling the maximize()
    # method.
    bo.maximize(init_points=5, n_iter=15, kappa=2)

    # The output values can be accessed with self.res
    print(bo.res['max'])

    # If we are not satisfied with the current results we can pickup from
    # where we left, maybe pass some more exploration points to the algorithm
    # change any parameters we may choose, and the let it run again.
    bo.explore({'x': [0.6], 'y': [-0.23]})

    # Making changes to the gaussian process can impact the algorithm
    # dramatically.
    gp_params = {'kernel': None,
                 'alpha': 1e-5}

    # Run it again with different acquisition function
    bo.maximize(n_iter=5, acq='ei', **gp_params)

    # Finally, we take a look at the final results.
    print(bo.res['max'])
    print(bo.res['all'])