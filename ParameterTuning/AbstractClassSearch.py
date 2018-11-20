#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/03/2018

@author: Maurizio Ferrari Dacrema
"""

from enum import Enum
import traceback, pickle

import numpy as np
from Base.Evaluation.Evaluator import SequentialEvaluator



class EvaluatorWrapper(object):

    def __init__(self, evaluator_object):

        self.evaluator_object = evaluator_object


    def evaluateRecommender(self, recommender_object, paramether_dictionary = None):

        return self.evaluator_object.evaluateRecommender(recommender_object)






class DictionaryKeys(Enum):
    # Fields to be filled by caller
    # Dictionary of paramethers needed by the constructor
    CONSTRUCTOR_POSITIONAL_ARGS = 'constructor_positional_args'
    CONSTRUCTOR_KEYWORD_ARGS = 'constructor_keyword_args'

    # List containing all positional arguments needed by the fit function
    FIT_POSITIONAL_ARGS = 'fit_positional_args'
    FIT_KEYWORD_ARGS = 'fit_keyword_args'

    # Contains the dictionary of all keyword args to use for validation
    # With the respectives range
    FIT_RANGE_KEYWORD_ARGS = 'fit_range_keyword_args'

    # Label to be written on log
    LOG_LABEL = 'log_label'



def from_fit_params_to_saved_params_function_default(recommender, paramether_dictionary):

    paramether_dictionary = paramether_dictionary.copy()

    # Attributes that might be determined through early stopping
    # Name in param_dictionary: name in object
    attributes_to_clone = {"epochs": 'epochs_best', "max_epochs": 'epochs_best'}

    for external_attribute_name in attributes_to_clone:

        recommender_attribute_name = attributes_to_clone[external_attribute_name]

        if hasattr(recommender, recommender_attribute_name):
            paramether_dictionary[external_attribute_name] = getattr(recommender, recommender_attribute_name)

    return paramether_dictionary




def writeLog(string, logFile):

    print(string)

    if logFile!=None:
        logFile.write(string)
        logFile.flush()





class AbstractClassSearch(object):

    ALGORITHM_NAME = "AbstractClassSearch"

    def __init__(self, recommender_class,
                 evaluator_validation = None, evaluator_test = None,
                 from_fit_params_to_saved_params_function=None):


        super(AbstractClassSearch, self).__init__()

        self.recommender_class = recommender_class

        self.results_test_best = {}
        self.paramether_dictionary_best = {}

        if evaluator_validation is None:
            raise ValueError("AbstractClassSearch: evaluator_validation must be provided")
        else:
            self.evaluator_validation = evaluator_validation

        if evaluator_test is None:
            self.evaluator_test = None
        else:
            self.evaluator_test = evaluator_test


        if from_fit_params_to_saved_params_function is None:
            self.from_fit_params_to_saved_params_function = from_fit_params_to_saved_params_function_default
        else:
            self.from_fit_params_to_saved_params_function = from_fit_params_to_saved_params_function







    def search(self, dictionary_input, metric ="map", logFile = None, parallelPoolSize = 2, parallelize = True):
        raise NotImplementedError("Function search not implementated for this class")








    def runSingleCase(self, paramether_dictionary_to_evaluate, metric):


        try:

            # Create an object of the same class of the imput
            # Passing the paramether as a dictionary
            recommender = self.recommender_class(*self.dictionary_input[DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS],
                                                 **self.dictionary_input[DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS])


            print(self.ALGORITHM_NAME + ": Testing config: {}".format(paramether_dictionary_to_evaluate))

            recommender.fit(*self.dictionary_input[DictionaryKeys.FIT_POSITIONAL_ARGS],
                            **self.dictionary_input[DictionaryKeys.FIT_KEYWORD_ARGS],
                            **paramether_dictionary_to_evaluate)

            #result_dict = self.evaluator_validation(recommender, self.URM_validation, paramether_dictionary_to_evaluate)

            result_dict, _ = self.evaluator_validation.evaluateRecommender(self, paramether_dictionary_to_evaluate)
            result_dict = result_dict[list(result_dict.keys())[0]]


            paramether_dictionary_to_save = self.from_fit_params_to_saved_params_function(recommender, paramether_dictionary_to_evaluate)

            self.from_fit_params_to_saved_params[frozenset(paramether_dictionary_to_evaluate.items())] = paramether_dictionary_to_save




            if self.best_solution_val == None or self.best_solution_val<result_dict[metric]:

                writeLog(self.ALGORITHM_NAME + ": New best config found. Config: {} - results: {}\n".format(paramether_dictionary_to_save, result_dict), self.logFile)

                pickle.dump(paramether_dictionary_to_save.copy(),
                            open(self.output_root_path + "_best_parameters", "wb"),
                            protocol=pickle.HIGHEST_PROTOCOL)

                self.best_solution_val = result_dict[metric]
                self.best_solution_parameters = paramether_dictionary_to_save.copy()
                #self.best_solution_object = recommender

                if self.save_best_model:
                    print(self.ALGORITHM_NAME + ": Saving model in {}\n".format(self.output_root_path))
                    recommender.saveModel(self.output_root_path, file_name = self.recommender_class.RECOMMENDER_NAME + "_best_model")

                if self.evaluator_test is not None:
                    self.evaluate_on_test()

            else:
                writeLog(self.ALGORITHM_NAME + ": Config is suboptimal. Config: {} - results: {}\n".format(paramether_dictionary_to_save, result_dict), self.logFile)


            return result_dict[metric]


        except Exception as e:

            writeLog(self.ALGORITHM_NAME + ": Testing config: {} - Exception {}\n".format(paramether_dictionary_to_evaluate, str(e)), self.logFile)
            traceback.print_exc()

            return - np.inf





    def evaluate_on_test(self):

        # Create an object of the same class of the imput
        # Passing the paramether as a dictionary
        recommender = self.recommender_class(*self.dictionary_input[DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS],
                                             **self.dictionary_input[DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS])

        if self.save_model != "no":
            recommender.loadModel(self.output_root_path, file_name = "_best_model")

        else:
            recommender.fit(*self.dictionary_input[DictionaryKeys.FIT_POSITIONAL_ARGS],
                            **self.dictionary_input[DictionaryKeys.FIT_KEYWORD_ARGS],
                            **self.best_solution_parameters)


        result_dict, result_string = self.evaluator_test.evaluateRecommender(recommender, self.best_solution_parameters)
        result_dict = result_dict[list(result_dict.keys())[0]]

        pickle.dump(result_dict.copy(),
                    open(self.output_root_path + "_best_result_test", "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        writeLog(self.ALGORITHM_NAME + ": Best result evaluated on URM_test. Config: {} - results: {}\n".format(self.best_solution_parameters, result_string), self.logFile)

        return result_dict















