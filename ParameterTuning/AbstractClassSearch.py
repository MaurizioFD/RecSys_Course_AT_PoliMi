#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/03/2018

@author: Maurizio Ferrari Dacrema
"""

from enum import Enum


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
    attributes_to_clone = {"epochs": 'epochs_best'}

    for external_attribute_name in attributes_to_clone:

        recommender_attribute_name = attributes_to_clone[external_attribute_name]

        if hasattr(recommender, recommender_attribute_name):
            paramether_dictionary[external_attribute_name] = getattr(recommender, recommender_attribute_name)

    return paramether_dictionary




def writeLog(string, logFile):

    print(string)

    if logFile is not None:
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
            self.evaluator_validation = EvaluatorWrapper(evaluator_validation)

        if evaluator_test is None:
            self.evaluator_test = None
        else:
            self.evaluator_test = EvaluatorWrapper(evaluator_test)


        if from_fit_params_to_saved_params_function is None:
            self.from_fit_params_to_saved_params_function = from_fit_params_to_saved_params_function_default
        else:
            self.from_fit_params_to_saved_params_function = from_fit_params_to_saved_params_function







    def search(self, recommender_constructor_dict,
               metric_to_optimize = "MAP",
               n_cases = None,
               output_folder_path = None,
               output_file_name_root = None,
               parallelize = False,
               save_model = "best"
               ):

        raise NotImplementedError("Function search not implementated for this class")
