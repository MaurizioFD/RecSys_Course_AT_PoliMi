#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/03/2018

@author: Maurizio Ferrari Dacrema
"""

import time, os, traceback
import pandas as pd
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
import numpy as np
from Recommenders.DataIO import DataIO
from Evaluation.Evaluator import get_result_string_df
from numpy.core._exceptions import _ArrayMemoryError

MEMORY_ERROR_EXCEPTION_TUPLE = (_ArrayMemoryError, MemoryError)

try:
    from tensorflow.python.framework.errors_impl import ResourceExhaustedError, InternalError, UnknownError
    MEMORY_ERROR_EXCEPTION_TUPLE += (ResourceExhaustedError, InternalError, UnknownError,)
except ImportError:
    print("Tensorflow is not available")



def create_result_multiindex_dataframe(n_cases, result_df):
    # The dataframe will have the case number and cutoff as index, the metric name as column

    cases_cutoff_multiindex = pd.MultiIndex.from_product([np.arange(n_cases), result_df.index])
    cases_cutoff_multiindex.set_names("cutoff", level=1, inplace=True)

    return pd.DataFrame(index=cases_cutoff_multiindex, columns=result_df.keys())


def add_result_to_multiindex_dataframe(destination_result_df, new_result_df, position):
    # Simple way to set a multiindex slice with another dataframe. Other solutions exist but better safe than sorry

    for index in new_result_df.index:
        destination_result_df.loc[position, index] = new_result_df.loc[index].copy()



class SearchInputRecommenderArgs(object):


    def __init__(self,
                   # Dictionary of hyperparameters needed by the constructor
                   CONSTRUCTOR_POSITIONAL_ARGS = None,
                   CONSTRUCTOR_KEYWORD_ARGS = None,

                   # List containing all positional arguments needed by the fit function
                   FIT_POSITIONAL_ARGS = None,
                   FIT_KEYWORD_ARGS = None,

                   # Dictionary containing the earlystopping keyword arguments
                   EARLYSTOPPING_KEYWORD_ARGS = None
                   ):


          super(SearchInputRecommenderArgs, self).__init__()

          if CONSTRUCTOR_POSITIONAL_ARGS is None:
              CONSTRUCTOR_POSITIONAL_ARGS = []

          if CONSTRUCTOR_KEYWORD_ARGS is None:
              CONSTRUCTOR_KEYWORD_ARGS = {}

          if FIT_POSITIONAL_ARGS is None:
              FIT_POSITIONAL_ARGS = []

          if FIT_KEYWORD_ARGS is None:
              FIT_KEYWORD_ARGS = {}

          if EARLYSTOPPING_KEYWORD_ARGS is None:
              EARLYSTOPPING_KEYWORD_ARGS = {}

          assert isinstance(CONSTRUCTOR_POSITIONAL_ARGS, list), "CONSTRUCTOR_POSITIONAL_ARGS must be a list"
          assert isinstance(CONSTRUCTOR_KEYWORD_ARGS, dict), "CONSTRUCTOR_KEYWORD_ARGS must be a dict"

          assert isinstance(FIT_POSITIONAL_ARGS, list), "FIT_POSITIONAL_ARGS must be a list"
          assert isinstance(FIT_KEYWORD_ARGS, dict), "FIT_KEYWORD_ARGS must be a dict"

          assert isinstance(EARLYSTOPPING_KEYWORD_ARGS, dict), "EARLYSTOPPING_KEYWORD_ARGS must be a dict"


          self.CONSTRUCTOR_POSITIONAL_ARGS = CONSTRUCTOR_POSITIONAL_ARGS
          self.CONSTRUCTOR_KEYWORD_ARGS = CONSTRUCTOR_KEYWORD_ARGS

          self.FIT_POSITIONAL_ARGS = FIT_POSITIONAL_ARGS
          self.FIT_KEYWORD_ARGS = FIT_KEYWORD_ARGS

          self.EARLYSTOPPING_KEYWORD_ARGS = EARLYSTOPPING_KEYWORD_ARGS





    def copy(self):


        clone_object = SearchInputRecommenderArgs(
                            CONSTRUCTOR_POSITIONAL_ARGS = self.CONSTRUCTOR_POSITIONAL_ARGS.copy(),
                            CONSTRUCTOR_KEYWORD_ARGS = self.CONSTRUCTOR_KEYWORD_ARGS.copy(),
                            FIT_POSITIONAL_ARGS = self.FIT_POSITIONAL_ARGS.copy(),
                            FIT_KEYWORD_ARGS = self.FIT_KEYWORD_ARGS.copy(),
                            EARLYSTOPPING_KEYWORD_ARGS = self.EARLYSTOPPING_KEYWORD_ARGS.copy(),
                            )


        return clone_object




def get_result_string_prettyprint(result_series_single_cutoff, n_decimals=7):

    output_str = ""

    for metric, value in result_series_single_cutoff.items():
        output_str += "{}: {:.{n_decimals}f}, ".format(metric, value, n_decimals = n_decimals)

    return output_str


class NeverMatch(Exception):
    'An exception class that is never raised by any code anywhere'

class SearchAbstractClass(object):

    ALGORITHM_NAME = "SearchAbstractClass"

    # Available values for the save_model attribute
    _SAVE_MODEL_VALUES = ["all", "best", "last", "no"]

    # Available values for the evaluate_on_test attribute
    _EVALUATE_ON_TEST_VALUES = ["all", "best", "last", "no"]

    # Value to be assigned to invalid configuration or if an Exception is raised
    INVALID_CONFIG_VALUE = np.finfo(np.float16).max

    def __init__(self, recommender_class,
                 evaluator_validation = None,
                 evaluator_test = None,
                 verbose = True):

        super(SearchAbstractClass, self).__init__()

        self.recommender_class = recommender_class
        self.verbose = verbose
        self.log_file = None
        self.evaluator_validation = evaluator_validation

        if evaluator_test is None:
            self.evaluator_test = None
        else:
            self.evaluator_test = evaluator_test


    def search(self, recommender_input_args,
               hyperparameter_search_space,
               metric_to_optimize = "MAP",
               cutoff_to_optimize = None,
               n_cases = None,
               output_folder_path = None,
               output_file_name_root = None,
               parallelize = False,
               save_model = "best",
               evaluate_on_test = "best",
               save_metadata = True,
               terminate_on_memory_error = True,
               ):

        raise NotImplementedError("Function search not implemented for this class")


    def _was_already_evaluated_check(self, current_fit_hyperparameters_dict):
        """
        Check if the current hyperparameter configuration was already evaluated
        :param current_fit_hyperparameters_dict:
        :return:
        """

        raise NotImplementedError("Function search not implemented for this class")

    def _set_search_attributes(self, recommender_input_args,
                               recommender_input_args_last_test,
                               hyperparameter_names,
                               metric_to_optimize,
                               cutoff_to_optimize,
                               output_folder_path,
                               output_file_name_root,
                               resume_from_saved,
                               save_metadata,
                               save_model,
                               evaluate_on_test,
                               n_cases,
                               terminate_on_memory_error):


        if save_model not in self._SAVE_MODEL_VALUES:
           raise ValueError("{}: argument save_model must be in '{}', provided was '{}'.".format(self.ALGORITHM_NAME, self._SAVE_MODEL_VALUES, save_model))

        if evaluate_on_test not in self._EVALUATE_ON_TEST_VALUES:
           raise ValueError("{}: argument evaluate_on_test must be in '{}', provided was '{}'.".format(self.ALGORITHM_NAME, self._EVALUATE_ON_TEST_VALUES, evaluate_on_test))


        self.output_folder_path = output_folder_path
        self.output_file_name_root = output_file_name_root

        # If directory does not exist, create
        if not os.path.exists(self.output_folder_path):
            os.makedirs(self.output_folder_path)

        self.log_file = open(self.output_folder_path + self.output_file_name_root + "_{}.txt".format(self.ALGORITHM_NAME), "a")

        if save_model == "last" and recommender_input_args_last_test is None:
            self._write_log("{}: argument save_model is 'last' but no recommender_input_args_last_test provided, saving best model on train data alone.".format(self.ALGORITHM_NAME))
            save_model = "best"



        self.recommender_input_args = recommender_input_args
        self.recommender_input_args_last_test = recommender_input_args_last_test
        self.metric_to_optimize = metric_to_optimize
        self.cutoff_to_optimize = cutoff_to_optimize
        self.resume_from_saved = resume_from_saved
        self.terminate_on_memory_error = terminate_on_memory_error
        self.save_metadata = save_metadata
        self.save_model = save_model
        self.evaluate_on_test = "no" if self.evaluator_test is None else evaluate_on_test

        self.model_counter = 0
        self.n_cases = n_cases
        self._init_metadata_dict(n_cases = n_cases, hyperparameter_names = hyperparameter_names)

        if self.save_metadata:
            self.dataIO = DataIO(folder_path = self.output_folder_path)



    def _init_metadata_dict(self, n_cases, hyperparameter_names):

        self.metadata_dict = {"algorithm_name_search": self.ALGORITHM_NAME,
                              "algorithm_name_recommender": self.recommender_class.RECOMMENDER_NAME,
                              "metric_to_optimize": self.metric_to_optimize,
                              "cutoff_to_optimize": self.cutoff_to_optimize,
                              "exception_list": [None]*n_cases,

                              "hyperparameters_df": pd.DataFrame(columns = hyperparameter_names, index = np.arange(n_cases), dtype=object),
                              "hyperparameters_best": None,
                              "hyperparameters_best_index": None,

                              "result_on_validation_df": None,
                              "result_on_validation_best": None,
                              "result_on_test_df": None,
                              "result_on_test_best": None,
                              "result_on_earlystopping_df": pd.DataFrame(dtype=object) if issubclass(self.recommender_class, Incremental_Training_Early_Stopping) else None,

                              "time_df": pd.DataFrame(columns = ["train", "validation", "test"], index = np.arange(n_cases)),

                              "time_on_train_total": 0.0,
                              "time_on_train_avg": 0.0,

                              "time_on_validation_total": 0.0,
                              "time_on_validation_avg": 0.0,

                              "time_on_test_total": 0.0,
                              "time_on_test_avg": 0.0,

                              "result_on_last": None,
                              "time_on_last_df": pd.DataFrame(columns = ["train", "test"], index = [0]),
                              }

    def _print(self, string):
        if self.verbose:
            print(string)


    def _write_log(self, string):

        self._print(string)

        if self.log_file is not None:
            self.log_file.write(string)
            self.log_file.flush()


    def _fit_model(self, current_fit_hyperparameters):

        start_time = time.time()

        # Construct a new recommender instance
        recommender_instance = self.recommender_class(*self.recommender_input_args.CONSTRUCTOR_POSITIONAL_ARGS,
                                                      **self.recommender_input_args.CONSTRUCTOR_KEYWORD_ARGS)

        recommender_instance.fit(*self.recommender_input_args.FIT_POSITIONAL_ARGS,
                                 **self.recommender_input_args.FIT_KEYWORD_ARGS,
                                 **self.recommender_input_args.EARLYSTOPPING_KEYWORD_ARGS,
                                 **current_fit_hyperparameters)

        train_time = time.time() - start_time

        return recommender_instance, train_time


    def _evaluate_on_validation(self, current_fit_hyperparameters, was_already_evaluated_flag, was_already_evaluated_index):
        """
        Fit and evaluate model with the given hyperparameter configuration on the validation set, or
        load previously explored configuration
        :param current_fit_hyperparameters:
        :param was_already_evaluated_flag:
        :param was_already_evaluated_index:
        :return:
        """


        if not was_already_evaluated_flag:
            # Add hyperparameter values into dataframe iteratively because the simple solution
            # hyperparameters_df.loc[self.model_counter] = current_fit_hyperparameters
            # would (sometimes?) automatically convert integers to floats, which is undesirable (e.g., for the topK value)
            # This occurs when the dictionary contains only numerical data (int, floats) but not when it contains also strings or booleans
            for key in current_fit_hyperparameters.keys():
                self.metadata_dict["hyperparameters_df"].loc[self.model_counter,key] = current_fit_hyperparameters[key]

            recommender_instance, train_time = self._fit_model(current_fit_hyperparameters)
            start_time = time.time()

            # Evaluate recommender and get results for the first cutoff
            result_df, _ = self.evaluator_validation.evaluateRecommender(recommender_instance)

            evaluation_time = time.time() - start_time

            # If the recommender uses Earlystopping, get the selected number of epochs instead of the maximum
            if isinstance(recommender_instance, Incremental_Training_Early_Stopping):
                for epoch_key, epoch_value in recommender_instance.get_early_stopping_final_epochs_dict().items():
                    self.metadata_dict["hyperparameters_df"].loc[self.model_counter, epoch_key] = int(epoch_value)

                ## This is to ensure backward compatibility
                if "result_on_earlystopping_df" not in self.metadata_dict:
                    self.metadata_dict["result_on_earlystopping_df"] = pd.DataFrame(dtype=object) if issubclass(self.recommender_class, Incremental_Training_Early_Stopping) else None

                # Add the data from all validation steps
                if self.metadata_dict["result_on_earlystopping_df"] is not None and recommender_instance.get_validation_summary_table() is not None:
                    earlystopping_df_multiindex = pd.concat({self.model_counter: recommender_instance.get_validation_summary_table()}, names=['model_counter'])
                    self.metadata_dict["result_on_earlystopping_df"] = pd.concat([self.metadata_dict["result_on_earlystopping_df"], earlystopping_df_multiindex])


        else:
            # If it was already evaluated load the data
            recommender_instance = None

            self.metadata_dict["hyperparameters_df"].loc[self.model_counter] = self.metadata_dict["hyperparameters_df"].loc[was_already_evaluated_index].copy()
            result_df = self.metadata_dict["result_on_validation_df"].loc[was_already_evaluated_index].copy()
            train_time = self.metadata_dict["time_df"].loc[was_already_evaluated_index, "train"]
            evaluation_time = self.metadata_dict["time_df"].loc[was_already_evaluated_index, "validation"]


        if self.metadata_dict["result_on_validation_df"] is None:
            # The dataframe will have the case number and cutoff as index, the metric name as column
            self.metadata_dict["result_on_validation_df"] = create_result_multiindex_dataframe(self.n_cases, result_df)

        add_result_to_multiindex_dataframe(self.metadata_dict["result_on_validation_df"], result_df, self.model_counter)

        self.metadata_dict["time_df"].loc[self.model_counter, "train"] = train_time
        self.metadata_dict["time_df"].loc[self.model_counter, "validation"] = evaluation_time

        self.metadata_dict["time_on_train_avg"] = self.metadata_dict["time_df"]["train"].mean(axis=0, skipna=True)
        self.metadata_dict["time_on_train_total"] = self.metadata_dict["time_df"]["train"].sum(axis=0, skipna=True)
        self.metadata_dict["time_on_validation_avg"] = self.metadata_dict["time_df"]["validation"].mean(axis=0, skipna=True)
        self.metadata_dict["time_on_validation_total"] = self.metadata_dict["time_df"]["validation"].sum(axis=0, skipna=True)

        return result_df, recommender_instance





    def _evaluate_on_test(self, recommender_instance, current_fit_hyperparameters_dict,
                                was_already_evaluated_flag, was_already_evaluated_index, print_log = True):

        if was_already_evaluated_flag:
            result_df_test = self.metadata_dict['result_on_test_df'].loc[was_already_evaluated_index].copy()
            evaluation_test_time = self.metadata_dict["time_df"].loc[was_already_evaluated_index, "test"]

        else:
            start_time = time.time()
            result_df_test, _ = self.evaluator_test.evaluateRecommender(recommender_instance)
            evaluation_test_time = time.time() - start_time


        result_string = get_result_string_df(result_df_test)

        if print_log:
            self._write_log("{}: Config evaluated with evaluator_test. Config: {} - results:\n{}\n".format(self.ALGORITHM_NAME,
                                                                                                            current_fit_hyperparameters_dict,
                                                                                                            result_string))


        if self.metadata_dict["result_on_test_df"] is None:
            # The dataframe will have the case number and cutoff as index, the metric name as column
            self.metadata_dict["result_on_test_df"] = create_result_multiindex_dataframe(self.n_cases, result_df_test)

        add_result_to_multiindex_dataframe(self.metadata_dict["result_on_test_df"], result_df_test, self.model_counter)

        self.metadata_dict["time_df"].loc[self.model_counter, "test"] = evaluation_test_time
        self.metadata_dict["time_on_test_avg"] = self.metadata_dict["time_df"]["test"].mean(axis=0, skipna=True)
        self.metadata_dict["time_on_test_total"] = self.metadata_dict["time_df"]["test"].sum(axis=0, skipna=True)

        return result_df_test


    def _evaluate_on_test_with_data_last(self):

        start_time = time.time()

        # Construct a new recommender instance
        recommender_instance = self.recommender_class(*self.recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS,
                                                      **self.recommender_input_args_last_test.CONSTRUCTOR_KEYWORD_ARGS)

        # Check if last was already evaluated
        if self.resume_from_saved and self.metadata_dict["result_on_last"] is not None:
            self._print("{}: Resuming '{}'... Result on last already available.".format(self.ALGORITHM_NAME, self.output_file_name_root))
            return

        self._print("{}: Evaluation with constructor data for final test. Using best config: {}".format(self.ALGORITHM_NAME, self.metadata_dict["hyperparameters_best"]))

        # Use the hyperparameters that have been saved
        assert self.metadata_dict["hyperparameters_best"] is not None, "{}: Best hyperparameters not available, the search might have failed.".format(self.ALGORITHM_NAME)
        hyperparameters_best_args = self.metadata_dict["hyperparameters_best"].copy()

        recommender_instance.fit(*self.recommender_input_args_last_test.FIT_POSITIONAL_ARGS,
                                 **self.recommender_input_args.FIT_KEYWORD_ARGS,
                                 **hyperparameters_best_args)

        train_time = time.time() - start_time
        self.metadata_dict["time_on_last_df"].loc[0, "train"] = train_time

        if self.evaluate_on_test in ["all", "best", "last"]:
            start_time = time.time()
            result_df_test, _ = self.evaluator_test.evaluateRecommender(recommender_instance)
            evaluation_test_time = time.time() - start_time

            self._write_log("{}: Best config evaluated with evaluator_test with constructor data for final test. Config: {} - results:\n{}\n".format(self.ALGORITHM_NAME,
                                                                                                                                              self.metadata_dict["hyperparameters_best"],
                                                                                                                                              get_result_string_df(result_df_test)))
            self.metadata_dict["result_on_last"] = result_df_test
            self.metadata_dict["time_on_last_df"].loc[0, "test"] = evaluation_test_time


        if self.save_metadata:
            self.dataIO.save_data(data_dict_to_save = self.metadata_dict.copy(),
                                  file_name = self.output_file_name_root + "_metadata")

        if self.save_model in ["all", "best", "last"]:
            self._print("{}: Saving model in {}\n".format(self.ALGORITHM_NAME, self.output_folder_path + self.output_file_name_root))
            recommender_instance.save_model(self.output_folder_path, file_name =self.output_file_name_root + "_best_model_last")




    def _objective_function(self, current_fit_hyperparameters_dict):

        try:
            self._print("{}: Testing config: {}".format(self.ALGORITHM_NAME, current_fit_hyperparameters_dict))

            was_already_evaluated_flag, was_already_evaluated_index = self._was_already_evaluated_check(current_fit_hyperparameters_dict)
            result_df, recommender_instance = self._evaluate_on_validation(current_fit_hyperparameters_dict, was_already_evaluated_flag, was_already_evaluated_index)

            result_series = result_df.loc[self.metadata_dict["cutoff_to_optimize"]]
            current_result = - result_series[self.metric_to_optimize]

            current_fit_hyperparameters_dict = self.metadata_dict["hyperparameters_df"].loc[self.model_counter].to_dict()

            # Save current model if "all" is chosen
            if self.save_model in ["all"] and not was_already_evaluated_flag:
                self._print("{}: Saving model in {}\n".format(self.ALGORITHM_NAME, self.output_folder_path + self.output_file_name_root))
                recommender_instance.save_model(self.output_folder_path, file_name = self.output_file_name_root + "_model_{}".format(self.model_counter))

            # Check if this is a new best hyperparameter configuration
            if self.metadata_dict["result_on_validation_best"] is None:
                new_best_config_found = True
            else:
                best_solution_val = self.metadata_dict["result_on_validation_best"][self.metric_to_optimize]
                new_best_config_found = best_solution_val < result_series[self.metric_to_optimize]


            if new_best_config_found:
                self._write_log("{}: New best config found. Config {}: {} - results: {}\n".format(self.ALGORITHM_NAME,
                                                                                                  self.model_counter,
                                                                                                  current_fit_hyperparameters_dict,
                                                                                                  get_result_string_prettyprint(result_series, n_decimals=7)))

                if self.evaluate_on_test in ["all", "best"]:
                    result_df_test = self._evaluate_on_test(recommender_instance, current_fit_hyperparameters_dict,
                                                            was_already_evaluated_flag, was_already_evaluated_index, print_log = True)


            else:

                # Config is either suboptimal or was already explored previously
                self._write_log("{}: Config {} {}. Config: {} - results: {}\n".format(self.ALGORITHM_NAME,
                                                                                      self.model_counter,
                                                                                      "is suboptimal" if not was_already_evaluated_flag else "was already explored at index {}".format(was_already_evaluated_index),
                                                                                      current_fit_hyperparameters_dict,
                                                                                      get_result_string_prettyprint(result_series, n_decimals=7)))

                if self.evaluate_on_test in ["all"]:
                    result_df_test = self._evaluate_on_test(recommender_instance, current_fit_hyperparameters_dict,
                                                            was_already_evaluated_flag, was_already_evaluated_index, print_log = True)


            if current_result >= self.INVALID_CONFIG_VALUE:
                self._write_log("{}: WARNING! Config {} returned a value equal or worse than the default value to be assigned to invalid configurations."
                                " If no better valid configuration is found, this hyperparameter search may produce an invalid result.\n")


            if new_best_config_found:
                self.metadata_dict["hyperparameters_best"] = current_fit_hyperparameters_dict.copy()
                self.metadata_dict["hyperparameters_best_index"] = self.model_counter
                self.metadata_dict["result_on_validation_best"] = result_series.to_dict()

                if self.evaluate_on_test in ["all", "best"]:
                    self.metadata_dict["result_on_test_best"] = result_df_test.copy()

                # Clean any previous data about the "last"
                # If the search has been extended then the "last" is recomputed only if a better solution is found
                self.metadata_dict["result_on_last"] = None
                self.metadata_dict["time_on_last_df"] = pd.DataFrame(columns = ["train", "test"], index = [0])

                # Save best model if "all" and "best" are chosen
                if self.save_model in ["all", "best"]:
                    self._print("{}: Saving model in {}\n".format(self.ALGORITHM_NAME, self.output_folder_path + self.output_file_name_root))
                    recommender_instance.save_model(self.output_folder_path, file_name =self.output_file_name_root + "_best_model")


        except (KeyboardInterrupt, SystemExit) as e:
            # If getting a interrupt, terminate without saving the exception
            raise e

        # Catch exception only if terminate_on_memory_error is True
        except MEMORY_ERROR_EXCEPTION_TUPLE if self.terminate_on_memory_error else (NeverMatch) as e:
            self._print("{}: Search for '{}' interrupted due to MemoryError.".format(self.ALGORITHM_NAME, self.metadata_dict["algorithm_name_recommender"]))
            return

        except:
            # Catch any error: Exception, Tensorflow errors etc...
            traceback_string = traceback.format_exc()
            self._write_log("{}: Config {} Exception. Config: {} - Exception: {}\n".format(self.ALGORITHM_NAME,
                                                                                           self.model_counter,
                                                                                           current_fit_hyperparameters_dict,
                                                                                           traceback_string))

            self.metadata_dict["exception_list"][self.model_counter] = traceback_string

            # Assign to this configuration the worst possible score
            # Being a minimization problem, set it to the max value of a float
            current_result = + self.INVALID_CONFIG_VALUE
            traceback.print_exc()


        if self.save_metadata:
            self.dataIO.save_data(data_dict_to_save = self.metadata_dict.copy(),
                                  file_name = self.output_file_name_root + "_metadata")

        self.model_counter += 1

        return current_result
