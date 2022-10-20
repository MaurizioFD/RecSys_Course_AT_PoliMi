#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/12/18

@author: Emanuele Chioso, Maurizio Ferrari Dacrema
"""

from skopt import gp_minimize
import pandas as pd
import numpy as np
import time, os
from skopt.space import Real, Integer, Categorical
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit

from HyperparameterTuning.SearchAbstractClass import SearchAbstractClass
import traceback

def _extend_dataframe(initial_dataframe, new_rows):

    if initial_dataframe is None:
        return None

    if isinstance(initial_dataframe.index, pd.MultiIndex):
        second_level_index = initial_dataframe.loc[0].index
        first_level_index = initial_dataframe.index.get_level_values(0)
        cases_cutoff_multiindex = pd.MultiIndex.from_product([range(first_level_index.max()+1, first_level_index.max()+1+new_rows),
                                                              second_level_index])
        cases_cutoff_multiindex.set_names("cutoff", level=1, inplace=True)
        new_df = pd.DataFrame(columns=initial_dataframe.columns, index=cases_cutoff_multiindex)
    else:
        new_df = pd.DataFrame(columns = initial_dataframe.columns, index = range(len(initial_dataframe), len(initial_dataframe)+new_rows))

    extended_dataframe = initial_dataframe.append(new_df, ignore_index=False)

    return extended_dataframe


class TimeoutError(Exception):
    def __init__(self, max_total_time_seconds, current_total_time):
        max_total_time_seconds_value, max_total_time_seconds_unit = seconds_to_biggest_unit(max_total_time_seconds)
        current_total_time_seconds_value, current_total_time_seconds_unit = seconds_to_biggest_unit(current_total_time)

        message = "Total training and evaluation time is {:.2f} {}, exceeding the maximum threshold of {:.2f} {}".format(
            current_total_time_seconds_value, current_total_time_seconds_unit, max_total_time_seconds_value, max_total_time_seconds_unit)

        super().__init__(message)


class NoValidConfigError(Exception):
    def __init__(self):
        message = "No valid config was found during the initial random initialization"
        super().__init__(message)



class SearchBayesianSkopt(SearchAbstractClass):

    ALGORITHM_NAME = "SearchBayesianSkopt"

    def __init__(self, recommender_class, evaluator_validation = None, evaluator_test = None, verbose = True):

        assert evaluator_validation is not None, "{}: evaluator_validation must be provided".format(self.ALGORITHM_NAME)

        super(SearchBayesianSkopt, self).__init__(recommender_class,
                                                  evaluator_validation = evaluator_validation,
                                                  evaluator_test = evaluator_test,
                                                  verbose = verbose)



    def _set_skopt_params(self,
                          n_points = 10000,
                          n_jobs = 1,
                          # noise = 'gaussian',
                          noise = 1e-5,
                          acq_func = 'gp_hedge',
                          acq_optimizer = 'auto',
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
        self.n_jobs = n_jobs
        self.acq_func = acq_func
        self.acq_optimizer = acq_optimizer
        # Ensure that different processes would use different random states to avoid exploring the same configurations
        self.random_state = int(os.getpid() + time.time()) % np.iinfo(np.int32).max
        self.n_restarts_optimizer = n_restarts_optimizer
        self.verbose = verbose
        self.xi = xi
        self.kappa = kappa
        self.noise = noise
        self.x0 = x0
        self.y0 = y0



    def _resume_from_saved(self):

        try:
            self.metadata_dict = self.dataIO.load_data(file_name = self.output_file_name_root + "_metadata")

            ## This code can be used to remove all explored cases starting from the first exception raised
            ## Useful if, for example, you accidentally saturate the RAM and get memory errors, and want to clean
            ## the metadata to continue the search from before the exceptions were raised
            # start = None
            # for i, exc in enumerate(self.metadata_dict["exception_list"]):
            #     if exc is not None and start is None:
            #         start=i
            #
            # if start is not None:
            #     assert self.metadata_dict['hyperparameters_best_index'] < start
            #     self._remove_intermediate_cases(list(range(start, len(self.metadata_dict['hyperparameters_df']))))
            #
            #     self.dataIO.save_data(data_dict_to_save = self.metadata_dict.copy(),
            #               file_name = self.output_file_name_root + "_metadata")
            #
            # raise KeyboardInterrupt

            # Check if the data structures have to be extended to accomodate more cases
            n_cases_in_loaded_data = len(self.metadata_dict["hyperparameters_df"])
            if n_cases_in_loaded_data < self.n_cases:
                new_cases = self.n_cases-n_cases_in_loaded_data
                self._write_log("{}: Extending previous number of cases from {} to {}.\n".format(self.ALGORITHM_NAME, n_cases_in_loaded_data, self.n_cases))
                self.metadata_dict["exception_list"].extend([None]*new_cases)
                for dataframe_name in ["hyperparameters_df", "time_df", "result_on_validation_df", "result_on_test_df"]:
                    self.metadata_dict[dataframe_name] = _extend_dataframe(self.metadata_dict[dataframe_name], new_cases)

        except (KeyboardInterrupt, SystemExit) as e:
            # If getting a interrupt, terminate without saving the exception
            raise e

        except FileNotFoundError:
            self._write_log("{}: Resuming '{}' Failed, no such file exists.\n".format(self.ALGORITHM_NAME, self.output_file_name_root))
            self.resume_from_saved = False
            return None, None

        except Exception as e:
            self._write_log("{}: Resuming '{}' Failed, generic exception: {}.\n".format(self.ALGORITHM_NAME, self.output_file_name_root, str(e)))
            raise e

        # Get hyperparameter list and corresponding result
        # Make sure that the hyperparameters only contain those given as input and not others like the number of epochs selected by earlystopping
        # Add only those having a search space, in the correct ordering
        hyperparameters_df = self.metadata_dict['hyperparameters_df'][self.hyperparams_names]

        # Check if search was only done partially.
        # Some hyperparameters may be nans, but at least one should have a definite value.
        # All valid hyperprameter cases should be at the beginning
        self.model_counter = hyperparameters_df.notna().any(axis=1).sum()

        # If the data structure exists but is empty, return None
        if self.model_counter == 0:
            self.resume_from_saved = False
            return None, None

        assert hyperparameters_df[:self.model_counter].notna().any(axis=1).all(),\
                   "{}: Resuming '{}' Failed due to inconsistent data, valid hyperparameter configurations are not contiguous at the beginning of the dataframe.".format(self.ALGORITHM_NAME, self.output_file_name_root)

        hyperparameters_df = hyperparameters_df[:self.model_counter]

        # Check if single value categorical. It is aimed at intercepting
        # Hyperparameters that are chosen via early stopping and set them as the
        # maximum value as per hyperparameter search space. If not, the gp_minimize will return an error
        # as some values will be outside (lower) than the search space
        for hyperparameter_index, hyperparameter_name in enumerate(self.hyperparams_names):
            if isinstance(self.hyperparams_values[hyperparameter_index], Categorical) and len(self.hyperparams_values[hyperparameter_index].categories) == 1:
                hyperparameters_df[hyperparameter_name] = self.hyperparams_values[hyperparameter_index].bounds[0]

        hyperparameters_list_input = hyperparameters_df.values.tolist()


        result_on_validation_df = self.metadata_dict['result_on_validation_df']

        # All valid hyperparameters must have either a valid result or an exception
        for index in range(self.model_counter):
            is_exception = self.metadata_dict["exception_list"][index] is None
            is_validation_valid = result_on_validation_df is not None and result_on_validation_df[self.metric_to_optimize].notna()[index].any()
            assert is_exception == is_validation_valid,\
                   "{}: Resuming '{}' Failed due to inconsistent data. There cannot be both a valid result and an exception for the same case.".format(self.ALGORITHM_NAME, self.output_file_name_root)

        if result_on_validation_df is not None:
            result_on_validation_df = result_on_validation_df.fillna(value = - self.INVALID_CONFIG_VALUE, inplace=False)
            result_on_validation_df = result_on_validation_df.loc[:self.model_counter-1]
            result_on_validation_list_input = (- result_on_validation_df[self.metric_to_optimize].loc[:,self.cutoff_to_optimize]).to_list()
        else:
            # The validation result dataframe is created at the first valid configuration
            # The search can only progress if there is at least a valid config in the initial random start
            # If no valid result is present, proceed only if there is at least one random start left to sample
            if self.model_counter >= self.n_random_starts:
                raise NoValidConfigError()

            result_on_validation_list_input = [+ self.INVALID_CONFIG_VALUE]*self.model_counter

        assert len(hyperparameters_list_input) == len(result_on_validation_list_input), \
            "{}: Resuming '{}' Failed due to inconsistent data, there is a different number of hyperparameters and results.".format(self.ALGORITHM_NAME, self.output_file_name_root)

        self._print("{}: Resuming '{}'... Loaded {} configurations.".format(self.ALGORITHM_NAME, self.output_file_name_root, self.model_counter))

        return hyperparameters_list_input, result_on_validation_list_input




    def _was_already_evaluated_check(self, current_fit_hyperparameters_dict):
        """
        Check if the current hyperparameter configuration was already evaluated
        :param current_fit_hyperparameters_dict:
        :return:
        """


        hyperparameters_df = self.metadata_dict["hyperparameters_df"].copy()

        # Check if single value categorical. It is aimed at intercepting
        # Hyperparameters that are chosen via early stopping and set them as the
        # maximum value as per hyperparameter search space. If not, the gp_minimize will return an error
        # as some values will be outside (lower) than the search space
        for hyperparameter_index, hyperparameter_name in enumerate(self.hyperparams_names):
            if isinstance(self.hyperparams_values[hyperparameter_index], Categorical) and len(self.hyperparams_values[hyperparameter_index].categories) == 1:
                hyperparameters_df[hyperparameter_name] = self.hyperparams_values[hyperparameter_index].bounds[0]

        # Series and dataframe need to be aligned
        current_fit_hyperpar_series = pd.Series(current_fit_hyperparameters_dict)
        hyperparameters_df, current_fit_hyperpar_series = hyperparameters_df.align(current_fit_hyperpar_series, axis=1, copy=False)

        is_equal = (hyperparameters_df == current_fit_hyperpar_series).all(axis=1)

        if is_equal.any():
            return True, is_equal[is_equal].index[0]

        return False, None




    def search(self, recommender_input_args,
               hyperparameter_search_space,
               metric_to_optimize = None,
               cutoff_to_optimize = None,
               n_cases = None,
               n_random_starts = None,
               output_folder_path = None,
               output_file_name_root = None,
               save_model = "best",
               save_metadata = True,
               resume_from_saved = False,
               recommender_input_args_last_test = None,
               evaluate_on_test = "best",
               max_total_time = None,
               terminate_on_memory_error = True,
               ):
        """

        :param recommender_input_args:
        :param hyperparameter_search_space:
        :param metric_to_optimize:
        :param cutoff_to_optimize:
        :param n_cases:
        :param n_random_starts:
        :param output_folder_path:
        :param output_file_name_root:
        :param save_model:          "no"    don't save anything
                                    "all"   save every model
                                    "best"  save the best model trained on train data alone and on last, if present
                                    "last"  save only last, if present
        :param save_metadata:
        :param recommender_input_args_last_test:
        :return:
        """

        ### default hyperparameters for BayesianSkopt are set here
        self._set_skopt_params()

        self._set_search_attributes(recommender_input_args,
                                    recommender_input_args_last_test,
                                    hyperparameter_search_space.keys(),
                                    metric_to_optimize,
                                    cutoff_to_optimize,
                                    output_folder_path,
                                    output_file_name_root,
                                    resume_from_saved,
                                    save_metadata,
                                    save_model,
                                    evaluate_on_test,
                                    n_cases,
                                    terminate_on_memory_error)


        self.n_random_starts = n_random_starts
        self.n_calls = n_cases
        self.n_jobs = 1
        self.n_loaded_counter = 0

        self.max_total_time = max_total_time

        if self.max_total_time is not None:
            total_time_value, total_time_unit = seconds_to_biggest_unit(self.max_total_time)
            self._print("{}: The search has a maximum allotted time of {:.2f} {}".format(self.ALGORITHM_NAME, total_time_value, total_time_unit))


        self.hyperparams = dict()
        self.hyperparams_names = list()
        self.hyperparams_values = list()

        skopt_types = [Real, Integer, Categorical]

        for name, hyperparam in hyperparameter_search_space.items():

            if any(isinstance(hyperparam, sko_type) for sko_type in skopt_types):
                self.hyperparams_names.append(name)
                self.hyperparams_values.append(hyperparam)
                self.hyperparams[name] = hyperparam

            else:
                raise ValueError("{}: Unexpected hyperparameter type: {} - {}".format(self.ALGORITHM_NAME, str(name), str(hyperparam)))

        try:
            if self.resume_from_saved:
                hyperparameters_list_input, result_on_validation_list_saved = self._resume_from_saved()
                self.x0 = hyperparameters_list_input
                self.y0 = result_on_validation_list_saved

                self.n_loaded_counter = self.model_counter


            if self.n_calls - self.model_counter > 0:
                # When resuming an incomplete search the gp_minimize will continue to tell you "Evaluating function at random point" instead
                # of "Searching for the next optimal point". This may be due to a bug in the print rather than the underlying process
                # https://github.com/scikit-optimize/scikit-optimize/issues/949
                self.result = gp_minimize(self._objective_function_list_input,
                                          self.hyperparams_values,
                                          base_estimator=None,
                                          n_calls=max(0, self.n_calls - self.model_counter),
                                          n_initial_points= max(0, self.n_random_starts - self.model_counter),
                                          initial_point_generator = "random",
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

        except ValueError as e:
            traceback.print_exc()
            self._write_log("{}: Search interrupted due to ValueError. The evaluated configurations may have had all the same value.\n".format(self.ALGORITHM_NAME))
            return

        except NoValidConfigError as e:
            self._write_log("{}: Search interrupted. {}\n".format(self.ALGORITHM_NAME, e))
            return

        except TimeoutError as e:
            # When in TimeoutError, stop search but continue to train the _last model, if requested
            self._write_log("{}: Search interrupted. {}\n".format(self.ALGORITHM_NAME, e))


        if self.n_loaded_counter < self.model_counter:
            self._write_log("{}: Search complete. Best config is {}: {}\n".format(self.ALGORITHM_NAME,
                                                                           self.metadata_dict["hyperparameters_best_index"],
                                                                           self.metadata_dict["hyperparameters_best"]))

        if self.recommender_input_args_last_test is not None:
            self._evaluate_on_test_with_data_last()







    def _objective_function_list_input(self, current_fit_hyperparameters_list_of_values):
        """
        This function parses the hyperparameter list provided by the gp_minimize function into a dictionary that
        can be used for the fitting of the model and provided to the objective function defined in the abstract class

        This function also checks if the search should be interrupted if the time has expired or no valid config has been found

        :param current_fit_hyperparameters_list_of_values:
        :return:
        """

        # The search can only progress if the total training + validation time is lower than max threshold
        # The time necessary for the last case is estimated based on the time the corresponding case took
        total_current_time = self.metadata_dict["time_on_train_total"] + self.metadata_dict["time_on_validation_total"]
        estimated_last_time = self.metadata_dict["time_df"].loc[self.metadata_dict['hyperparameters_best_index']][["train", "validation"]].sum() if \
                              self.metadata_dict['hyperparameters_best_index'] is not None else 0


        if self.max_total_time is not None:
            # If there is no "last" use the current total time, otherwise estimate its required time form the average
            if self.recommender_input_args_last_test is None and total_current_time > self.max_total_time:
                raise TimeoutError(self.max_total_time, total_current_time)
            elif self.recommender_input_args_last_test is not None and total_current_time + estimated_last_time> self.max_total_time:
                raise TimeoutError(self.max_total_time, total_current_time + estimated_last_time)


        current_fit_hyperparameters_dict = dict(zip(self.hyperparams_names, current_fit_hyperparameters_list_of_values))
        result = self._objective_function(current_fit_hyperparameters_dict)

        # The search can only progress if there is at least a valid config in the initial random start
        if self.metadata_dict['result_on_validation_df'] is None and self.model_counter >= self.n_random_starts:
             raise NoValidConfigError()

        return result

