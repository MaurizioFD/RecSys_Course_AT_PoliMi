#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/12/18

@author: Maurizio Ferrari Dacrema
"""

from HyperparameterTuning.SearchAbstractClass import SearchAbstractClass
import traceback

class SearchSingleCase(SearchAbstractClass):

    ALGORITHM_NAME = "SearchSingleCase"

    def __init__(self, recommender_class, evaluator_validation = None, evaluator_test = None, verbose = True):

        super(SearchSingleCase, self).__init__(recommender_class,
                                               evaluator_validation= evaluator_validation,
                                               evaluator_test=evaluator_test,
                                               verbose = verbose)



    def _resume_from_saved(self):

        try:
            self.metadata_dict = self.dataIO.load_data(file_name = self.output_file_name_root + "_metadata")

        except (KeyboardInterrupt, SystemExit) as e:
            # If getting a interrupt, terminate without saving the exception
            raise e

        except FileNotFoundError:
            self._write_log("{}: Resuming '{}' Failed, no such file exists.\n".format(self.ALGORITHM_NAME, self.output_file_name_root))
            return False

        except Exception as e:
            self._write_log("{}: Resuming '{}' Failed, generic exception: {}.\n".format(self.ALGORITHM_NAME, self.output_file_name_root, str(e)))
            traceback.print_exc()
            return False

        assert self.metadata_dict['algorithm_name_search'] == self.ALGORITHM_NAME, \
            "{}: Loaded data inconsistent with current search algorithm".format(self.ALGORITHM_NAME)

        assert len(self.metadata_dict['hyperparameters_df']) == 1, \
            "{}: Loaded data inconsistent with current search algorithm".format(self.ALGORITHM_NAME)

        self.model_counter = 1
        self.n_loaded_counter = self.model_counter

        self._print("{}: Resuming '{}'... Loaded {} configurations.".format(self.ALGORITHM_NAME, self.output_file_name_root, self.model_counter))
        return True


    def _was_already_evaluated_check(self, current_fit_hyperparameters_dict):
        """
        Check if the current hyperparameter configuration was already evaluated
        :param current_fit_hyperparameters_dict:
        :return:
        """

        return False, None

    def _evaluate_on_validation(self, current_fit_hyperparameters, was_already_evaluated_flag, was_already_evaluated_index):

        if self.evaluator_validation is not None:
            return super(SearchSingleCase, self)._evaluate_on_validation(current_fit_hyperparameters, was_already_evaluated_flag, was_already_evaluated_index)

        else:
            recommender_instance, train_time = self._fit_model(current_fit_hyperparameters)
            return {self.metadata_dict["metric_to_optimize"]: 0.0}, "", recommender_instance, train_time, None



    def search(self, recommender_input_args,
               fit_hyperparameters_values = None,
               metric_to_optimize = "MAP",
               cutoff_to_optimize = None,
               output_folder_path = None,
               output_file_name_root = None,
               save_metadata = True,
               recommender_input_args_last_test = None,
               resume_from_saved = False,
               save_model = "best",
               evaluate_on_test = "best",
               terminate_on_memory_error = True,
               ):


        assert fit_hyperparameters_values is not None, "{}: fit_hyperparameters_values must contain a dictionary".format(self.ALGORITHM_NAME)

        n_cases = 1

        self._set_search_attributes(recommender_input_args,
                                    recommender_input_args_last_test,
                                    fit_hyperparameters_values.keys(),
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


        self.n_loaded_counter = 0

        if self.resume_from_saved:
            if not self._resume_from_saved():
                self._objective_function(fit_hyperparameters_values)
        else:
            self._objective_function(fit_hyperparameters_values)

        if self.n_loaded_counter < self.model_counter:
            self._write_log("{}: Search complete. Best config is {}: {}\n".format(self.ALGORITHM_NAME,
                                                                           self.metadata_dict["hyperparameters_best_index"],
                                                                           self.metadata_dict["hyperparameters_best"]))


        if self.recommender_input_args_last_test is not None:
            self._evaluate_on_test_with_data_last()



