#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/07/2018

@author: Maurizio Ferrari Dacrema
"""

import time



def seconts_to_biggest_unit(time_in_seconds):

    conversion_factor = [
        ("sec", 60),
        ("min", 60),
        ("hour", 24),
        ("day", 365),
    ]

    terminate = False
    unit_index = 0

    new_time_value = time_in_seconds
    new_time_unit = "sec"

    while not terminate:

        next_time = new_time_value/conversion_factor[unit_index][1]

        if next_time >= 1.0:
            new_time_value = next_time

            unit_index += 1
            new_time_unit = conversion_factor[unit_index][0]

        else:
            terminate = True


    return new_time_value, new_time_unit




class Incremental_Training_Early_Stopping(object):
    """
    This class provides a function which trains a model applying early stopping

    The term "incremental" refers to the model that is updated at every epoch
    The term "best" refers to the incremental model which corresponded to the best validation score

    The object must implement the following methods:

    _run_epoch(self, num_epoch)                 : trains the model for one epoch (e.g. calling another object implementing the training cython, pyTorch...)


    _prepare_model_for_validation(self)         : ensures the recommender being trained can compute the predictions needed for the validation step


    _update_best_model(self)                    : updates the best model with the current incremental one


    _train_with_early_stopping(.)               : Function that executes the training, validation and early stopping by using the previously implemented functions



    """

    def __init__(self):
        super(Incremental_Training_Early_Stopping, self).__init__()


    def get_early_stopping_final_epochs_dict(self):
        """
        This function returns a dictionary to be used as optimal parameters in the .fit() function
        It provides the flexibility to deal with multiple early-stopping in a single algorithm
        e.g. in NeuMF there are three model components each with its own optimal number of epochs
        the return dict would be {"epochs": epochs_best_neumf, "epochs_gmf": epochs_best_gmf, "epochs_mlp": epochs_best_mlp}
        :return:
        """

        return {"epochs": self.epochs_best}



    def _run_epoch(self, num_epoch):
        """
        This function should run a single epoch on the object you train. This may either involve calling a function to do an epoch
        on a Cython object or a loop on the data points directly in python

        :param num_epoch:
        :return:
        """
        raise NotImplementedError()


    def _prepare_model_for_validation(self):
        """
        This function is executed before the evaluation of the current model
        It should ensure the current object "self" can be passed to the evaluator object

        E.G. if the epoch is done via Cython or PyTorch, this function should get the new parameter values from
        the cython or pytorch objects into the self. pyhon object
        :return:
        """
        raise NotImplementedError()


    def _update_best_model(self):
        """
        This function is called when the incremental model is found to have better validation score than the current best one
        So the current best model should be replaced by the current incremental one.

        Important, remember to clone the objects and NOT to create a pointer-reference, otherwise the best solution will be altered
        by the next epoch
        :return:
        """
        raise NotImplementedError()



    def _train_with_early_stopping(self, epochs, validation_every_n = None, stop_on_validation = False,
                                    validation_metric = None, lower_validatons_allowed = None, evaluator_object = None,
                                    algorithm_name = "Incremental_Training_Early_Stopping"):
        """

        :param epochs:                      max number of epochs the training will last
        :param validation_every_n:          number of epochs after which the model will be evaluated and a best_model selected
        :param stop_on_validation:          [True/False] whether to stop the training before the max number of epochs
        :param validation_metric:           which metric to use when selecting the best model, higher values are better
        :param lower_validatons_allowed:    number of contiguous validation steps required for the tranining to early-stop
        :param evaluator_object:            evaluator instance used to compute the validation metrics.
                                                If multiple cutoffs are available, the first one is used
        :param algorithm_name:              name of the algorithm to be displayed in the output updates
        :return: -


        Supported uses:

        - Train for max number of epochs with no validaton nor early stopping:

            _train_with_early_stopping(epochs = 100,
                                        evaluator_object = None
                                        validation_every_n,         not used
                                        stop_on_validation,         not used
                                        validation_metric,          not used
                                        lower_validatons_allowed,   not used
                                        )


        - Train for max number of epochs with validaton but NOT early stopping:

            _train_with_early_stopping(epochs = 100,
                                        evaluator_object = evaluator
                                        stop_on_validation = False
                                        validation_every_n = int value
                                        validation_metric = metric name string
                                        lower_validatons_allowed,   not used
                                        )


        - Train for max number of epochs with validaton AND early stopping:

            _train_with_early_stopping(epochs = 100,
                                        evaluator_object = evaluator
                                        stop_on_validation = True
                                        validation_every_n = int value
                                        validation_metric = metric name string
                                        lower_validatons_allowed = int value
                                        )



        """

        assert epochs>0, "{}: Number of epochs must be >= 0, passed was {}".format(algorithm_name, epochs)

        # Train for max number of epochs with no validaton nor early stopping
        # OR Train for max number of epochs with validaton but NOT early stopping
        # OR Train for max number of epochs with validaton AND early stopping
        assert evaluator_object is None or\
               (evaluator_object is not None and not stop_on_validation and validation_every_n is not None and validation_metric is not None) or\
               (evaluator_object is not None and stop_on_validation and validation_every_n is not None and validation_metric is not None and lower_validatons_allowed is not None),\
            "{}: Inconsistent parameters passed, please check the supported uses".format(algorithm_name)




        start_time = time.time()

        self.best_validation_metric = None
        lower_validatons_count = 0
        convergence = False

        self.epochs_best = 0

        currentEpoch = 0

        while currentEpoch < epochs and not convergence:

            self._run_epoch(currentEpoch)

            # If no validation required, always keep the latest
            if evaluator_object is None:

                self.epochs_best = currentEpoch

            # Determine whether a validaton step is required
            elif (currentEpoch + 1) % validation_every_n == 0:

                print("{}: Validation begins...".format(algorithm_name))

                self._prepare_model_for_validation()

                # If the evaluator validation has multiple cutoffs, choose the first one
                results_run, results_run_string = evaluator_object.evaluateRecommender(self)
                results_run = results_run[list(results_run.keys())[0]]

                print("{}: {}".format(algorithm_name, results_run_string))

                # Update optimal model
                current_metric_value = results_run[validation_metric]

                if self.best_validation_metric is None or self.best_validation_metric < current_metric_value:

                    print("{}: New best model found! Updating.".format(algorithm_name))

                    self.best_validation_metric = current_metric_value

                    self._update_best_model()

                    self.epochs_best = currentEpoch +1
                    lower_validatons_count = 0

                else:
                    lower_validatons_count += 1


                if stop_on_validation and lower_validatons_count >= lower_validatons_allowed:
                    convergence = True

                    elapsed_time = time.time() - start_time
                    new_time_value, new_time_unit = seconts_to_biggest_unit(elapsed_time)

                    print("{}: Convergence reached! Terminating at epoch {}. Best value for '{}' at epoch {} is {:.4f}. Elapsed time {:.2f} {}".format(
                        algorithm_name, currentEpoch+1, validation_metric, self.epochs_best, self.best_validation_metric, new_time_value, new_time_unit))


            elapsed_time = time.time() - start_time
            new_time_value, new_time_unit = seconts_to_biggest_unit(elapsed_time)

            print("{}: Epoch {} of {}. Elapsed time {:.2f} {}".format(
                algorithm_name, currentEpoch+1, epochs, new_time_value, new_time_unit))

            currentEpoch += 1


        # If no validation required, keep the latest
        if evaluator_object is None:

            self._prepare_model_for_validation()
            self._update_best_model()


        # Stop when max epochs reached and not early-stopping
        if not convergence:
            elapsed_time = time.time() - start_time
            new_time_value, new_time_unit = seconts_to_biggest_unit(elapsed_time)

            if evaluator_object is not None:
                print("{}: Terminating at epoch {}. Best value for '{}' at epoch {} is {:.4f}. Elapsed time {:.2f} {}".format(
                    algorithm_name, currentEpoch+1, validation_metric, self.epochs_best, self.best_validation_metric, new_time_value, new_time_unit))
            else:
                print("{}: Terminating at epoch {}. Elapsed time {:.2f} {}".format(
                    algorithm_name, currentEpoch+1, new_time_value, new_time_unit))

