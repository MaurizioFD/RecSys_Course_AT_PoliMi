#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/07/2018

@author: Maurizio Ferrari Dacrema
"""

import time




class Incremental_Training_Early_Stopping(object):
    """
    This class provides a function which trains a model applying early stopping

    The term "incremental" refers to the model that is updated at every epoch
    The term "best" refers to the incremental model which corresponded to the best validation score

    The object must implement the following methods:

    __initialize_incremental_model(self)    : initializes the incremental model


    _run_epoch(self, num_epoch)             : trains the model for one epoch (e.g. calling another object implementing the training cython, pyTorch...)


    __update_incremental_model(self)        : updates the incremental model with the new one


     __update_best_model(self)           : updates the best model with the current incremental one


    """

    def __init__(self):
        super(Incremental_Training_Early_Stopping, self).__init__()

    def _initialize_incremental_model(self):
        """
        This function should initialized the data structures required by the model you are going to train.

        E.g. If the model uses a similarity matrix, here you should instantiate the global objects
        :return:
        """
        raise NotImplementedError()

    def _run_epoch(self, num_epoch):
        """
        This function should run a single epoch on the object you train. This may either involve calling a function to do an epoch
        on a Cython object or a loop on the data points directly in python

        :param num_epoch:
        :return:
        """
        raise NotImplementedError()

    def _update_incremental_model(self):
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

    # def _validation_incremental_model(self):
    #     raise NotImplementedError()


    def _train_with_early_stopping(self, epochs, validation_every_n, stop_on_validation,
                                    validation_metric, lower_validatons_allowed, evaluator_object,
                                    algorithm_name = "Incremental_Training_Early_Stopping"):


        start_time = time.time()

        self.best_validation_metric = None
        lower_validatons_count = 0
        convergence = False

        self._initialize_incremental_model()

        self.epochs_best = 0

        currentEpoch = 0

        while currentEpoch < epochs and not convergence:

            self._run_epoch(currentEpoch)

            # Determine whether a validaton step is required
            if evaluator_object is not None and (currentEpoch + 1) % validation_every_n == 0:

                print("{}: Validation begins...".format(algorithm_name))

                self._update_incremental_model()

                results_run, _ = evaluator_object.evaluateRecommender(self)
                results_run = results_run[list(results_run.keys())[0]]

                print("{}: {}".format(algorithm_name, results_run))

                # Update the D_best and V_best
                # If validation is required, check whether result is better
                if stop_on_validation:

                    current_metric_value = results_run[validation_metric]

                    if self.best_validation_metric is None or self.best_validation_metric < current_metric_value:

                        self.best_validation_metric = current_metric_value

                        self._update_best_model()

                        self.epochs_best = currentEpoch +1
                        lower_validatons_count = 0

                    else:
                        lower_validatons_count += 1

                    if lower_validatons_count >= lower_validatons_allowed:
                        convergence = True
                        print("{}: Convergence reached! Terminating at epoch {}. Best value for '{}' at epoch {} is {:.4f}. Elapsed time {:.2f} min".format(
                            algorithm_name, currentEpoch+1, validation_metric, self.epochs_best, self.best_validation_metric, (time.time() - start_time) / 60))

                else:
                    self.epochs_best = currentEpoch

            # If no validation required, always keep the latest
            if not stop_on_validation:
                self._update_best_model()

            print("{}: Epoch {} of {}. Elapsed time {:.2f} min".format(
                algorithm_name, currentEpoch+1, epochs, (time.time() - start_time) / 60))

            currentEpoch += 1
