#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/03/2018

@author: Maurizio Ferrari Dacrema
"""

from ParameterTuning.AbstractClassSearch import AbstractClassSearch, DictionaryKeys, writeLog
from functools import partial
import traceback, pickle
import os, gc, math

import multiprocessing
from multiprocessing import Queue
from queue import Empty


def dump_garbage():
    """
    show us what's the garbage about
    """

    # force collection
    print("\nGARBAGE:")
    gc.collect()

    print("\nGARBAGE OBJECTS:")
    for x_pos in range(len(gc.garbage)):
        x = gc.garbage[x_pos]
        s = str(x)
        if len(s) > 80: s = s[:80]
        #print("type: {} \n\t s {} \n\t reffered by: {}".format(type(x), s, gc.get_referrers(x)))
        print("POS: {}, type: {} \n\t s {} \n".format(x_pos, type(x), s))

    print("\nDONE")
    pass

# gc.enable()
# gc.set_debug(gc.DEBUG_LEAK)


import itertools, random, time

def get_RAM_status():

    tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])

    return tot_m, used_m, free_m



def dereference_recommender_attributes(recommender_object):

    if recommender_object is None:
        return

    object_attributes = recommender_object.__dict__

    for key in object_attributes.keys():
        object_attributes[key] = None





def get_memory_threshold_reached(max_ram_occupied_perc):

    if max_ram_occupied_perc is not None:
        tot_RAM, used_RAM, _ = get_RAM_status()
        max_ram_occupied_bytes = tot_RAM*max_ram_occupied_perc

        memory_threshold_reached = used_RAM > max_ram_occupied_bytes
        memory_used_quota = used_RAM/tot_RAM

    else:

        memory_threshold_reached = False
        memory_used_quota = 0.0

    return memory_threshold_reached, memory_used_quota



import sys

class RandomSearch(AbstractClassSearch):

    ALGORITHM_NAME = "RandomSearch"

    def __init__(self, recommender_class, URM_test = None, evaluation_function_validation=None):

        super(RandomSearch, self).__init__(recommender_class, URM_test = URM_test, evaluation_function_validation= evaluation_function_validation)


    def build_all_cases_to_evaluate(self, n_cases):

        hyperparamethers_range_dictionary = self.dictionary_input[DictionaryKeys.FIT_RANGE_KEYWORD_ARGS]

        key_list = list(hyperparamethers_range_dictionary.keys())

        # Unpack list ranges from hyperparamethers to validate onto
        # * operator allows to transform a list of objects into positional arguments
        test_cases = itertools.product(*hyperparamethers_range_dictionary.values())

        paramether_dictionary_list = []

        for current_case in test_cases:

            paramether_dictionary_to_evaluate = {}

            for index in range(len(key_list)):

                paramether_dictionary_to_evaluate[key_list[index]] = current_case[index]

            paramether_dictionary_list.append(paramether_dictionary_to_evaluate)

        # Replicate list if necessary
        paramether_dictionary_list = paramether_dictionary_list * math.ceil(n_cases/len(paramether_dictionary_list))

        return paramether_dictionary_list








    def search(self, dictionary_input, metric ="map", n_cases = 30, output_root_path = None, parallelPoolSize = None, parallelize = True,
               save_model = "best", max_ram_occupied_perc = None):

        # Associate the params that will be returned by BayesianOpt object to those you want to save
        # E.g. with early stopping you know which is the optimal number of epochs only afterwards
        # but you might want to save it as well
        self.from_fit_params_to_saved_params = {}

        self.dictionary_input = dictionary_input.copy()
        self.output_root_path = output_root_path
        self.logFile = open(self.output_root_path + "_" + self.ALGORITHM_NAME + ".txt", "a")
        self.metric = metric
        self.model_counter = 0


        if max_ram_occupied_perc is None:
            self.max_ram_occupied_perc = 0.7
        else:
            # Try if current ram status is possible to read
            try:
                get_RAM_status()
                self.max_ram_occupied_perc = max_ram_occupied_perc
            except:
                writeLog(self.ALGORITHM_NAME + ": Unable to read RAM status, ignoring max RAM setting", self.logFile)
                self.max_ram_occupied_perc = None




        if save_model in ["no", "best", "all"]:
            self.save_model = save_model
        else:
            raise ValueError(self.ALGORITHM_NAME + ": save_model not recognized, acceptable values are: {}, given is {}".format(
                ["no", "best", "all"], save_model))


        if parallelPoolSize is None:
            self.parallelPoolSize = 1
        else:
            #self.parallelPoolSize = int(multiprocessing.cpu_count()/2)
            self.parallelPoolSize = parallelPoolSize

        self.best_solution_val = None
        self.best_solution_parameters = None
        self.best_solution_object = None


        paramether_dictionary_list = self.build_all_cases_to_evaluate(n_cases)

        # Randomize ordering of cases
        random.shuffle(paramether_dictionary_list)



        self.runSingleCase_partial = partial(self.runSingleCase,
                                             metric = metric)


        if parallelize:
            self.run_multiprocess_search(paramether_dictionary_list, n_cases)
        else:
            self.run_singleprocess_search(paramether_dictionary_list, n_cases)



        writeLog(self.ALGORITHM_NAME + ": Best config is: Config {}, {} value is {:.4f}\n".format(
            self.best_solution_parameters, metric, self.best_solution_val), self.logFile)




        return self.best_solution_parameters.copy()




    def update_on_new_result(self, process_object, num_cases_evaluated):

        paramether_dictionary_to_save = self.from_fit_params_to_saved_params_function(process_object.recommender,
                                                                                      process_object.paramether_dictionary_to_evaluate)

        if process_object.exception is not None:

            writeLog(self.ALGORITHM_NAME + ": Exception for config {}: {}\n".format(
                self.model_counter, paramether_dictionary_to_save, str(process_object.exception)), self.logFile)

            return


        if process_object.result_dict is None:
            writeLog(self.ALGORITHM_NAME + ": Result is None for config {}\n".format(
                self.model_counter, paramether_dictionary_to_save), self.logFile)

            return



        self.model_counter += 1

        # Always save best model separately
        if self.save_model == "all":
            print(self.ALGORITHM_NAME + ": Saving model in {}\n".format(self.output_root_path))
            process_object.recommender.saveModel(self.output_root_path, file_name="_model_{}".format(self.model_counter))

            pickle.dump(paramether_dictionary_to_save.copy(),
                        open(self.output_root_path + "_parameters_{}".format(self.model_counter), "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)


        if self.best_solution_val == None or self.best_solution_val<process_object.result_dict[self.metric]:


            writeLog(self.ALGORITHM_NAME + ": New best config found. Config {}: {} - results: {}\n".format(
                self.model_counter, paramether_dictionary_to_save, process_object.result_dict), self.logFile)

            pickle.dump(paramether_dictionary_to_save.copy(),
                        open(self.output_root_path + "_best_parameters", "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)

            self.best_solution_val = process_object.result_dict[self.metric]
            self.best_solution_parameters = paramether_dictionary_to_save.copy()

            dereference_recommender_attributes(self.best_solution_object)

            self.best_solution_object = process_object.recommender

            # Always save best model separately
            if self.save_model != "no":
                print(self.ALGORITHM_NAME + ": Saving model in {}\n".format(self.output_root_path))
                process_object.recommender.saveModel(self.output_root_path, file_name="_best_model")

            if self.URM_test is not None:
                self.evaluate_on_test(self.URM_test)


        else:
            writeLog(self.ALGORITHM_NAME + ": Config is suboptimal. Config {}: {} - results: {}\n".format(
                self.model_counter, paramether_dictionary_to_save, process_object.result_dict), self.logFile)

            dereference_recommender_attributes(process_object.recommender)


        #dump_garbage()



    def run_singleprocess_search(self, paramether_dictionary_list, num_cases_max):

        num_cases_evaluated = 0

        while num_cases_evaluated < num_cases_max:

            process_object = Process_object_data_and_evaluation(self.recommender_class, self.dictionary_input,
                                                                paramether_dictionary_list[num_cases_evaluated],
                                                                self.ALGORITHM_NAME, self.URM_validation, self.evaluation_function_validation)

            process_object.run("main")

            self.update_on_new_result(process_object, num_cases_evaluated)

            process_object = None

            #gc.collect()
            #dump_garbage()

            num_cases_evaluated += 1







    def run_multiprocess_search(self, paramether_dictionary_list, num_cases_max):


        # Te following function runs the search in parallel. As different configurations might have signifiantly divergent
        # runtime threads must be joined from the first to terminate and the objects might be big, therefore parallel.pool is not suitable

        num_cases_evaluated = 0
        num_cases_started = 0
        num_cases_active = 0
        termination_sent = False

        process_list = [None] * self.parallelPoolSize

        queue_job_todo = Queue()
        queue_job_done = Queue()


        get_memory_threshold_reached_partial = partial(get_memory_threshold_reached,
                                                       max_ram_occupied_perc = self.max_ram_occupied_perc)



        for current_process_index in range(self.parallelPoolSize):


            newProcess = multiprocessing.Process(target=process_worker, args=(queue_job_todo, queue_job_done, current_process_index, get_memory_threshold_reached_partial, ))

            process_list[current_process_index] = newProcess

            newProcess.start()
            newProcess = None

            print("Started process: {}".format(current_process_index))



        memory_threshold_reached, memory_used_quota = get_memory_threshold_reached(self.max_ram_occupied_perc)


        while num_cases_evaluated < num_cases_max:

            # Create as many new jobs as needed
            # Stop:     if the max number of paralle processes is reached or the max ram occupancy is reached
            #           if no other cases to explore
            # If no termination sent and active == 0, start one otherwise everything stalls
            # WARNING: apparently the function "queue_job_todo.empty()" is not reliable
            while ((num_cases_active < self.parallelPoolSize and not memory_threshold_reached) or (num_cases_active == 0)) \
                    and not termination_sent:

                memory_threshold_reached, memory_used_quota = get_memory_threshold_reached(self.max_ram_occupied_perc)

                if memory_threshold_reached:
                    writeLog(self.ALGORITHM_NAME + ": Memory threshold reached, occupied {:.4f} %\n".format(memory_used_quota), self.logFile)



                if  num_cases_started < num_cases_max and not memory_threshold_reached:

                    process_object = Process_object_data_and_evaluation(self.recommender_class, self.dictionary_input,
                                                                        paramether_dictionary_list[num_cases_started],
                                                                        self.ALGORITHM_NAME, self.URM_validation, self.evaluation_function)

                    queue_job_todo.put(process_object)
                    num_cases_started += 1
                    num_cases_active += 1
                    process_object = None


                if  num_cases_started >= num_cases_max and not termination_sent:
                    print("Termination sent")
                    queue_job_todo.put(None)
                    termination_sent = True


            # Read all completed jobs. WARNING: apparently the function "empty" is not reliable
            queue_job_done_is_empty = False

            while not queue_job_done_is_empty:

                try:
                    process_object = queue_job_done.get_nowait()

                    self.update_on_new_result(process_object, num_cases_evaluated)
                    num_cases_evaluated += 1
                    num_cases_active -=1
                    process_object = None

                except Empty:
                    queue_job_done_is_empty = True



            time.sleep(1)
            #print("num_cases_evaluated {}".format(num_cases_evaluated))

            #print("Evaluated {}, started {}, active {}".format(num_cases_evaluated, num_cases_started, num_cases_active))


        queue_job_todo.get()

        for current_process in process_list:
            #print("Waiting to Join {}".format(current_process))
            current_process.join()
            print("Joined {}".format(current_process))






def process_worker(queue_job_todo, queue_job_done, process_id, get_memory_threshold_reached):
    "Function to be used by the process, just run the wrapper object"

    process_object = queue_job_todo.get()
    memory_threshold_warning_printed = False

    while process_object is not None:

        # # Avoid queue.put to prevent process termination until all queue elements have been pulled
        # queue.cancel_join_thread()

        # Wait until there is enough RAM
        memory_threshold_reached, memory_used_quota = get_memory_threshold_reached()

        if not memory_threshold_reached:

            memory_threshold_warning_printed = False

            process_object.run(process_id)

            # "Send" result object ro main process
            queue_job_done.put(process_object)

            # Dereference
            process_object = None

            process_object = queue_job_todo.get()

        else:
            if not memory_threshold_warning_printed:
                memory_threshold_warning_printed = True
                print("Process: {} - Memory threshold reached, occupied {:.4f} %\n".format(process_id, memory_used_quota))

            time.sleep(5)


    #Ensure termination signal stays in queue
    queue_job_todo.put(None)

    # Termination signal
    print("Process: {} - Termination signal received".format(process_id))


    return












class Process_object_data_and_evaluation(object):

    def __init__(self, recommender_class, dictionary_input, paramether_dictionary_to_evaluate, ALGORITHM_NAME,
                 URM_validation, evaluation_function):

        super(Process_object_data_and_evaluation, self).__init__()


        self.recommender_class = recommender_class
        self.URM_validation = URM_validation
        self.dictionary_input = dictionary_input.copy()
        self.paramether_dictionary_to_evaluate = paramether_dictionary_to_evaluate.copy()
        self.ALGORITHM_NAME = ALGORITHM_NAME
        self.evaluation_function = evaluation_function

        self.exception = None
        self.recommender = None
        self.result_dict = None


    def __del__(self):
        # self.recommender_class = None
        # self.URM_validation = None
        # self.dictionary_input.clear()
        # self.paramether_dictionary_to_evaluate = None
        # self.ALGORITHM_NAME = None
        # self.evaluation_function = None
        # self.exception = None
        # self.recommender = None
        # self.result_dict = None

        object_attributes = self.__dict__

        for key in object_attributes.keys():
            object_attributes[key] = None




    def run(self, process_id):

        try:

            # Create an object of the same class of the imput
            # Passing the paramether as a dictionary
            self.recommender = self.recommender_class(*self.dictionary_input[DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS],
                                                 **self.dictionary_input[DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS])


            print(self.ALGORITHM_NAME + ": Process {} Config: {}".format(
                process_id, self.paramether_dictionary_to_evaluate))


            self.recommender.fit(*self.dictionary_input[DictionaryKeys.FIT_POSITIONAL_ARGS],
                            **self.dictionary_input[DictionaryKeys.FIT_KEYWORD_ARGS],
                            **self.paramether_dictionary_to_evaluate)

            self.result_dict = self.evaluation_function(self.recommender, self.URM_validation, self.paramether_dictionary_to_evaluate)


            print(self.ALGORITHM_NAME + ": Process {} Completed config: {} - result {}".format(
                process_id, self.paramether_dictionary_to_evaluate, self.result_dict))


            #self.result_dict = {"map": 0.0}

            return


        except Exception as exception:

            traceback.print_exc()

            print(self.ALGORITHM_NAME + ": Process {} Exception {}".format(
                process_id, str(exception)))

            self.result_dict = None
            self.exception = exception
