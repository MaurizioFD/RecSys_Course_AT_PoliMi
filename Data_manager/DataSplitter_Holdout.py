#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/01/18

@author: Maurizio Ferrari Dacrema
"""

import scipy.sparse as sps
import numpy as np
from Recommenders.DataIO import DataIO

from Data_manager.DataSplitter import DataSplitter as _DataSplitter
from Data_manager.DataReader import DataReader as _DataReader
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_user_wise, split_train_in_two_percentage_global_sample
from Data_manager.DataReader_utils import compute_density, reconcile_mapper_with_removed_tokens
from Data_manager.data_consistency_check import assert_disjoint_matrices, assert_URM_ICM_mapper_consistency





class DataSplitter_Holdout(_DataSplitter):
    """
    The splitter creates a random holdout of three split: train, validation and test
    The split is performed user-wise
    """

    """
     - It exposes the following functions
        - load_data(save_folder_path = None, force_new_split = False)   loads the data or creates a new split
    
    
    """

    DATA_SPLITTER_NAME = "DataSplitter_Holdout"

    # Split quota percentage for Train, Validation, Test
    _SPLIT_QUOTA_LIST_DEFAULT = [70, 10, 20]

    _SPLIT_URM_NAME_LIST = ["URM_train", "URM_validation", "URM_test"]

    SPLIT_URM_DICT = None
    SPLIT_ICM_DICT = None
    SPLIT_UCM_DICT = None
    SPLIT_ICM_MAPPER_DICT = None
    SPLIT_UCM_MAPPER_DICT = None
    SPLIT_GLOBAL_MAPPER_DICT = None



    def __init__(self, dataReader_object:_DataReader,
                 split_interaction_quota_list = None, user_wise = True, allow_cold_users = False,
                 forbid_new_split = False, force_new_split = False):
        """

        :param dataReader_object:
        :param n_folds:
        :param force_new_split:
        :param forbid_new_split:
        :param save_folder_path:    path in which to save the loaded dataset
                                    None    use default "dataset_name/split_name/"
                                    False   do not save
        """

        if split_interaction_quota_list is None:
            split_interaction_quota_list = self._SPLIT_QUOTA_LIST_DEFAULT.copy()
            self._print("input_split_item_quota_list not provided, using default '{}'".format(split_interaction_quota_list))

        assert len(split_interaction_quota_list) == 3, "{}: input_split_item_quota_list must contain 3 values: Train, Validation, Test".format(self.DATA_SPLITTER_NAME)
        assert all(split_quota >= 0.0 and split_quota <= 100 for split_quota in split_interaction_quota_list), "{}: input_split_item_quota_list must contain values between 0 and 100".format(self.DATA_SPLITTER_NAME)
        assert sum(split_interaction_quota_list) == 100, "{}: input_split_item_quota_list must be a probability distribution and sum to 100, current data sums to '{}'".format(self.DATA_SPLITTER_NAME, sum(split_interaction_quota_list))


        self.input_split_interaction_quota_list = split_interaction_quota_list.copy()
        self.actual_split_interaction_quota_list = None
        self.allow_cold_users = allow_cold_users
        self.user_wise = user_wise

        super(DataSplitter_Holdout, self).__init__(dataReader_object, forbid_new_split=forbid_new_split, force_new_split=force_new_split)



    def _get_split_subfolder_name(self):
        """

        :return: warm_{n_folds}_fold/
        """

        if self.user_wise:
            user_wise_string = "user_wise"
        else:
            user_wise_string = "global_sample"

        return "holdout_{}_{}/".format('_'.join(str(split_quota) for split_quota in self.input_split_interaction_quota_list), user_wise_string)



    def get_statistics_URM(self):

        self._assert_is_initialized()

        n_users, n_items = self.SPLIT_URM_DICT["URM_train"].shape

        statistics_string = "DataReader: {}\n" \
                            "\tNum items: {}\n" \
                            "\tNum users: {}\n" \
                            "\tTrain \t\tquota {:.2f} ({:.2f}), \tinteractions {}, \tdensity {:.2E}\n" \
                            "\tValidation \tquota {:.2f} ({:.2f}), \tinteractions {}, \tdensity {:.2E}\n" \
                            "\tTest \t\tquota {:.2f} ({:.2f}), \tinteractions {}, \tdensity {:.2E}\n".format(
            self.dataReader_object._get_dataset_name(),
            n_items,
            n_users,
            self.input_split_interaction_quota_list[0], self.actual_split_interaction_quota_list[0], self.SPLIT_URM_DICT["URM_train"].nnz, compute_density(self.SPLIT_URM_DICT["URM_train"]),
            self.input_split_interaction_quota_list[1], self.actual_split_interaction_quota_list[1], self.SPLIT_URM_DICT["URM_validation"].nnz, compute_density(self.SPLIT_URM_DICT["URM_validation"]),
            self.input_split_interaction_quota_list[2], self.actual_split_interaction_quota_list[2], self.SPLIT_URM_DICT["URM_test"].nnz, compute_density(self.SPLIT_URM_DICT["URM_test"]),
        )

        self._print(statistics_string)

        print("\n")




    def get_ICM_from_name(self, ICM_name):
        return self.SPLIT_ICM_DICT[ICM_name].copy()

    def get_UCM_from_name(self, UCM_name):
        return self.SPLIT_UCM_DICT[UCM_name].copy()

    def get_statistics_ICM(self):

        self._assert_is_initialized()

        if len(self.dataReader_object.get_loaded_ICM_names())>0:

            for ICM_name, ICM_object in self.SPLIT_ICM_DICT.items():

                n_items, n_features = ICM_object.shape

                statistics_string = "\tICM name: {}, Num features: {}, feature occurrences: {}, density {:.2E}".format(
                    ICM_name,
                    n_features,
                    ICM_object.nnz,
                    compute_density(ICM_object)
                )

                print(statistics_string)


    def get_statistics_UCM(self):

        self._assert_is_initialized()

        if len(self.dataReader_object.get_loaded_UCM_names())>0:

            for UCM_name, UCM_object in self.SPLIT_UCM_DICT.items():

                n_items, n_features = UCM_object.shape

                statistics_string = "\tUCM name: {}, Num features: {}, feature occurrences: {}, density {:.2E}".format(
                    UCM_name,
                    n_features,
                    UCM_object.nnz,
                    compute_density(UCM_object)
                )

                print(statistics_string)


    def _assert_is_initialized(self):
         assert self.SPLIT_URM_DICT is not None, "{}: Unable to load data split. The split has not been generated yet, call the load_data function to do so.".format(self.DATA_SPLITTER_NAME)


    def get_holdout_split(self):
        """
        The train set is defined as all data except the one of that fold, which is the test
        :return: URM_train, URM_validation, URM_test
        """

        self._assert_is_initialized()

        return self.SPLIT_URM_DICT["URM_train"].copy(),\
               self.SPLIT_URM_DICT["URM_validation"].copy(),\
               self.SPLIT_URM_DICT["URM_test"].copy()


    def _split_data_from_original_dataset(self, save_folder_path):


        self.loaded_dataset = self.dataReader_object.load_data()
        self._load_from_DataReader_ICM_and_mappers(self.loaded_dataset)

        URM_all = self.loaded_dataset.get_URM_all()

        train_quota, validation_quota, test_quota = self.input_split_interaction_quota_list
        train_quota /= 100
        validation_quota /= 100
        test_quota /= 100

        if self.user_wise:
            URM_train_validation, URM_test = split_train_in_two_percentage_user_wise(URM_all, train_percentage = train_quota + validation_quota)
        else:
            URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = train_quota + validation_quota)

        #Adjust train quota to account for the reduced size of the sample
        # URM_train_validation * adjusted_train_quota = URM_all * train quota

        adjusted_train_quota = URM_all.nnz * train_quota / URM_train_validation.nnz

        if self.user_wise:
            URM_train, URM_validation = split_train_in_two_percentage_user_wise(URM_train_validation, train_percentage = adjusted_train_quota)
        else:
            URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage = adjusted_train_quota)




        if not self.allow_cold_users:

            user_interactions = np.ediff1d(URM_train.indptr)
            user_to_preserve = user_interactions >= 1
            user_to_remove = np.logical_not(user_to_preserve)

            n_users = URM_train.shape[0]

            if user_to_remove.sum() >0 :

                self._print("Removing {} ({:.2f} %) of {} users because they have no interactions in train data.".format(user_to_remove.sum(), user_to_remove.sum()/n_users*100, n_users))

                URM_train = URM_train[user_to_preserve,:]
                URM_validation = URM_validation[user_to_preserve,:]
                URM_test = URM_test[user_to_preserve,:]

                self.SPLIT_GLOBAL_MAPPER_DICT["user_original_ID_to_index"] = reconcile_mapper_with_removed_tokens(self.SPLIT_GLOBAL_MAPPER_DICT["user_original_ID_to_index"],
                                                                                                                  np.arange(0, len(user_to_remove), dtype=np.int)[user_to_remove])

                for UCM_name, UCM_object in self.SPLIT_UCM_DICT.items():
                    UCM_object = UCM_object[user_to_preserve,:]
                    self.SPLIT_UCM_DICT[UCM_name] = UCM_object


        self.SPLIT_URM_DICT = {
            "URM_train": URM_train,
            "URM_validation": URM_validation,
            "URM_test": URM_test,
        }

        self._compute_real_split_interaction_quota()

        self._save_split(save_folder_path)

        self._print("Split complete")



    def _compute_real_split_interaction_quota(self):

        self._assert_is_initialized()

        n_interactions_total = 0
        self.actual_split_interaction_quota_list = [None] * len(self.input_split_interaction_quota_list)

        for _, URM_object in self.SPLIT_URM_DICT.items():
            n_interactions_total += URM_object.nnz

        for index, (_, URM_object) in enumerate(self.SPLIT_URM_DICT.items()):

            real_quota = URM_object.nnz/n_interactions_total*100

            self.actual_split_interaction_quota_list[index] = real_quota





    def _save_split(self, save_folder_path):

        if save_folder_path:

            if self.allow_cold_users:
                allow_cold_users_suffix = "allow_cold_users"

            else:
                allow_cold_users_suffix = "only_warm_users"

            if self.user_wise:
                user_wise_string = "user_wise"
            else:
                user_wise_string = "global_sample"


            name_suffix = "_{}_{}".format(allow_cold_users_suffix, user_wise_string)


            split_parameters_dict = {
                            "input_split_interaction_quota_list": self.input_split_interaction_quota_list,
                            "actual_split_interaction_quota_list": self.actual_split_interaction_quota_list,
                            "allow_cold_users": self.allow_cold_users
                            }

            dataIO = DataIO(folder_path = save_folder_path)

            dataIO.save_data(data_dict_to_save = split_parameters_dict,
                             file_name = "split_parameters" + name_suffix)

            dataIO.save_data(data_dict_to_save = self.SPLIT_GLOBAL_MAPPER_DICT,
                             file_name = "split_mappers" + name_suffix)

            dataIO.save_data(data_dict_to_save = self.SPLIT_URM_DICT,
                             file_name = "split_URM" + name_suffix)

            if len(self.SPLIT_ICM_DICT)>0:
                dataIO.save_data(data_dict_to_save = self.SPLIT_ICM_DICT,
                                 file_name = "split_ICM" + name_suffix)

                dataIO.save_data(data_dict_to_save = self.SPLIT_ICM_MAPPER_DICT,
                                 file_name = "split_ICM_mappers" + name_suffix)


            if len(self.SPLIT_UCM_DICT)>0:
                dataIO.save_data(data_dict_to_save = self.SPLIT_UCM_DICT,
                                 file_name = "split_UCM" + name_suffix)

                dataIO.save_data(data_dict_to_save = self.SPLIT_UCM_MAPPER_DICT,
                                 file_name = "split_UCM_mappers" + name_suffix)



    def _load_previously_built_split_and_attributes(self, save_folder_path):
        """
        Loads all URM and ICM
        :return:
        """

        if self.allow_cold_users:
            allow_cold_users_suffix = "allow_cold_users"
        else:
            allow_cold_users_suffix = "only_warm_users"

        if self.user_wise:
            user_wise_string = "user_wise"
        else:
            user_wise_string = "global_sample"


        name_suffix = "_{}_{}".format(allow_cold_users_suffix, user_wise_string)


        dataIO = DataIO(folder_path = save_folder_path)

        split_parameters_dict = dataIO.load_data(file_name ="split_parameters" + name_suffix)

        for attrib_name in split_parameters_dict.keys():
             self.__setattr__(attrib_name, split_parameters_dict[attrib_name])


        self.SPLIT_GLOBAL_MAPPER_DICT = dataIO.load_data(file_name ="split_mappers" + name_suffix)

        self.SPLIT_URM_DICT = dataIO.load_data(file_name ="split_URM" + name_suffix)

        if len(self.dataReader_object.get_loaded_ICM_names())>0:
            self.SPLIT_ICM_DICT = dataIO.load_data(file_name ="split_ICM" + name_suffix)

            self.SPLIT_ICM_MAPPER_DICT = dataIO.load_data(file_name ="split_ICM_mappers" + name_suffix)

        if len(self.dataReader_object.get_loaded_UCM_names())>0:
            self.SPLIT_UCM_DICT = dataIO.load_data(file_name ="split_UCM" + name_suffix)

            self.SPLIT_UCM_MAPPER_DICT = dataIO.load_data(file_name ="split_UCM_mappers" + name_suffix)




    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                DATA CONSISTENCY                                     ##########
    ##########                                                                                     ##########
    #########################################################################################################


    def _verify_data_consistency(self):

        self._assert_is_initialized()

        print_preamble = "{} consistency check: ".format(self.DATA_SPLITTER_NAME)

        assert len(self.SPLIT_URM_DICT) == len(self._SPLIT_URM_NAME_LIST),\
            print_preamble + "The available URM are not as many as they are supposed to be. URMs are {}, expected URMs are {}".format(len(self.SPLIT_URM_DICT), len(self._SPLIT_URM_NAME_LIST))

        assert all(URM_name in self.SPLIT_URM_DICT for URM_name in self._SPLIT_URM_NAME_LIST), print_preamble + "Not all URMs have been created"
        assert all(URM_name in self._SPLIT_URM_NAME_LIST for URM_name in self.SPLIT_URM_DICT.keys()), print_preamble + "The split contains URMs that should not exist"


        URM_shape = None

        for URM_name, URM_object in self.SPLIT_URM_DICT.items():

            if URM_shape is None:
                URM_shape = URM_object.shape

                n_users, n_items = URM_shape

                assert n_users != 0,  print_preamble + "Number of users in URM is 0"
                assert n_items != 0,  print_preamble + "Number of items in URM is 0"

            assert URM_shape == URM_object.shape,  print_preamble + "URM shape is inconsistent"


        assert self.SPLIT_URM_DICT["URM_train"].nnz != 0, print_preamble + "Number of interactions in URM Train is 0"
        assert self.SPLIT_URM_DICT["URM_test"].nnz != 0, print_preamble + "Number of interactions in URM Test is 0"

        # Assert URM_validation is not empty only when the input quota list is zero
        # It may create problems on the user-wise split if the validation quota is too small and no items gets selected
        # Although we assume in that case it would be acceptable to receive a warning your validation data cannot be built
        assert (self.SPLIT_URM_DICT["URM_validation"].nnz == 0 and self.input_split_interaction_quota_list[1] == 0.0)\
            or (self.SPLIT_URM_DICT["URM_validation"].nnz != 0 and self.input_split_interaction_quota_list[1] > 0.0)\
            , print_preamble + "Number of interactions in Validation is 0"

        quota_oscillation_allowed = 0.2

        for URM_index, URM_name in enumerate(self._SPLIT_URM_NAME_LIST):

            input_quota = self.input_split_interaction_quota_list[URM_index]
            actual_quota = self.actual_split_interaction_quota_list[URM_index]
            max_value_allowed = input_quota * (1 + quota_oscillation_allowed)
            min_value_allowed = input_quota * (1 - quota_oscillation_allowed)

            if actual_quota < min_value_allowed or actual_quota > max_value_allowed:
                print(print_preamble + "The differentce between the input interaction quota '{}' and actual interaction quota '{}' of '{}' higher than {} %".format(
                    input_quota, actual_quota, URM_name, quota_oscillation_allowed*100))


        URM = self.SPLIT_URM_DICT["URM_train"].copy()

        user_interactions = np.ediff1d(sps.csr_matrix(URM).indptr)

        if not self.allow_cold_users:
            assert np.all(user_interactions != 0), print_preamble + "Cold users exist despite not being allowed as per DataSplitter parameters, {} users out of {}".format(
                (user_interactions == 0).sum(), n_users)


        assert assert_disjoint_matrices(list(self.SPLIT_URM_DICT.values()))


        assert_URM_ICM_mapper_consistency(URM_DICT = self.SPLIT_URM_DICT,
                                          user_original_ID_to_index=self.SPLIT_GLOBAL_MAPPER_DICT["user_original_ID_to_index"],
                                          item_original_ID_to_index=self.SPLIT_GLOBAL_MAPPER_DICT["item_original_ID_to_index"],
                                          ICM_DICT = self.SPLIT_ICM_DICT,
                                          ICM_MAPPER_DICT = self.SPLIT_ICM_MAPPER_DICT,
                                          UCM_DICT = self.SPLIT_UCM_DICT,
                                          UCM_MAPPER_DICT = self.SPLIT_UCM_MAPPER_DICT,
                                          DATA_SPLITTER_NAME = self.DATA_SPLITTER_NAME)
