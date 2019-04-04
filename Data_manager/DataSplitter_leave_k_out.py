#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/01/18

@author: Maurizio Ferrari Dacrema
"""

import scipy.sparse as sps
import numpy as np
import pickle

from Data_manager.DataSplitter import DataSplitter

from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix


class DataSplitter_leave_k_out(DataSplitter):
    """
    The splitter tries to load from the specific folder related to a dataset, a split in the format corresponding to
    the splitter class. Basically each split is in a different subfolder
    - The "original" subfolder contains the whole dataset, is composed by a single URM with all data and may contain
        ICMs as well, either one or many, depending on the dataset
    - The other subfolders "warm", "cold" ecc contains the splitted data.

    The dataReader class involvement is limited to the following cased:
    - At first the dataSplitter tries to load from the subfolder corresponding to that split. Say "warm"
    - If the dataReader is succesful in loading the files, then a split already exists and the loading is complete
    - If the dataReader raises a FileNotFoundException, then no split is available.
    - The dataSplitter then creates a new instance of dataReader using default parameters, so that the original data will be loaded
    - At this point the chosen dataSplitter takes the URM_all and selected ICM to perform the split
    - The dataSplitter saves the splitted data in the appropriate subfolder.
    - Finally, the dataReader is instantiated again with the correct parameters, to load the data just saved
    """

    """
     - It exposes the following functions
        - load_data(save_folder_path = None, force_new_split = False)   loads the data or creates a new split
    
    
    """


    def __init__(self, dataReader_object, k_value = 1, forbid_new_split = False, force_new_split = False, validation_set = True, leave_last_out = False):
        """

        :param dataReader_object:
        :param n_folds:
        :param force_new_split:
        :param forbid_new_split:
        :param save_folder_path:    path in which to save the loaded dataset
                                    None    use default "dataset_name/split_name/"
                                    False   do not save
        """


        assert k_value>=1, "DataSplitter_leave_k_out: k_value must be  greater or equal than 1"

        self.k_value = k_value
        self.validation_set = validation_set
        self.allow_cold_users = False
        self.leave_last_out = leave_last_out

        print("DataSplitter_leave_k_out: Cold users not allowed")

        super(DataSplitter_leave_k_out, self).__init__(dataReader_object, forbid_new_split=forbid_new_split, force_new_split=force_new_split)



    def _get_split_subfolder_name(self):
        """

        :return: warm_{n_folds}_fold/
        """

        if self.leave_last_out:
            order_suffix = "last"
        else:
            order_suffix = "random"


        return "leave_{}_out_{}/".format(self.k_value, order_suffix)


    def get_statistics_URM(self):

        # This avoids the fixed bit representation of numpy preventing
        # an overflow when computing the product
        n_items = int(self.n_items)
        n_users = int(self.n_users)

        print("DataSplitter_k_fold for DataReader: {}\n"
              "\t Num items: {}\n"
              "\t Num users: {}\n".format(self.dataReader_object._get_dataset_name(), n_items, n_users),
              "\t Train interactions: {}, density: {:.2E}\n".format(self.URM_train.nnz, self.URM_train.nnz/(int(n_items)*int(n_users))),
              "\t Test interactions: {}, density: {:.2E}".format(self.URM_test.nnz, self.URM_test.nnz/(int(n_items)*int(n_users))))

        if self.validation_set:
            print("\t Validation interactions: {}, density: {:.2E}\n".format(self.URM_validation.nnz, self.URM_validation.nnz/(int(n_items)*int(n_users))))

        print("\n")







    def get_statistics_ICM(self):

        for ICM_name in self.dataReader_object.get_loaded_ICM_names():

            ICM_object = getattr(self, ICM_name)
            n_items = ICM_object.shape[0]
            n_features = ICM_object.shape[1]

            print("\t Statistics for {}: n_features {}, feature occurrences {}, density: {:.2E}".format(
                ICM_name, n_features, ICM_object.nnz, ICM_object.nnz/(int(n_items)*int(n_features))
            ))

        print("\n")





    def get_holdout_split(self):
        """
        The train set is defined as all data except the one of that fold, which is the test
        :param n_fold:
        :return: URM_train, URM_validation, URM_test
        """

        return self.URM_train.copy(), self.URM_validation.copy(), self.URM_test.copy()





    def _split_data_from_original_dataset(self, save_folder_path):


        self.dataReader_object.load_data()
        URM = self.dataReader_object.get_URM_all()
        URM = sps.csr_matrix(URM)

        split_number = 2
        if self.validation_set:
            split_number+=1

        # Min interactions at least self.k_value for each split +1 for train and validation
        min_user_interactions = (split_number -1)*self.k_value +1


        if not self.allow_cold_users:
            user_interactions = np.ediff1d(URM.indptr)
            user_to_preserve = user_interactions >= min_user_interactions

            print("DataSplitter_leave_k_out: Removing {} ({:.2f} %) of {} users because they have less than the {} interactions required for {} splits".format(
                 URM.shape[0] - user_to_preserve.sum(), (1-user_to_preserve.sum()/URM.shape[0])*100, URM.shape[0], min_user_interactions, split_number))

            URM = URM[user_to_preserve,:]


        self.n_users, self.n_items = URM.shape

        URM = sps.csr_matrix(URM)



        URM_train_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows = self.n_users,
                                            auto_create_col_mapper=False, n_cols = self.n_items)

        URM_test_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows = self.n_users,
                                            auto_create_col_mapper=False, n_cols = self.n_items)

        if self.validation_set:
             URM_validation_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows = self.n_users,
                                                              auto_create_col_mapper=False, n_cols = self.n_items)



        for user_id in range(self.n_users):

            start_user_position = URM.indptr[user_id]
            end_user_position = URM.indptr[user_id+1]

            user_profile = URM.indices[start_user_position:end_user_position]


            if not self.leave_last_out:
                indices_to_suffle = np.arange(len(user_profile), dtype=np.int)

                np.random.shuffle(indices_to_suffle)

                user_interaction_items = user_profile[indices_to_suffle]
                user_interaction_data = URM.data[start_user_position:end_user_position][indices_to_suffle]

            else:

                # The first will be sampled so the last interaction must be the first one
                interaction_position = URM.data[start_user_position:end_user_position]

                sort_interaction_index = np.argsort(-interaction_position)

                user_interaction_items = user_profile[sort_interaction_index]
                user_interaction_data = URM.data[start_user_position:end_user_position][sort_interaction_index]






            #Test interactions
            user_interaction_items_test = user_interaction_items[0:self.k_value]
            user_interaction_data_test = user_interaction_data[0:self.k_value]

            URM_test_builder.add_data_lists([user_id]*self.k_value, user_interaction_items_test, user_interaction_data_test)


            #validation interactions
            if self.validation_set:
                user_interaction_items_validation = user_interaction_items[self.k_value:self.k_value*2]
                user_interaction_data_validation = user_interaction_data[self.k_value:self.k_value*2]

                URM_validation_builder.add_data_lists([user_id]*self.k_value, user_interaction_items_validation, user_interaction_data_validation)


            #Train interactions
            user_interaction_items_train = user_interaction_items[self.k_value*2:]
            user_interaction_data_train = user_interaction_data[self.k_value*2:]

            URM_train_builder.add_data_lists([user_id]*len(user_interaction_items_train), user_interaction_items_train, user_interaction_data_train)



        self.URM_train = URM_train_builder.get_SparseMatrix()
        self.URM_test = URM_test_builder.get_SparseMatrix()


        dict_to_save = {"k_value": self.k_value,
                         "n_items": self.n_items,
                         "n_users": self.n_users,
                         "allow_cold_users": self.allow_cold_users
                         }


        if self.validation_set:
            self.URM_validation = URM_validation_builder.get_SparseMatrix()
            validation_set_suffix = "validation_set"
        else:
            validation_set_suffix = "no_validation_set"


        if self.allow_cold_users:
            allow_cold_users_suffix = "allow_cold_users"
        else:
            allow_cold_users_suffix = "only_warm_users"


        sps.save_npz(save_folder_path + "{}_{}_{}".format("URM_train", allow_cold_users_suffix, validation_set_suffix), self.URM_train)
        sps.save_npz(save_folder_path + "{}_{}_{}".format("URM_test", allow_cold_users_suffix, validation_set_suffix), self.URM_test)

        if self.validation_set:
            sps.save_npz(save_folder_path + "{}_{}_{}".format("URM_validation", allow_cold_users_suffix, validation_set_suffix), self.URM_validation)


        pickle.dump(dict_to_save,
                    open(save_folder_path + "split_parameters_{}_{}".format(allow_cold_users_suffix, validation_set_suffix), "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)



        for ICM_name in self.dataReader_object.get_loaded_ICM_names():

            ICM_object = self.dataReader_object.get_ICM_from_name(ICM_name)

            sps.save_npz(save_folder_path + "{}".format(ICM_name), ICM_object)

            pickle.dump(self.dataReader_object.get_ICM_feature_to_index_mapper_from_name(ICM_name),
                        open(save_folder_path + "tokenToFeatureMapper_{}".format(ICM_name), "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)

        print("DataSplitter: Split complete")





    def _load_previously_built_split_and_attributes(self, save_folder_path):
        """
        Loads all URM and ICM
        :return:
        """
        if self.validation_set:
            validation_set_suffix = "validation_set"
        else:
            validation_set_suffix = "no_validation_set"

        if self.allow_cold_users:
            allow_cold_users_suffix = "allow_cold_users"
        else:
            allow_cold_users_suffix = "only_warm_users"


        data_dict = pickle.load(open(save_folder_path + "split_parameters_{}_{}".format(allow_cold_users_suffix, validation_set_suffix), "rb"))

        for attrib_name in data_dict.keys():
             self.__setattr__(attrib_name, data_dict[attrib_name])


        for ICM_name in self.dataReader_object.get_loaded_ICM_names():

            ICM_object = sps.load_npz(save_folder_path + "{}.npz".format(ICM_name))
            self.__setattr__(ICM_name, ICM_object)

            pickle.load(open(save_folder_path + "tokenToFeatureMapper_{}".format(ICM_name), "rb"))
            self.__setattr__("tokenToFeatureMapper_{}".format(ICM_name), ICM_object)


        self.URM_train = sps.load_npz(save_folder_path + "{}_{}_{}.npz".format("URM_train", allow_cold_users_suffix, validation_set_suffix))
        self.URM_test = sps.load_npz(save_folder_path + "{}_{}_{}.npz".format("URM_test", allow_cold_users_suffix, validation_set_suffix))

        if self.validation_set:
            self.URM_validation = sps.load_npz(save_folder_path + "{}_{}_{}.npz".format("URM_validation", allow_cold_users_suffix, validation_set_suffix))

