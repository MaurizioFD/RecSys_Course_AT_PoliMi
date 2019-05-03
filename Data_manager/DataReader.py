#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 01/01/2018

@author: Maurizio Ferrari Dacrema
"""


import scipy.sparse as sps
import numpy as np
import os, traceback
from Base.DataIO import DataIO
from Data_manager.data_consistency_check import assert_URM_ICM_mapper_consistency


def gini_index(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = np.array(array, dtype=np.float)
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient




#################################################################################################################
#############################
#############################               DATA READER
#############################
#################################################################################################################



class DataReader(object):
    """
    Abstract class for the DataReaders, each shoud be implemented for a specific dataset
    DataReader has the following functions:
     - It loads the data of the original dataset and saves it into sparse matrices
     - It exposes the following functions
        - load_data(save_folder_path = None)        loads the data and saves into the specified folder, if None uses default, if False des not save
        - get_URM_all()                             returns a copy of the whole URM
        - get_ICM_from_name(ICM_name)               returns a copy of the specified ICM
        - get_loaded_ICM_names()                    returns a copy of the loaded ICM names, which can be used in get_ICM_from_name
        - get_loaded_ICM_dict()                     returns a copy of the loaded ICM in a dictionary [ICM_name]->ICM_sparse
        - DATASET_SUBFOLDER_DEFAULT                 path of the data folder
        - item_original_ID_to_index
        - user_original_ID_to_index

    """
    DATASET_SPLIT_ROOT_FOLDER = "Data_manager_split_datasets/"
    DATASET_OFFLINE_ROOT_FOLDER = "Data_manager_offline_datasets/"


    # This subfolder contains the preprocessed data, already loaded from the original data file
    DATASET_SUBFOLDER_ORIGINAL = "original/"

    # Available URM split
    AVAILABLE_URM = ["URM_all"]

    # Available ICM for the given dataset, there might be no ICM, one or many
    AVAILABLE_ICM = []

    # Mappers existing for all datasets, associating USER_ID and ITEM_ID to the new designation
    GLOBAL_MAPPER = ["item_original_ID_to_index", "user_original_ID_to_index"]

    # Mappers specific for a given dataset, they might be related to more complex data structures or FEATURE_TOKENs
    DATASET_SPECIFIC_MAPPER = []

    # This flag specifies if the given dataset contains implicit preferences or explicit ratings
    IS_IMPLICIT = False

    _DATA_READER_NAME = "DataReader"

    # Dictionary for data matrices
    _LOADED_URM_DICT = None
    _LOADED_ICM_DICT = None
    _LOADED_ICM_MAPPER_DICT = None
    _LOADED_GLOBAL_MAPPER_DICT = None

    def __init__(self, reload_from_original_data = False, ICM_to_load_list = None):

        super(DataReader, self).__init__()

        self.reload_from_original_data = reload_from_original_data
        if self.reload_from_original_data:
            self._print("reload_from_original_data is True, previously loaded data will be ignored")

        if ICM_to_load_list is None:
            self.ICM_to_load_list = self.AVAILABLE_ICM.copy()
        else:
            assert all([ICM_to_load in self.AVAILABLE_ICM for ICM_to_load in ICM_to_load_list]), \
                "{}: ICM_to_load_list contains ICM names which are not available for the current DataReader".format(self._DATA_READER_NAME)
            self.ICM_to_load_list = ICM_to_load_list.copy()

    def is_implicit(self):
        return self.IS_IMPLICIT

    def _print(self, message):
        print("{}: {}".format(self._DATA_READER_NAME, message))

    def _assert_is_initialized(self):
         assert self._LOADED_URM_DICT is not None, "DataReader {}: Unable to load data split. The split has not been generated yet, call the load_data function to do so.".format(self._get_dataset_name())

    def _get_dataset_name(self):
        return self._get_dataset_name_root()[:-1]

    def get_ICM_from_name(self, ICM_name):
        self._assert_is_initialized()
        return self._LOADED_ICM_DICT[ICM_name].copy()

    def get_URM_from_name(self, URM_name):
        self._assert_is_initialized()
        return self._LOADED_URM_DICT[URM_name].copy()

    def get_ICM_feature_to_index_mapper_from_name(self, ICM_name):
        self._assert_is_initialized()
        return self._LOADED_ICM_MAPPER_DICT[ICM_name].copy()

    def get_loaded_ICM_names(self):
        return self.ICM_to_load_list.copy()

    def get_all_available_ICM_names(self):
        return self.AVAILABLE_ICM.copy()

    def get_loaded_URM_names(self):
        return self.AVAILABLE_URM.copy()

    def get_item_original_ID_to_index_mapper(self):
        return self._LOADED_GLOBAL_MAPPER_DICT["item_original_ID_to_index"].copy()

    def get_user_original_ID_to_index_mapper(self):
        return self._LOADED_GLOBAL_MAPPER_DICT["user_original_ID_to_index"].copy()

    def get_loaded_Global_mappers(self):

        global_mappers_dict = {}

        for mapper_name, mapper_object in self._LOADED_GLOBAL_MAPPER_DICT.items():

            global_mappers_dict[mapper_name] = mapper_object.copy()

        return global_mappers_dict


    def get_loaded_ICM_dict(self):

        ICM_dict = {}

        for ICM_name in self.get_loaded_ICM_names():

            ICM_dict[ICM_name] = self.get_ICM_from_name(ICM_name)

        return ICM_dict

    def get_loaded_URM_dict(self):

        URM_dict = {}

        for URM_name in self.get_loaded_URM_names():

            URM_dict[URM_name] = self.get_URM_from_name(URM_name)


        return URM_dict

    def get_URM_all(self):
        return self.get_URM_from_name("URM_all")


    def _load_from_original_file(self):

        raise NotImplementedError("{}: _load_from_original_file was not implemented for the required dataset. Impossible to load the data".format(self._DATA_READER_NAME))


    def _get_dataset_name_root(self):
        """
        Returns the root of the folder tree which contains all of the dataset data/splits and files

        :return: Dataset_name/
        """
        raise NotImplementedError("{}:_get_dataset_name_root was not implemented for the required dataset. Impossible to load the data".format(self._DATA_READER_NAME))




    def _get_dataset_name_data_subfolder(self):
        """
        Returns the subfolder inside the dataset folder tree which contains the specific data to be loaded
        This method must be overridden by any data post processing object like k-cores / user sampling / interaction sampling etc
        to be applied before the data split

        :return: original or k_cores etc...
        """
        return self.DATASET_SUBFOLDER_ORIGINAL


    def load_data(self, save_folder_path = None):
        """

        :param save_folder_path:    path in which to save the loaded dataset
                                    None    use default "dataset_name/original/"
                                    False   do not save
        :return:
        """

        # Setting the empty dictionary here
        # This avoids a static field which would be common to all DataReader object active
        self._LOADED_URM_DICT = {}
        self._LOADED_ICM_DICT = {}
        self._LOADED_ICM_MAPPER_DICT = {}
        self._LOADED_GLOBAL_MAPPER_DICT = {}

        # Use default e.g., "dataset_name/original/"
        if save_folder_path is None:
            save_folder_path = self.DATASET_SPLIT_ROOT_FOLDER + self._get_dataset_name_root() + self._get_dataset_name_data_subfolder()


        # If save_folder_path contains any path try to load a previously built split from it
        if save_folder_path is not False and not self.reload_from_original_data:

            try:

                self._load_from_saved_sparse_matrix(save_folder_path)

                self._print("Verifying data consistency...")
                self._verify_data_consistency()
                self._print("Verifying data consistency... Passed!")

                self.print_statistics()
                return

            except FileNotFoundError:

                self._print("Preloaded data not found, reading from original files...")

            except Exception:

                self._print("Reading split from {} caused the following exception...".format(save_folder_path))
                traceback.print_exc()
                raise Exception("{}: Exception while reading split".format(self._get_dataset_name()))


        self._load_from_original_file()

        self._print("Verifying data consistency...")
        self._verify_data_consistency()
        self._print("Verifying data consistency... Passed!")

        if save_folder_path not in [False]:

            # If directory does not exist, create
            if not os.path.exists(save_folder_path):
                self._print("Creating folder '{}'".format(save_folder_path))
                os.makedirs(save_folder_path)

            else:
                self._print("Found already existing folder '{}'".format(save_folder_path))

            self._save_dataset(save_folder_path)

            self._print("Saving complete!")

        self.print_statistics()


    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                LOAD AND SAVE                                        ##########
    ##########                                                                                     ##########
    #########################################################################################################


    def _save_dataset(self, save_folder_path):

        dataIO = DataIO(folder_path = save_folder_path)

        dataIO.save_data(data_dict_to_save = self._LOADED_GLOBAL_MAPPER_DICT,
                         file_name = "dataset_global_mappers")

        dataIO.save_data(data_dict_to_save = self._LOADED_URM_DICT,
                         file_name = "dataset_URM")

        if len(self.get_loaded_ICM_names()) > 0:
            dataIO.save_data(data_dict_to_save = self._LOADED_ICM_DICT,
                             file_name = "dataset_ICM")

            dataIO.save_data(data_dict_to_save = self._LOADED_ICM_MAPPER_DICT,
                             file_name = "dataset_ICM_mappers")




    def _load_from_saved_sparse_matrix(self, save_folder_path):
        """
        Loads all URM and ICM
        :return:
        """

        dataIO = DataIO(folder_path = save_folder_path)

        self._LOADED_GLOBAL_MAPPER_DICT = dataIO.load_data(file_name ="dataset_global_mappers")

        self._LOADED_URM_DICT = dataIO.load_data(file_name ="dataset_URM")

        if len(self.get_loaded_ICM_names()) > 0:
            self._LOADED_ICM_DICT = dataIO.load_data(file_name ="dataset_ICM")

            self._LOADED_ICM_MAPPER_DICT = dataIO.load_data(file_name ="dataset_ICM_mappers")


    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                DATASET STATISTICS                                   ##########
    ##########                                                                                     ##########
    #########################################################################################################


    def print_statistics(self):

        self._assert_is_initialized()

        URM_all = self.get_URM_all()

        n_users, n_items = URM_all.shape

        n_interactions = URM_all.nnz


        URM_all = sps.csr_matrix(URM_all)
        user_profile_length = np.ediff1d(URM_all.indptr)

        max_interactions_per_user = user_profile_length.max()
        avg_interactions_per_user = n_interactions/n_users
        min_interactions_per_user = user_profile_length.min()

        URM_all = sps.csc_matrix(URM_all)
        item_profile_length = np.ediff1d(URM_all.indptr)

        max_interactions_per_item = item_profile_length.max()
        avg_interactions_per_item = n_interactions/n_items
        min_interactions_per_item = item_profile_length.min()


        print("DataReader: current dataset is: {}\n"
              "\tNumber of items: {}\n"
              "\tNumber of users: {}\n"
              "\tNumber of interactions in URM_all: {}\n"
              "\tInteraction density: {:.2E}\n"
              "\tInteractions per user:\n"
              "\t\t Min: {:.2E}\n"
              "\t\t Avg: {:.2E}\n"    
              "\t\t Max: {:.2E}\n"     
              "\tInteractions per item:\n"    
              "\t\t Min: {:.2E}\n"
              "\t\t Avg: {:.2E}\n"    
              "\t\t Max: {:.2E}\n"
              "\tGini Index: {:.2f}\n".format(
            self.__class__,
            n_items,
            n_users,
            n_interactions,
            n_interactions/(n_items*n_users),
            min_interactions_per_user,
            avg_interactions_per_user,
            max_interactions_per_user,
            min_interactions_per_item,
            avg_interactions_per_item,
            max_interactions_per_item,
            gini_index(user_profile_length),
        ))



    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                DATA CONSISTENCY                                     ##########
    ##########                                                                                     ##########
    #########################################################################################################


    def _verify_data_consistency(self):

        self._assert_is_initialized()

        print_preamble = "{} consistency check: ".format(self._get_dataset_name())

        URM_all = self.get_URM_all()
        n_interactions = URM_all.nnz

        assert n_interactions != 0, print_preamble + "Number of interactions in URM is 0"

        assert all(loaded_ICM_name in self._LOADED_ICM_DICT for loaded_ICM_name in self.get_loaded_ICM_names()), \
            print_preamble + "The DataReader has not loaded all the ICMs it was supposed to load."

        assert all(loaded_ICM_name in self.get_loaded_ICM_names() for loaded_ICM_name in self._LOADED_ICM_DICT), \
            print_preamble + "The DataReader has loaded an ICM which was not supposed to load"

        assert_URM_ICM_mapper_consistency(URM_DICT = self.get_loaded_URM_dict(),
                                          GLOBAL_MAPPER_DICT = self.get_loaded_Global_mappers(),
                                          ICM_DICT = self.get_loaded_ICM_dict(),
                                          ICM_MAPPER_DICT = self._LOADED_ICM_MAPPER_DICT,
                                          DATA_SPLITTER_NAME = self._get_dataset_name())
