#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/11/18

@author: Maurizio Ferrari Dacrema
"""

import traceback, os


class DataSplitter(object):
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
    DATASET_SPLIT_ROOT_FOLDER = "Data_manager_split_datasets/"
    ICM_SPLIT_SUFFIX = [""]


    def __init__(self, dataReader_object, forbid_new_split = False, force_new_split = False):
        """

        :param dataReader_object:
        :param n_folds:
        :param force_new_split:
        :param forbid_new_split:
        :param save_folder_path:    path in which to save the loaded dataset
                                    None    use default "dataset_name/split_name/"
                                    False   do not save
        """

        super(DataSplitter, self).__init__()

        self.dataReader_object = dataReader_object
        self.forbid_new_split = forbid_new_split
        self.force_new_split = force_new_split


    def get_dataReader_object(self):
        return self.dataReader_object

    # Allow to use ICM functions on the DataSplitter
    def _get_dataset_name(self):
        return self.get_dataReader_object()._get_dataset_name()

    def get_ICM_from_name(self, ICM_name):
        return getattr(self, ICM_name).copy()

    def get_loaded_ICM_names(self):
        return self.get_dataReader_object().get_loaded_ICM_names()

    def get_all_available_ICM_names(self):
        return self.get_dataReader_object().get_all_available_ICM_names().copy()


    def get_loaded_ICM_dict(self):
        return self.get_dataReader_object().get_loaded_ICM_dict()





    def load_data(self, save_folder_path = None):

        # Use default "dataset_name/split_name/original" or "dataset_name/split_name/k-cores"
        if save_folder_path is None:
            save_folder_path = self.DATASET_SPLIT_ROOT_FOLDER + \
                               self.dataReader_object._get_dataset_name_root() + \
                               self._get_split_subfolder_name() + \
                               self.dataReader_object._get_dataset_name_data_subfolder()


        # If save_folder_path contains any path try to load a previously built split from it
        if save_folder_path is not False and not self.force_new_split:

            try:

                self._load_previously_built_split_and_attributes(save_folder_path)

            except FileNotFoundError:

                # Split not found, either stop or create a new one
                if self.forbid_new_split:
                    raise ValueError("DataSplitter_k_fold: Preloaded data not found, but creating a new split is forbidden. Terminating")

                else:
                    print("DataSplitter_k_fold: Preloaded data not found, reading from original files...")

                    # If directory does not exist, create
                    if not os.path.exists(save_folder_path):
                        os.makedirs(save_folder_path)

                    self._split_data_from_original_dataset(save_folder_path)
                    self._load_previously_built_split_and_attributes(save_folder_path)

                    print("DataSplitter_k_fold: Preloaded data not found, reading from original files... Done")


            except Exception:

                print("DataSplitter_k_fold: Reading split from {} caused the following exception...".format(save_folder_path))
                traceback.print_exc()
                raise Exception("DataSplitter_k_fold: Exception while reading split")



        self.get_statistics_URM()
        self.get_statistics_ICM()

        print("DataSplitter_k_fold: Done.")



    def _get_split_subfolder_name(self):
        """
        :return: Dataset_name/split_name/
        """
        raise NotImplementedError("DataReader: The following method was not implemented for the required dataset. Impossible to load the data")




    def _split_data_from_original_dataset(self, save_folder_path):
        raise NotImplementedError("DataSplitter: _split_data_from_original_dataset not implemented")




    def _load_previously_built_split_and_attributes(self, save_folder_path):
        """
        Loads all URM and ICM
        :return:
        """
        raise NotImplementedError("DataSplitter: _split_data_from_original_dataset not implemented")




    def get_statistics_URM(self):

        raise NotImplementedError("DataSplitter: _split_data_from_original_dataset not implemented")







    def get_statistics_ICM(self):

        raise NotImplementedError("DataSplitter: _split_data_from_original_dataset not implemented")

