#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/11/18

@author: Maurizio Ferrari Dacrema
"""

from Data_manager.DataReader import DataReader


class DataPostprocessing(DataReader):
    """
    This class provides the interface for the DataReaderPostprocessing objects
    """

    def __init__(self, dataReader_object):
        super(DataPostprocessing, self).__init__()

        self.dataReader_object = dataReader_object


    def get_loaded_ICM_names(self):
        return self.dataReader_object.get_loaded_ICM_names()


    def _get_dataset_name(self):
        return self.dataReader_object._get_dataset_name()


    def _get_dataset_name_root(self):
        return self.dataReader_object._get_dataset_name_root()


    def is_implicit(self):
        return self.dataReader_object.is_implicit()


    def _get_dataset_name_data_subfolder(self):
        """
        Returns the subfolder inside the dataset folder tree which contains the specific data to be loaded
        This method must be overridden by any data post processing object like k-cores / user sampling / interaction sampling etc
        to be applied before the data split

        :return: Dataset_name/
        """
        raise NotImplementedError("DataReaderPostprocessing: The following method was not implemented for the required class.")


    def _load_from_original_file(self):
        """
        _load_from_original_file will call the load of the dataset and then apply on it the postprocessing required
        :return:
        """
        raise NotImplementedError("DataReaderPostprocessing: The following method was not implemented for the required class.")
