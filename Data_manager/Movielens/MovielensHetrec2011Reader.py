#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/02/2019

@author: Maurizio Ferrari Dacrema
"""



import zipfile, shutil
import pandas as pd
from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.Dataset import Dataset
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL, load_CSV_into_SparseBuilder


class MovielensHetrec2011Reader(DataReader):

    DATASET_URL = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-movielens-2k-v2.zip"
    DATASET_SUBFOLDER = "MovielensHetrec2011/"
    AVAILABLE_ICM = []

    IS_IMPLICIT = False


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "hetrec2011-movielens-2k-v2.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find data zip file. Downloading...")

            download_from_URL(self.DATASET_URL, zipFile_path, "hetrec2011-movielens-2k-v2.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "hetrec2011-movielens-2k-v2.zip")


        URM_path = dataFile.extract("user_ratedmovies.dat", path=zipFile_path + "decompressed/")


        self._print("Loading Interactions")
        URM_all_dataframe = pd.read_csv(filepath_or_buffer=URM_path, sep="\t", header=0,
                                        dtype={0:str, 1:str, 2:float}, usecols=[0, 1, 2])
        URM_all_dataframe.columns = ["UserID", "ItemID", "Data"]


        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)


        self._print("Cleaning Temporary Files")

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset

