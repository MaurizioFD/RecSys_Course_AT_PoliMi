#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""

import pandas as pd
import zipfile, shutil
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL
from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.Movielens._utils_movielens_parser import _loadURM, _loadICM_genres_years


class Movielens1MReader(DataReader):

    DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    DATASET_SUBFOLDER = "Movielens1M/"
    AVAILABLE_URM = ["URM_all", "URM_timestamp"]
    AVAILABLE_ICM = ["ICM_genres", "ICM_year"]
    AVAILABLE_UCM = ["UCM_all"]

    IS_IMPLICIT = False

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original
        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "ml-1m.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find data zip file. Downloading...")

            download_from_URL(self.DATASET_URL, zipFile_path, "ml-1m.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "ml-1m.zip")


        ICM_genre_path = dataFile.extract("ml-1m/movies.dat", path=zipFile_path + "decompressed/")
        UCM_path = dataFile.extract("ml-1m/users.dat", path=zipFile_path + "decompressed/")
        URM_path = dataFile.extract("ml-1m/ratings.dat", path=zipFile_path + "decompressed/")

        self._print("Loading Interactions")
        URM_all_dataframe, URM_timestamp_dataframe = _loadURM(URM_path, header=None, separator='::')

        self._print("Loading Item Features genres")
        ICM_genres_dataframe, ICM_years_dataframe = _loadICM_genres_years(ICM_genre_path, header=None, separator='::', genresSeparator="|")

        self._print("Loading User Features")
        UCM_dataframe = pd.read_csv(filepath_or_buffer=UCM_path, sep="::", header=None, dtype={0:str, 1:str, 2:str, 3:str, 4:str}, engine='python')
        UCM_dataframe.columns = ["UserID", "gender", "age_group", "occupation", "zip_code"]

        # For each user a list of features
        UCM_list = [[feature_name + "_" + str(UCM_dataframe[feature_name][index]) for feature_name in ["gender", "age_group", "occupation", "zip_code"]] for index in range(len(UCM_dataframe))]
        UCM_dataframe = pd.DataFrame(UCM_list, index=UCM_dataframe["UserID"]).stack()
        UCM_dataframe = UCM_dataframe.reset_index()[[0, 'UserID']]
        UCM_dataframe.columns = ['FeatureID', 'UserID']
        UCM_dataframe["Data"] = 1


        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        dataset_manager.add_URM(URM_timestamp_dataframe, "URM_timestamp")
        dataset_manager.add_ICM(ICM_genres_dataframe, "ICM_genres")
        dataset_manager.add_ICM(ICM_years_dataframe, "ICM_year")
        dataset_manager.add_UCM(UCM_dataframe, "UCM_all")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)

        self._print("Cleaning Temporary Files")

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset

