#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import zipfile, shutil
import pandas as pd
from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL
from Data_manager.Movielens._utils_movielens_parser import _loadICM_tags, _loadICM_genres_years, _loadURM


class Movielens20MReader(DataReader):

    DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
    DATASET_SUBFOLDER = "Movielens20M/"
    AVAILABLE_ICM = ["ICM_all", "ICM_genres", "ICM_tags", "ICM_year"]
    AVAILABLE_URM = ["URM_all", "URM_timestamp"]

    IS_IMPLICIT = False

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original

        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "ml-20m.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find data zip file. Downloading...")

            download_from_URL(self.DATASET_URL, zipFile_path, "ml-20m.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "ml-20m.zip")


        ICM_genre_path = dataFile.extract("ml-20m/movies.csv", path=zipFile_path + "decompressed/")
        ICM_tags_path = dataFile.extract("ml-20m/tags.csv", path=zipFile_path + "decompressed/")
        URM_path = dataFile.extract("ml-20m/ratings.csv", path=zipFile_path + "decompressed/")


        self._print("Loading Item Features Genres")
        ICM_genres_dataframe, ICM_years_dataframe = _loadICM_genres_years(ICM_genre_path, header=0, separator=',', genresSeparator="|")

        self._print("Loading Item Features Tags")
        ICM_tags_dataframe = _loadICM_tags(ICM_tags_path, header=0, separator=',')

        ICM_all_dataframe = pd.concat([ICM_genres_dataframe, ICM_tags_dataframe])

        self._print("Loading Interactions")
        URM_all_dataframe, URM_timestamp_dataframe = _loadURM(URM_path, header=0, separator=',')

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        dataset_manager.add_URM(URM_timestamp_dataframe, "URM_timestamp")
        dataset_manager.add_ICM(ICM_genres_dataframe, "ICM_genres")
        dataset_manager.add_ICM(ICM_years_dataframe, "ICM_year")
        dataset_manager.add_ICM(ICM_tags_dataframe, "ICM_tags")
        dataset_manager.add_ICM(ICM_all_dataframe, "ICM_all")


        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)



        self._print("Cleaning Temporary Files")

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        self._print("saving URM and ICM")

        return loaded_dataset








