#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import zipfile, shutil


from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import downloadFromURL, merge_ICM
from Data_manager.Movielens20M.Movielens20MReader import _loadICM_genres, _loadURM_preinitialized_item_id, _loadICM_tags




class Movielens10MReader(DataReader):

    DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-10m.zip"
    DATASET_SUBFOLDER = "Movielens10M/"
    AVAILABLE_URM = ["URM_all", "URM_timestamp"]
    AVAILABLE_ICM = ["ICM_all", "ICM_genres", "ICM_tags"]
    DATASET_SPECIFIC_MAPPER = []

    IS_IMPLICIT = True



    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):

        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "ml-10m.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("Movielens10MReader: Unable to fild data zip file. Downloading...")


            downloadFromURL(self.DATASET_URL, zipFile_path, "ml-10m.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "ml-10m.zip")


        genres_path = dataFile.extract("ml-10M100K/movies.dat", path=zipFile_path + "decompressed/")
        tags_path = dataFile.extract("ml-10M100K/tags.dat", path=zipFile_path + "decompressed/")
        URM_path = dataFile.extract("ml-10M100K/ratings.dat", path=zipFile_path + "decompressed/")


        print("Movielens10MReader: loading genres")
        ICM_genres, tokenToFeatureMapper_ICM_genres, self.item_original_ID_to_index = _loadICM_genres(genres_path, header=True, separator='::', genresSeparator="|")

        self._LOADED_ICM_DICT["ICM_genres"] = ICM_genres
        self._LOADED_ICM_MAPPER_DICT["ICM_genres"] = tokenToFeatureMapper_ICM_genres

        print("Movielens10MReader: loading tags")
        ICM_tags, tokenToFeatureMapper_ICM_tags, _ = _loadICM_tags(tags_path, header=True, separator='::', if_new_item = "ignore",
                                                                             item_original_ID_to_index = self.item_original_ID_to_index)
        self._LOADED_ICM_DICT["ICM_tags"] = ICM_tags
        self._LOADED_ICM_MAPPER_DICT["ICM_tags"] = tokenToFeatureMapper_ICM_tags

        print("Movielens10MReader: loading URM")
        URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index, URM_timestamp = _loadURM_preinitialized_item_id(URM_path, separator="::",
                                                                                          header = False, if_new_user = "add", if_new_item = "ignore",
                                                                                          item_original_ID_to_index = self.item_original_ID_to_index)
        self._LOADED_URM_DICT["URM_all"] = URM_all
        self._LOADED_URM_DICT["URM_timestamp"] = URM_timestamp
        self._LOADED_GLOBAL_MAPPER_DICT["user_original_ID_to_index"] = self.user_original_ID_to_index
        self._LOADED_GLOBAL_MAPPER_DICT["item_original_ID_to_index"] = self.item_original_ID_to_index

        ICM_all, tokenToFeatureMapper_ICM_all = merge_ICM(ICM_genres, ICM_tags,
                                                          tokenToFeatureMapper_ICM_genres,
                                                          tokenToFeatureMapper_ICM_tags)

        self._LOADED_ICM_DICT["ICM_all"] = ICM_all
        self._LOADED_ICM_MAPPER_DICT["ICM_all"] = tokenToFeatureMapper_ICM_all



        print("Movielens10MReader: cleaning temporary files")

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        print("Movielens10MReader: loading complete")

