#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26/11/19

@author: Maurizio Ferrari Dacrema
"""

import pandas as pd


def _loadICM_genres_years(genres_path, header=True, separator=',', genresSeparator="|"):

    ICM_genres_dataframe = pd.read_csv(filepath_or_buffer=genres_path, sep=separator, header=header, dtype={0:str, 1:str, 2:str}, engine='python')
    ICM_genres_dataframe.columns = ["ItemID", "Title", "GenreList"]

    ICM_years_dataframe = ICM_genres_dataframe.copy()
    ICM_years_dataframe["Year"] = ICM_years_dataframe["Title"].str.extract(pat='\(([0-9]{4})\)$')
    ICM_years_dataframe = ICM_years_dataframe[ICM_years_dataframe["Year"].notnull()]
    ICM_years_dataframe["Year"] = ICM_years_dataframe["Year"].astype(int)
    ICM_years_dataframe = ICM_years_dataframe[['ItemID', 'Year']]
    ICM_years_dataframe.rename(columns={'Year': 'Data'}, inplace=True)
    ICM_years_dataframe["FeatureID"] = "Year"


    # Split GenreList in order to obtain a dataframe with a tag per row
    ICM_genres_dataframe = pd.DataFrame(ICM_genres_dataframe["GenreList"].str.split(genresSeparator).tolist(),
                                        index=ICM_genres_dataframe["ItemID"]).stack()

    ICM_genres_dataframe = ICM_genres_dataframe.reset_index()[[0, 'ItemID']]
    ICM_genres_dataframe.columns = ['FeatureID', 'ItemID']
    ICM_genres_dataframe = ICM_genres_dataframe[['ItemID', 'FeatureID']]
    ICM_genres_dataframe["Data"] = 1

    return ICM_genres_dataframe, ICM_years_dataframe



def _loadURM(URM_path, header=None, separator=','):

    URM_all_dataframe = pd.read_csv(filepath_or_buffer=URM_path, sep=separator, header=header, dtype={0:str, 1:str, 2:float, 3:int}, engine='python')
    URM_all_dataframe.columns = ["UserID", "ItemID", "Interaction", "Timestamp"]

    URM_timestamp_dataframe = URM_all_dataframe.copy().drop(columns=["Interaction"])
    URM_all_dataframe = URM_all_dataframe.drop(columns=["Timestamp"])
    URM_timestamp_dataframe.columns = ["UserID", "ItemID", "Data"]
    URM_all_dataframe.columns = ["UserID", "ItemID", "Data"]

    return URM_all_dataframe, URM_timestamp_dataframe





def _loadICM_tags(tags_path, header=True, separator=','):

    # Tags
    from Data_manager.TagPreprocessing import tagFilterAndStemming

    fileHandle = open(tags_path, "r", encoding="latin1")

    if header is not None:
        fileHandle.readline()

    movie_id_list = []
    tags_lists = []

    for index, line in enumerate(fileHandle):

        if index % 100000 == 0 and index>0:
            print("Processed {} rows".format(index))

        if (len(line)) > 1:
            line = line.split(separator)

            line[-1] = line[-1].replace("\n", "")

            # If a movie has no genre, ignore it
            movie_id = line[1]
            this_tag_list = line[2]

            # Remove non alphabetical character and split on spaces
            this_tag_list = tagFilterAndStemming(this_tag_list)

            movie_id_list.append(movie_id)
            tags_lists.append(this_tag_list)

    fileHandle.close()

    ICM_dataframe = pd.DataFrame(tags_lists, index=movie_id_list).stack()
    ICM_dataframe = ICM_dataframe.reset_index()[["level_0", 0]]
    ICM_dataframe.columns = ['ItemID', 'FeatureID']
    ICM_dataframe["Data"] = 1


    return ICM_dataframe





def _loadUCM(UCM_path, header=True, separator=','):

    # Genres
    from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

    ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = None, on_new_col = "add",
                                                    preinitialized_row_mapper = None, on_new_row = "add")


    fileHandle = open(UCM_path, "r", encoding="latin1")
    numCells = 0

    if header:
        fileHandle.readline()

    for line in fileHandle:
        numCells += 1
        if (numCells % 1000000 == 0):
            print("Processed {} rows".format(numCells))

        if (len(line)) > 1:
            line = line.split(separator)

            line[-1] = line[-1].replace("\n", "")

            user_id = line[0]

            token_list = []
            token_list.append("gender_" + str(line[1]))
            token_list.append("age_group_" + str(line[2]))
            token_list.append("occupation_" + str(line[3]))
            token_list.append("zip_code_" + str(line[4]))

            # Rows movie ID
            # Cols features
            ICM_builder.add_single_row(user_id, token_list, data = 1.0)


    fileHandle.close()

    return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()






