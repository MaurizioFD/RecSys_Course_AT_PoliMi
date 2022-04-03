#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 27/04/2019

@author: Maurizio Ferrari Dacrema
"""

import os, json, zipfile, shutil, platform, warnings

import scipy.sparse as sps
from pandas import DataFrame
import pandas as pd
import numpy as np


def json_not_serializable_handler(o):
    """
    Json cannot serialize automatically some data types, for example numpy integers (int32).
    This may be a limitation of numpy-json interfaces for Python 3.6 and may not occur in Python 3.7
    :param o:
    :return:
    """

    if isinstance(o, np.integer):
        return int(o)

    if isinstance(o, np.bool_):
        return bool(o)

    raise TypeError("json_not_serializable_handler: object '{}' is not serializable.".format(type(o)))



class DataIO(object):
    """ DataIO"""

    _DEFAULT_TEMP_FOLDER = ".temp"

    # _MAX_PATH_LENGTH_LINUX = 4096
    _MAX_PATH_LENGTH_WINDOWS = 255

    def __init__(self, folder_path):
        super(DataIO, self).__init__()

        self._is_windows = platform.system() == "Windows"

        self.folder_path = folder_path if folder_path[-1] == "/" else folder_path + "/"
        self._key_string_alert_done = False

        # if self._is_windows:
        #     self.folder_path = "\\\\?\\" + self.folder_path


    def _print(self, message):
        print("{}: {}".format("DataIO", message))


    def _get_temp_folder(self, file_name):
        """
        Creates a temporary folder to be used during the data saving
        :return:
        """

        # Ignore the .zip extension
        file_name = file_name[:-4]
        current_temp_folder = "{}{}_{}_{}/".format(self.folder_path, self._DEFAULT_TEMP_FOLDER, os.getpid(), file_name)

        if os.path.exists(current_temp_folder):
            self._print("Folder {} already exists, could be the result of a previous failed save attempt or multiple saver are active in parallel. " \
            "Folder will be removed.".format(current_temp_folder))

            shutil.rmtree(current_temp_folder, ignore_errors=True)

        os.makedirs(current_temp_folder)

        return current_temp_folder


    def _check_dict_key_type(self, dict_to_save):
        """
        Check whether the keys of the dictionary are string. If not, transforms them into strings
        :param dict_to_save:
        :return:
        """

        all_keys_are_str = all(isinstance(key, str) for key in dict_to_save.keys())

        if all_keys_are_str:
            return dict_to_save

        if not self._key_string_alert_done:
            self._print("Json dumps supports only 'str' as dictionary keys. Transforming keys to string, note that this will alter the mapper content.")
            self._key_string_alert_done = True

        dict_to_save_key_str = {str(key):val for (key,val) in dict_to_save.items()}

        assert len(dict_to_save_key_str) == len(dict_to_save), \
            "DataIO: Transforming dictionary keys into strings altered its content. Duplicate keys may have been produced."

        return dict_to_save_key_str


    def save_data(self, file_name, data_dict_to_save):

        # If directory does not exist, create with .temp_model_folder
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        if file_name[-4:] != ".zip":
            file_name += ".zip"


        current_temp_folder = self._get_temp_folder(file_name)

        try:

            data_format = {}
            attribute_to_save_as_json = {}

            data_dict_to_save = self._check_dict_key_type(data_dict_to_save)

            for attrib_name, attrib_data in data_dict_to_save.items():

                current_file_path = current_temp_folder + attrib_name

                if isinstance(attrib_data, DataFrame):
                    # attrib_data.to_hdf(current_file_path + ".h5", key="DataFrame", mode='w', append = False, format="table")
                    # Save human readable version as a precaution. Append "." so that it is classified as auxiliary file and not loaded
                    attrib_data.to_csv(current_temp_folder + "." + attrib_name + ".csv", index=True)

                    # Using "fixed" as a format causes a PerformanceWarning because it saves types that are not native of C
                    # This is acceptable because it provides the flexibility of using python objects as types (strings, None, etc..)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        attrib_data.to_hdf(current_file_path + ".h5", key="DataFrame", mode='w', append = False, format="fixed")


                elif isinstance(attrib_data, sps.spmatrix):
                    sps.save_npz(current_file_path, attrib_data)

                elif isinstance(attrib_data, np.ndarray):
                    # allow_pickle is FALSE to prevent using pickle and ensure portability
                    np.save(current_file_path, attrib_data, allow_pickle=False)

                else:
                    # Try to parse it as json, if it fails and the data is a dictionary, use another zip file
                    try:
                        _ = json.dumps(attrib_data, default=json_not_serializable_handler)
                        attribute_to_save_as_json[attrib_name] = attrib_data

                    except TypeError:

                        if isinstance(attrib_data, dict):
                            dataIO = DataIO(folder_path = current_temp_folder)
                            dataIO.save_data(file_name = attrib_name, data_dict_to_save=attrib_data)

                        else:
                            raise TypeError("Type not recognized for attribute: {}".format(attrib_name))



            # Save list objects
            if len(data_format)>0:
                attribute_to_save_as_json[".data_format"] = data_format.copy()

            for attrib_name, attrib_data in attribute_to_save_as_json.items():
                current_file_path = current_temp_folder + attrib_name

                # if self._is_windows and len(current_file_path + ".json") >= self._MAX_PATH_LENGTH_WINDOWS:
                #     current_file_path = "\\\\?\\" + current_file_path

                absolute_path = current_file_path + ".json" if current_file_path.startswith(os.getcwd()) else os.getcwd() + current_file_path + ".json"

                assert not self._is_windows or (self._is_windows and len(absolute_path) <= self._MAX_PATH_LENGTH_WINDOWS), \
                    "DataIO: Path of file exceeds {} characters, which is the maximum allowed under standard paths for Windows.".format(self._MAX_PATH_LENGTH_WINDOWS)


                with open(current_file_path + ".json", 'w') as outfile:
                    if isinstance(attrib_data, dict):
                        attrib_data = self._check_dict_key_type(attrib_data)

                    json.dump(attrib_data, outfile, default=json_not_serializable_handler)



            with zipfile.ZipFile(self.folder_path + file_name + ".temp", 'w', compression=zipfile.ZIP_DEFLATED) as myzip:
                for file_to_compress in os.listdir(current_temp_folder):
                    myzip.write(current_temp_folder + file_to_compress, arcname = file_to_compress)

            # Replace file only after the new archive has been successfully created
            # Prevents accidental deletion of previous versions of the file if the current write fails
            os.replace(self.folder_path + file_name + ".temp", self.folder_path + file_name)

        except Exception as exec:

            shutil.rmtree(current_temp_folder, ignore_errors=True)
            raise exec


        shutil.rmtree(current_temp_folder, ignore_errors=True)




    def load_data(self, file_name):

        if file_name[-4:] != ".zip":
            file_name += ".zip"

        dataFile = zipfile.ZipFile(self.folder_path + file_name)

        dataFile.testzip()

        current_temp_folder = self._get_temp_folder(file_name)

        try:

            try:
                data_format = dataFile.extract(".data_format.json", path = current_temp_folder)
                with open(data_format, "r") as json_file:
                    data_format = json.load(json_file)
            except KeyError:
                data_format = {}


            data_dict_loaded = {}

            for file_name in dataFile.namelist():

                # Discard auxiliary data structures
                if file_name.startswith("."):
                    continue

                decompressed_file_path = dataFile.extract(file_name, path = current_temp_folder)
                file_extension = file_name.split(".")[-1]
                attrib_name = file_name[:-len(file_extension)-1]

                if file_extension == "csv":
                    # Compatibility with previous version
                    attrib_data = pd.read_csv(decompressed_file_path, index_col=False)

                elif file_extension == "h5":
                    attrib_data = pd.read_hdf(decompressed_file_path, key=None, mode='r')

                elif file_extension == "npz":
                    attrib_data = sps.load_npz(decompressed_file_path)

                elif file_extension == "npy":
                    # allow_pickle is FALSE to prevent using pickle and ensure portability
                    attrib_data = np.load(decompressed_file_path, allow_pickle=False)

                elif file_extension == "zip":
                    dataIO = DataIO(folder_path = current_temp_folder)
                    attrib_data = dataIO.load_data(file_name = file_name)

                elif file_extension == "json":
                    with open(decompressed_file_path, "r") as json_file:
                        attrib_data = json.load(json_file)

                else:
                    raise Exception("Attribute type not recognized for: '{}' of class: '{}'".format(decompressed_file_path, file_extension))

                data_dict_loaded[attrib_name] = attrib_data


        except Exception as exec:

            shutil.rmtree(current_temp_folder, ignore_errors=True)
            raise exec

        shutil.rmtree(current_temp_folder, ignore_errors=True)


        return data_dict_loaded






from scipy.sparse import random
import unittest


class MyTestCase(unittest.TestCase):

    def test_save_and_load(self):

        arrays = [
           np.array(["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"]),
           np.array(["one", "two", "one", "two", "one", "two", "one", "two"]),
           ]

        multiindex_df = pd.DataFrame(np.random.randn(8, 4), index=arrays)

        sps_random = random(100, 400, density=0.25)

        dataframe = pd.DataFrame(sps_random.copy().toarray())
        dataframe["I am INT"] = np.arange(0, len(dataframe))

        dataframe.loc[1,"I am a mess"] = "A"
        dataframe.loc[2,"I am a mess"] = None

        original_data_dict = {
                        "sps_random": sps_random.copy(),
                        "result_folder_path": "this is just a string",
                        "cutoff_list_validation": [5, 10, 20],
                        "dataframe": dataframe,
                        "multiindex_df_row": multiindex_df,
                        "multiindex_df_col": multiindex_df.transpose(),
                        "nested_dict": {"A": "a", "B": sps_random.copy()}
                        }


        dataIO = DataIO("_test_DataIO/")
        dataIO.save_data(file_name="test_DataIO", data_dict_to_save=original_data_dict)
        loaded_data_dict = dataIO.load_data(file_name="test_DataIO")

        shutil.rmtree("_test_DataIO/", ignore_errors=True)

        self.assertEqual(original_data_dict.keys(), loaded_data_dict.keys())

        # Check data type of each column
        self.assertTrue((original_data_dict['dataframe'].dtypes == loaded_data_dict['dataframe'].dtypes).all()), "Datatypes are different"

        # Check column with different data types: float, int, string, None
        self.assertTrue(type(original_data_dict['dataframe'].loc[0,"I am a mess"]) == type(loaded_data_dict['dataframe'].loc[0,"I am a mess"])), "Datatypes are different"
        self.assertTrue(type(original_data_dict['dataframe'].loc[1,"I am a mess"]) == type(loaded_data_dict['dataframe'].loc[1,"I am a mess"])), "Datatypes are different"
        self.assertTrue(type(original_data_dict['dataframe'].loc[2,"I am a mess"]) == type(loaded_data_dict['dataframe'].loc[2,"I am a mess"])), "Datatypes are different"
        self.assertTrue(type(original_data_dict['dataframe'].loc[3,"I am a mess"]) == type(loaded_data_dict['dataframe'].loc[3,"I am a mess"])), "Datatypes are different"

        # Check various data types: scipy sparse, string, list...
        self.assertTrue(np.array_equal(original_data_dict['sps_random'].toarray(), loaded_data_dict['sps_random'].toarray()))
        self.assertTrue(original_data_dict['result_folder_path'] == loaded_data_dict['result_folder_path'])
        self.assertTrue(original_data_dict['cutoff_list_validation'] == loaded_data_dict['cutoff_list_validation'])

        self.assertTrue(original_data_dict['dataframe'].equals(loaded_data_dict['dataframe']))
        self.assertTrue(original_data_dict['multiindex_df_row'].equals(loaded_data_dict['multiindex_df_row']))
        self.assertTrue(original_data_dict['multiindex_df_col'].equals(loaded_data_dict['multiindex_df_col']))

        # Check content of nested dictionary
        self.assertEqual(original_data_dict['nested_dict'].keys(), loaded_data_dict['nested_dict'].keys())
        self.assertTrue(original_data_dict['nested_dict']["A"] == loaded_data_dict['nested_dict']["A"])
        self.assertTrue(np.array_equal(original_data_dict['nested_dict']["B"].toarray(), loaded_data_dict['nested_dict']["B"].toarray()))

if __name__ == '__main__':

    unittest.main()
