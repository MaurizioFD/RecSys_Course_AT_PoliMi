#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26/11/18

@author: Maurizio Ferrari Dacrema
"""
import os, shutil
from Data_manager.DataReader import DataReader

if __name__ == '__main__':

    print("This script removes all split data from Data_manager_split_datasets without removing the 'original' folders or the downloaded dataset")

    input_proceed = input("Proceed? (y/n): ")




    if input_proceed == "y":

        #walk_generator = os.walk("../" + DataReader.DATASET_SPLIT_ROOT_FOLDER)
        dir_list = os.listdir("../" + DataReader.DATASET_SPLIT_ROOT_FOLDER)

        print("Fount {} dataset drectories: {}".format(len(dir_list), dir_list))

        input_proceed = input("Remove pre-splitted files? (y/n): ")


        for dataset_directory in dir_list:

            subdirectory_list = os.listdir("../" + DataReader.DATASET_SPLIT_ROOT_FOLDER + dataset_directory)



            for subdirectory in subdirectory_list:

                if subdirectory != DataReader.DATASET_SUBFOLDER_ORIGINAL[:-1]:

                    subdirectory = "../" + DataReader.DATASET_SPLIT_ROOT_FOLDER + dataset_directory + "/" + subdirectory

                    if os.path.isdir(subdirectory):

                        shutil.rmtree(subdirectory, ignore_errors=True)

                        print("Removing: {}".format(subdirectory))


        print("Finished!")


    else: print("Terminating")