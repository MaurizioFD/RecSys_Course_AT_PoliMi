#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import numpy as np
import scipy.sparse as sps
import zipfile

from data.IncrementalSparseMatrix import IncrementalSparseMatrix




def split_train_validation_test(URM_all, split_train_test_validation_quota):

    URM_all = URM_all.tocoo()
    URM_shape = URM_all.shape

    numInteractions= len(URM_all.data)

    split = np.random.choice([1, 2, 3], numInteractions, p=split_train_test_validation_quota)


    trainMask = split == 1
    URM_train = sps.coo_matrix((URM_all.data[trainMask], (URM_all.row[trainMask], URM_all.col[trainMask])), shape=URM_shape)
    URM_train = URM_train.tocsr()

    testMask = split == 2

    URM_test = sps.coo_matrix((URM_all.data[testMask], (URM_all.row[testMask], URM_all.col[testMask])), shape=URM_shape)
    URM_test = URM_test.tocsr()

    validationMask = split == 3

    URM_validation = sps.coo_matrix((URM_all.data[validationMask], (URM_all.row[validationMask], URM_all.col[validationMask])), shape=URM_shape)
    URM_validation = URM_validation.tocsr()


    return URM_train, URM_validation, URM_test






import sys, time, pickle


def urllretrieve_reporthook(count, block_size, total_size):

    global start_time

    if count == 0:
        start_time = time.time()
        return

    duration = time.time() - start_time + 1

    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count*block_size*100/total_size),100)

    sys.stdout.write("\rDataReader: Downloaded {:.2f}%, {:.2f} MB, {:.0f} KB/s, {:.0f} seconds passed".format(
                    percent, progress_size / (1024 * 1024), speed, duration))

    sys.stdout.flush()


def downloadFromURL(URL, destinationFolder):

    from urllib.request import urlretrieve

    urlretrieve (URL, destinationFolder, reporthook=urllretrieve_reporthook)

    sys.stdout.write("\n")
    sys.stdout.flush()





class Movielens10MReader(object):

    DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-10m.zip"


    def __init__(self, split_train_test_validation_quota =[0.6, 0.2, 0.2]):

        super(Movielens10MReader, self).__init__()

        if sum(split_train_test_validation_quota) != 1.0 or len(split_train_test_validation_quota) != 3:
            raise ValueError("Movielens10MReader: splitTrainTestValidation must be a probability distribution over Train, Test and Validation")

        print("Movielens10MReader: loading data...")

        dataSubfolder = "./data/Movielens_10M/"

        try:

            dataFile = zipfile.ZipFile(dataSubfolder + "ml-10m.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("Movielens10MReader: Unable to fild data zip file. Downloading...")


            downloadFromURL(self.DATASET_URL, dataSubfolder + "ml-10m.zip")

            dataFile = zipfile.ZipFile(dataSubfolder + "ml-10m.zip")



        URM_path = dataFile.extract("ml-10M100K/ratings.dat", path=dataSubfolder)


        try:
            self.URM_train = sps.load_npz(dataSubfolder + "URM_train.npz")
            self.URM_test = sps.load_npz(dataSubfolder + "URM_test.npz")
            self.URM_validation = sps.load_npz(dataSubfolder + "URM_validation.npz")

            self.column_token_to_id_mapper = pickle.load(open(dataSubfolder + "column_token_to_id_mapper", "rb"))
            self.row_token_to_id_mapper = pickle.load(open(dataSubfolder + "row_token_to_id_mapper", "rb"))


        except FileNotFoundError:

            # Rebuild split

            print("Movielens10MReader: URM_train or URM_test or URM_validation not found. Building new ones")


            URM_builder = self._load_URM(URM_path)

            URM_all = URM_builder.get_SparseMatrix()
            self.column_token_to_id_mapper = URM_builder.get_column_token_to_id_mapper()
            self.row_token_to_id_mapper = URM_builder.get_row_token_to_id_mapper()



            ###################################################################
            ################ SPLIT DATA

            self.URM_train, self.URM_test, self.URM_validation = split_train_validation_test(URM_all, split_train_test_validation_quota)


            ###################################################################
            ################ SAVE SPLIT


            print("Movielens10MReader: saving URM_train and URM_test")
            sps.save_npz(dataSubfolder + "URM_train.npz", self.URM_train)
            sps.save_npz(dataSubfolder + "URM_test.npz", self.URM_test)
            sps.save_npz(dataSubfolder + "URM_validation.npz", self.URM_validation)


            pickle.dump(self.column_token_to_id_mapper, open(dataSubfolder + "column_token_to_id_mapper", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.row_token_to_id_mapper, open(dataSubfolder + "row_token_to_id_mapper", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

        print("Movielens10MReader: loading complete")




    def get_URM_train(self):
        return self.URM_train

    def get_URM_test(self):
        return self.URM_test

    def get_URM_validation(self):
        return self.URM_validation




    def _load_URM (self, filePath, header = False, separator="::"):

        URM_builder = IncrementalSparseMatrix(auto_create_col_mapper=True, auto_create_row_mapper=True)

        fileHandle = open(filePath, "r")
        numCells = 0

        if header:
            fileHandle.readline()

        for line in fileHandle:
            numCells += 1

            if (numCells % 1000000 == 0):
                print("Processed {} cells".format(numCells))

            if (len(line)) > 1:
                line = line.split(separator)

                line[-1] = line[-1].replace("\n", "")

                if not line[2] == "0" and not line[2] == "NaN":

                    user_id = line[0]
                    item_id = line[1]
                    rating = float(line[2])

                    URM_builder.add_data_lists([user_id], [item_id], [rating])


        fileHandle.close()


        return  URM_builder
