#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import numpy as np
import scipy.sparse as sps
import zipfile




def removeZeroRatingRowAndCol(URM):

    rows = URM.indptr
    numRatings = np.ediff1d(rows)
    mask = numRatings >= 1

    URM = URM[mask,:]

    cols = URM.tocsc().indptr
    numRatings = np.ediff1d(cols)
    mask = numRatings >= 1

    URM = URM[:,mask]

    return URM.tocsr()



def loadCSVintoSparse (filePath, header = False, separator="::"):

    values, rows, cols = [], [], []

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
                rows.append(int(line[0]))
                cols.append(int(line[1]))
                values.append(float(line[2]))

    fileHandle.close()

    return  sps.csr_matrix((values, (rows, cols)), dtype=np.float32)



def saveSparseIntoCSV (filePath, sparse_matrix, separator=","):

    sparse_matrix = sparse_matrix.tocoo()

    fileHandle = open(filePath, "w")

    for index in range(len(sparse_matrix.data)):
        fileHandle.write("{row}{separator}{col}{separator}{value}\n".format(
            row = sparse_matrix.row[index], col = sparse_matrix.col[index], value = sparse_matrix.data[index],
            separator = separator))




class Movielens10MReader(object):

    def __init__(self, splitTrainTest = False, splitTrainTestValidation =[0.6, 0.2, 0.2] , loadPredefinedTrainTest = True):

        super(Movielens10MReader, self).__init__()

        if sum(splitTrainTestValidation) != 1.0 or len(splitTrainTestValidation) != 3:
            raise ValueError("Movielens10MReader: splitTrainTestValidation must be a probability distribution over Train, Test and Validation")

        print("Movielens10MReader: loading data...")

        dataSubfolder = "./data/"

        dataFile = zipfile.ZipFile(dataSubfolder + "movielens_10m.zip")
        URM_path = dataFile.extract("ml-10M100K/ratings.dat", path=dataSubfolder)


        if not loadPredefinedTrainTest:
            self.URM_all = loadCSVintoSparse(URM_path, separator="::")
            self.URM_all = removeZeroRatingRowAndCol(self.URM_all)

        else:

            try:
                self.URM_train = sps.load_npz(dataSubfolder + "URM_train.npz")
                self.URM_test = sps.load_npz(dataSubfolder + "URM_test.npz")
                self.URM_validation = sps.load_npz(dataSubfolder + "URM_validation.npz")

                return

            except FileNotFoundError:
                # Rebuild split
                print("Movielens10MReader: URM_train or URM_test or URM_validation not found. Building new ones")

                splitTrainTest = True
                self.URM_all = loadCSVintoSparse(URM_path)
                self.URM_all = removeZeroRatingRowAndCol(self.URM_all)



        if splitTrainTest:

            self.URM_all = self.URM_all.tocoo()
            URM_shape = self.URM_all.shape

            numInteractions= len(self.URM_all.data)

            split = np.random.choice([1, 2, 3], numInteractions, p=splitTrainTestValidation)


            trainMask = split == 1
            self.URM_train = sps.coo_matrix((self.URM_all.data[trainMask], (self.URM_all.row[trainMask], self.URM_all.col[trainMask])), shape=URM_shape)
            self.URM_train = self.URM_train.tocsr()

            testMask = split == 2

            self.URM_test = sps.coo_matrix((self.URM_all.data[testMask], (self.URM_all.row[testMask], self.URM_all.col[testMask])), shape=URM_shape)
            self.URM_test = self.URM_test.tocsr()

            validationMask = split == 3

            self.URM_validation = sps.coo_matrix((self.URM_all.data[validationMask], (self.URM_all.row[validationMask], self.URM_all.col[validationMask])), shape=URM_shape)
            self.URM_validation = self.URM_validation.tocsr()

            del self.URM_all

            print("Movielens10MReader: saving URM_train and URM_test")
            sps.save_npz(dataSubfolder + "URM_train.npz", self.URM_train)
            sps.save_npz(dataSubfolder + "URM_test.npz", self.URM_test)
            sps.save_npz(dataSubfolder + "URM_validation.npz", self.URM_validation)

        print("Movielens10MReader: loading complete")




    def get_URM_train(self):
        return self.URM_train

    def get_URM_test(self):
        return self.URM_test

    def get_URM_validation(self):
        return self.URM_validation
