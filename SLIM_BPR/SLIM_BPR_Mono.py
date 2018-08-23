#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 28 June 2017

@author: Maurizio Ferrari Dacrema
"""

import os
import subprocess
import time

import numpy as np
import scipy.sparse as sps
from Recommender_utils import similarityMatrixTopK

from Base.Recommender import Recommender


class SLIM_BPR_Mono(Recommender):

    def __init__(self, URM_train, lambda_i = 0.0025, lambda_j = 0.00025, learning_rate = 0.05, topK = False):
        """
        WARNING: This class needs to save files on disk to call another executable. Neither this class nor the executable
        are thread safe

        :param URM_train:
        :param lambda_i:
        :param lambda_j:
        :param learning_rate:
        :param topK:
        """
        super(SLIM_BPR_Mono, self).__init__()

        self.URM_train = URM_train
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate
        self.normalize = False
        self.sparse_weights = True
        self.topK = topK

        #processPid = os.getpid()

        self.basePath = "data/"
        self.executablePath = "MyMediaLite/bin/item_recommendation"
        #self.trainFileName = "SLIM_BPR_Mono_URM_train_{}.csv".format(processPid)
        self.trainFileName = "SLIM_BPR_Mono_URM_train.csv"
        self.outputModelName = "SLIM_BPR_Mono_Model.txt"

        # Set permission to execute, code 0775, the o is needed in python to encode octal numbers
        os.chmod(self.executablePath, 0o775)

        self.checkMonoInstallation()


    def checkMonoInstallation(self):

        command = ['mono']

        print("\n\nChecking mono installation.... \n")

        try:

            output = subprocess.check_output(' '.join(command), shell=True)

            #"The program 'mono' is currently not installed. You can install it by typing: \n sudo apt install mono-runtime"

            monoExpectedOutput = "Usage is: mono [options] program [program-options]"

            output = str(output.decode())

        except Exception as e:

            if "Command 'mono' returned non-zero exit status 1" not in str(e):
                raise Exception("WARNING: No mono installation detected, the selected recommender requires it.")

    def removeTemporaryFiles(self):

        # Remove saved Model and URM

        print("Removing: {}".format(self.basePath + self.trainFileName))
        os.remove(self.basePath + self.trainFileName)

        print("Removing: {}".format(self.basePath + self.trainFileName + ".bin.PosOnlyFeedback"))
        os.remove(self.basePath + self.trainFileName + ".bin.PosOnlyFeedback")

        print("Removing: {}".format(self.basePath + self.outputModelName))
        os.remove(self.basePath + self.outputModelName)

    def writeSparseToFile(self, sparseMatrix, file):

        sparseMatrix = sparseMatrix.tocoo()

        data = sparseMatrix.data
        row = sparseMatrix.row
        col = sparseMatrix.col

        for index in range(len(data)):
            file.write("{},{},{}\n".format(row[index], col[index], data[index]))

            #if (index % 500000 == 0):
            #    print("Processed {} rows of {}".format(index, len(data)))

        file.close()

    def loadModelIntoDenseMatrix(self, filePath):

        SLIMsimilarity = open(filePath, "r")
        SLIMsimilarity.readline()  # program name
        SLIMsimilarity.readline()  # 2.99
        line = SLIMsimilarity.readline()  # size

        n_items_model = int(line.split(" ")[0])

        if n_items_model<self.n_items:
            print("The model file contains less items than the URM_train, it may be that some items do not have interactions.")

        # Shape requires the number of cells, which is the number of items
        self.W = np.zeros((self.n_items,self.n_items), dtype=np.float32)

        numCells = 0
        print("Loading SLIM model")

        for line in SLIMsimilarity:
            numCells += 1
            #if (numCells % 1000000 == 0):
            #    print("Processed {} cells".format(numCells))

            if (len(line)) > 1:
                line = line.split(" ")

                value = line[2].replace("\n", "")

                if not value == "0" and not value == "NaN":
                    row = int(line[0])
                    col = int(line[1])
                    value = float(value)

                    self.W[row, col] = value

        SLIMsimilarity.close()

        self.W = self.W.T

        if self.topK != False:

            self.W_sparse = similarityMatrixTopK(self.W, k=self.topK)

            self.sparse_weights = True
            del self.W

        else:
            self.sparse_weights = False


    def loadModelIntoSparseMatrix(self, filePath):

        values, rows, cols = [], [], []

        SLIMsimilarity = open(filePath, "r")
        SLIMsimilarity.readline()  # program name
        SLIMsimilarity.readline()  # 2.99
        line = SLIMsimilarity.readline()  # size

        n_items_model = int(line.split(" ")[0])

        if n_items_model<self.n_items:
            print("The model file contains less items than the URM_train, it may be that some items do not have interactions.")


        numCells = 0
        print("Loading SLIM model")

        for line in SLIMsimilarity:
            numCells += 1
            #if (numCells % 1000000 == 0):
            #    print("Processed {} cells".format(numCells))

            if (len(line)) > 1:
                line = line.split(" ")

                value = line[2].replace("\n", "")

                if not value == "0" and not value == "NaN":
                    rows.append(int(line[0]))
                    cols.append(int(line[1]))
                    values.append(float(value))

        SLIMsimilarity.close()

        self.W_sparse = sps.csr_matrix((values, (rows, cols)), shape=(self.n_items, self.n_items), dtype=np.float32)
        self.W_sparse = self.W_sparse.T
        self.sparse_weights = True

        if self.topK != False:
            self.W_sparse = similarityMatrixTopK(self.W_sparse, k=self.topK)


    def fit(self, epochs=30, deleteFiles=False):
        """
        Train SLIM wit BPR. If the model was already trained, overwrites matrix S
        Training is performed via batch gradient descent
        :param epochs:
        :return: -
        """

        recommenderMethod = "BPRSLIM"

        try:
            # If train file already exist, clean all data
            #trainFile = open(self.basePath + self.trainFileName, "r")
            #trainFile.close()
            print("Removing previous SLIM_BPR files")
            self.removeTemporaryFiles()

        except:
            pass

        print("Writing URM_train to {}".format(self.basePath + self.trainFileName))
        self.writeSparseToFile(self.URM_train, open(self.basePath + self.trainFileName, "w"))


        recommenderOptions = 'reg_i={reg_i} reg_j={reg_j} learn_rate={learn_rate} num_iter={num_iter}'.format(
            reg_i=self.lambda_i,
            reg_j=self.lambda_j,
            learn_rate=self.learning_rate,
            num_iter=epochs)

        command = ['"{}"'.format(self.executablePath),
                   '--training-file="{}"'.format(self.basePath + self.trainFileName),
                   #'--test-file="{}"'.format(basePath + testFileName),
                   '--recommender="{}"'.format(recommenderMethod),
                   '--no-id-mapping',
                   #'--save-user-mapping="{}"'.format(basePath + "save-user-mapping"),
                   #'--save-item-mapping="{}"'.format(basePath + "save-item-mapping"),
                   # '--test-ratio=0,25',
                   # '--rating-threshold={}'.format(ratingThreshold),
                   # '--prediction-file="{}"'.format(basePath + outputPredictionName),
                   '--save-model="{}"'.format(self.basePath + self.outputModelName),
                   #'--predict-items-number={}'.format(test_case_current[0]),
                   #'--measures="AUC prec@{} recall@{} NDCG MRR"'.format(itemsToPredict, itemsToPredict),
                   '--recommender-options="{}"'.format(recommenderOptions)
                   ]

        print("SLIM BPR hyperparameters: " + recommenderOptions)

        start_time = time.time()

        output = subprocess.check_output(' '.join(command), shell=True)

        print("\n\n" + str(output.decode()) + "\n\n")
        print("Train complete, time required {:.2f} seconds".format(time.time()-start_time))

        """
        training data: 179000 users, 4866 items, 4226544 events, sparsity 99.51476\n
        test data:     134486 users, 4820 items, 869112 events, sparsity 99.86592\n
        BPRSLIM reg_i=0.0025 reg_j=0.00025 num_iter=15 learn_rate=0.05 uniform_user_sampling=True with_replacement=False update_j=True \n
        training_time 00:02:51.6474470  prediction_time 00:12:23.3718540\n\n
        """

        # Read prediction file and calculate the scores
        #self.loadModelIntoSparseMatrix(basePath + outputModelName)
        self.loadModelIntoDenseMatrix(self.basePath + self.outputModelName)

        if deleteFiles:
            self.removeTemporaryFiles()