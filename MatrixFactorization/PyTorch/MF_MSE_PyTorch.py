#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/07/2018

@author: Maurizio Ferrari Dacrema
"""


from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Base.BaseRecommender import BaseRecommender
import os, sys
import numpy as np, pickle


import torch
from torch.autograd import Variable


from torch.utils.data import DataLoader






class MF_MSE_PyTorch(BaseRecommender, Incremental_Training_Early_Stopping):

    RECOMMENDER_NAME = "MF_MSE_PyTorch_Recommender"


    def __init__(self, URM_train, positive_threshold=4):


        super(MF_MSE_PyTorch, self).__init__()


        self.URM_train = URM_train
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]
        self.normalize = False

        self.positive_threshold = positive_threshold

        self.compute_item_score = self.compute_score_MF






    def compute_score_MF(self, user_id):

        scores_array = np.dot(self.W[user_id], self.H.T)

        return scores_array





    def fit(self, epochs=30, batch_size = 128, num_factors=10,
            learning_rate = 0.001, use_cuda = True,
            **earlystopping_kwargs):


        self.n_factors = num_factors


        # Select only positive interactions
        URM_train_positive = self.URM_train.copy()

        URM_train_positive.data = URM_train_positive.data >= self.positive_threshold
        URM_train_positive.eliminate_zeros()


        self.batch_size = batch_size
        self.learning_rate = learning_rate


        ########################################################################################################
        #
        #                                SETUP PYTORCH MODEL AND DATA READER
        #
        ########################################################################################################

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("MF_MSE_PyTorch: Using CUDA")
        else:
            self.device = torch.device('cpu')
            print("MF_MSE_PyTorch: Using CPU")

        from MatrixFactorization.PyTorch.MF_MSE_PyTorch_model import MF_MSE_PyTorch_model, DatasetIterator_URM

        n_users, n_items = self.URM_train.shape

        self.pyTorchModel = MF_MSE_PyTorch_model(n_users, n_items, self.n_factors).to(self.device)

        #Choose loss
        self.lossFunction = torch.nn.MSELoss(size_average=False)
        #self.lossFunction = torch.nn.BCELoss(size_average=False)
        self.optimizer = torch.optim.Adagrad(self.pyTorchModel.parameters(), lr = self.learning_rate)


        dataset_iterator = DatasetIterator_URM(self.URM_train)

        self.train_data_loader = DataLoader(dataset = dataset_iterator,
                                       batch_size = self.batch_size,
                                       shuffle = True,
                                       #num_workers = 2,
                                       )


        ########################################################################################################

        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)


        self.ITEM_factors = self.W_best.copy()
        self.USER_factors = self.H_best.copy()

        self._print("Computing NMF decomposition... Done!")
        
        sys.stdout.flush()



    def _initialize_incremental_model(self):

        self.W_incremental = self.pyTorchModel.get_W()
        self.W_best = self.W_incremental.copy()

        self.H_incremental = self.pyTorchModel.get_H()
        self.H_best = self.H_incremental.copy()



    def _update_incremental_model(self):

        self.W_incremental = self.pyTorchModel.get_W()
        self.H_incremental = self.pyTorchModel.get_H()

        self.W = self.W_incremental.copy()
        self.H = self.H_incremental.copy()


    def _update_best_model(self):

        self.W_best = self.W_incremental.copy()
        self.H_best = self.H_incremental.copy()



    def _run_epoch(self, num_epoch):


        for num_batch, (input_data, label) in enumerate(self.train_data_loader, 0):

            if num_batch % 1000 == 0:
                print("num_batch: {}".format(num_batch))

            # On windows requires int64, on ubuntu int32
            #input_data_tensor = Variable(torch.from_numpy(np.asarray(input_data, dtype=np.int64))).to(self.device)
            input_data_tensor = Variable(input_data).to(self.device)

            label_tensor = Variable(label).to(self.device)


            user_coordinates = input_data_tensor[:,0]
            item_coordinates = input_data_tensor[:,1]

            # FORWARD pass
            prediction = self.pyTorchModel(user_coordinates, item_coordinates)

            # Pass prediction and label removing last empty dimension of prediction
            loss = self.lossFunction(prediction.view(-1), label_tensor)

            # BACKWARD pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
