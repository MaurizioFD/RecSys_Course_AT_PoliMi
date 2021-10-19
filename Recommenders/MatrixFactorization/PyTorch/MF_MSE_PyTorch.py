#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/07/2018

@author: Maurizio Ferrari Dacrema
"""


from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
import time


import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Recommenders.MatrixFactorization.PyTorch.MF_MSE_PyTorch_model import MF_MSE_PyTorch_model, DatasetIterator_URM





class MF_MSE_PyTorch(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping):

    RECOMMENDER_NAME = "MF_MSE_PyTorch_Recommender"


    def __init__(self, URM_train):
        super(MF_MSE_PyTorch, self).__init__(URM_train)


    def fit(self, epochs=30, batch_size = 128, num_factors=100,
            learning_rate = 0.0001, use_cuda = True,
            **earlystopping_kwargs):


        self.n_factors = num_factors

        self.batch_size = batch_size
        self.learning_rate = learning_rate


        ########################################################################################################
        #
        #                                SETUP PYTORCH MODEL AND DATA READER
        #
        ########################################################################################################

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self._print("MF_MSE_PyTorch: Using CUDA")
        else:
            self.device = torch.device('cpu')
            self._print("MF_MSE_PyTorch: Using CPU")



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
                                       num_workers = 4,
                                       )


        ########################################################################################################

        self._prepare_model_for_validation()
        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)


        self.ITEM_factors = self.ITEM_factors_best.copy()
        self.USER_factors = self.USER_factors_best.copy()



    def _prepare_model_for_validation(self):
        self.ITEM_factors = self.pyTorchModel.get_ITEM_factors()
        self.USER_factors = self.pyTorchModel.get_USER_factors()


    def _update_best_model(self):
        self.ITEM_factors_best = self.ITEM_factors.copy()
        self.USER_factors_best = self.USER_factors.copy()



    def _run_epoch(self, num_epoch):

        start_time = time.time()

        for num_batch, (input_data, label) in enumerate(self.train_data_loader, 0):

            if (num_batch+1) % 10000 == 0 or (num_batch+1) == len(self.train_data_loader):
                self._print("Epoch {}, Batch: [{}/{}], Samples per second {:.2f}".format(num_epoch+1, num_batch+1, len(self.train_data_loader), (num_batch+1)*self.batch_size/(time.time()-start_time)))

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
