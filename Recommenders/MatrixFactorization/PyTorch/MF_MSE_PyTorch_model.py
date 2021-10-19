#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 05/07/18

@author: Maurizio Ferrari Dacrema
"""

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np


class MF_MSE_PyTorch_model(torch.nn.Module):

    def __init__(self, n_users, n_items, n_factors):

        super(MF_MSE_PyTorch_model, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors

        self.USER_factors = torch.nn.Embedding(num_embeddings = self.n_users, embedding_dim = self.n_factors)
        self.ITEM_factors = torch.nn.Embedding(num_embeddings = self.n_items, embedding_dim = self.n_factors)

        self.layer_1 = torch.nn.Linear(in_features = self.n_factors, out_features = 1)

        self.activation_function = torch.nn.ReLU()



    def forward(self, user_coordinates, item_coordinates):

        current_user_factors = self.USER_factors(user_coordinates)
        current_item_factors = self.ITEM_factors(item_coordinates)

        prediction = torch.mul(current_user_factors, current_item_factors)

        prediction = self.layer_1(prediction)
        prediction = self.activation_function(prediction)

        return prediction



    def get_USER_factors(self):
        return self.USER_factors.weight.detach().cpu().numpy()


    def get_ITEM_factors(self):
        return self.ITEM_factors.weight.detach().cpu().numpy()
















class DatasetIterator_URM(Dataset):

    def __init__(self, URM):

        URM = URM.tocoo()

        self.n_data_points = URM.nnz

        self.user_item_coordinates = np.empty((self.n_data_points, 2))

        self.user_item_coordinates[:,0] = URM.row.copy()
        self.user_item_coordinates[:,1] = URM.col.copy()
        self.rating = URM.data.copy().astype(np.float)

        self.user_item_coordinates = torch.Tensor(self.user_item_coordinates).type(torch.LongTensor)
        self.rating = torch.Tensor(self.rating)





    def __getitem__(self, index):
        """
        Format is (row, col, data)
        :param index:
        :return:
        """

        return self.user_item_coordinates[index, :], self.rating[index]


    def __len__(self):

        return self.n_data_points

