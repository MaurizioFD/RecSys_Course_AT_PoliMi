#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 04/01/2023

@author: Maurizio Ferrari Dacrema
"""

import torch
import scipy.sparse as sps
import numpy as np
import copy

def loss_MSE(model, batch, device):
    user, item, rating = batch

    # Compute prediction for each element in batch
    prediction = model.forward(user.to(device), item.to(device))

    # Compute total loss for batch
    loss = (prediction - rating.to(device)).pow(2).mean()

    return loss



def loss_BPR(model, batch, device):
    user, item_positive, item_negative = batch

    # Compute prediction for each element in batch
    all_items = torch.cat([item_positive, item_negative]).to(device)
    all_users = torch.cat([user, user]).to(device)

    all_predictions = model.forward(all_users, all_items)
    x_i, x_j = torch.split(all_predictions, [len(user), len(user)])

    x_ij = x_i - x_j

    # Compute total loss for batch
    loss = -x_ij.sigmoid().log().mean()

    return loss






def get_optimizer(optimizer_label, model, learning_rate, l2_reg):

    if optimizer_label.lower() == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr = learning_rate, weight_decay = l2_reg)
    elif optimizer_label.lower() == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr = learning_rate, weight_decay = l2_reg)
    elif optimizer_label.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = l2_reg)
    elif optimizer_label.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay = l2_reg)
    else:
        raise ValueError("sgd_mode attribute value not recognized.")



def _sps_to_coo_tensor(URM_train, device):
    URM_train = sps.coo_matrix(URM_train)
    return torch.sparse_coo_tensor(np.array([URM_train.row, URM_train.col]),
                                   URM_train.data,
                                   URM_train.shape, dtype=torch.float32, device=device)


def clone_pytorch_model_to_numpy_dict(model):

    cloned_state_dict = copy.deepcopy(model.state_dict())
    cloned_state_dict = {key:val.detach().cpu().numpy() for key,val in cloned_state_dict.items()}

    return cloned_state_dict
