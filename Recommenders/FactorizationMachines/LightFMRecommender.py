#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11/07/2021

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.DataIO import DataIO
from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender
from lightfm import LightFM
import numpy as np
from copy import deepcopy
import scipy.sparse as sps



class _BaseLightFMWrapper(BaseRecommender, Incremental_Training_Early_Stopping):
    """
    Wrapper of the LightFM library
    See: https://github.com/lyst/lightfm

    https://www.ethanrosenthal.com/2016/11/07/implicit-mf-part-2/

    When no user_features or item_features are explicitly included, then LightFM assumes that both feature matrices are in
    fact identity matrices of size (num_users X num_users) or (num_items X num_items) for user and item feature matrices,
    respectively. What this is effectively doing is one-hot-encoding each user and item ID as a single feature vector.

    In the case where you do pass an item_features matrix, then LightFM does not do any one-hot-encoding. Thus, each user
    and item ID does not get its own vector unless you explicitly define one. The easiest way to do this is to make your own
    identity matrix and stack it on the side of the item_features matrix that we already created. This way, each item is described
    by a single vector for its unique ID and then a set of vectors for each of its tags.

    """

    RECOMMENDER_NAME = "BaseLightFMWrapper"

    LOSS_VALUES = ['bpr', 'warp', 'warp-kos']
    SGD_MODE_VALUES = ['adagrad', 'adadelta']



    def _compute_item_score(self, user_id_array, items_to_compute = None):

        # Create a single (n_items, ) array with the item score, then copy it for every user
        if items_to_compute is None:
            items_to_compute = np.arange(self.n_items)
            # item_features = self.ICM_train
        else:
            items_to_compute = np.array(items_to_compute)
            # item_features = self.ICM_train[items_to_compute,:] if self.ICM_train is not None else None

        item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf
        # user_features = self.UCM_train[user_id_array,:] if self.UCM_train is not None else None


        for user_index, user_id in enumerate(user_id_array):
            # try:
            item_scores[user_index,items_to_compute] = self.lightFM_model.predict(int(user_id),
                                                                                 items_to_compute,
                                                                                 item_features = self.ICM_train,
                                                                                 user_features = self.UCM_train)
            # except:
            #     print(user_id)

        return item_scores



    def _init_model(self, loss, sgd_mode, n_components, item_alpha, user_alpha, learning_rate):

        self.lightFM_model = LightFM(loss = loss,
                                     item_alpha = item_alpha,
                                     user_alpha = user_alpha,
                                     no_components = n_components,
                                     k = 5, n = 10,
                                     # k (int, optional) – for k-OS training, the k-th positive example will be selected from the n positive examples sampled for every user.
                                     # n (int, optional) – for k-OS training, maximum number of positives sampled for each update.
                                     learning_schedule = sgd_mode,
                                     learning_rate = learning_rate,
                                     rho = 0.95, epsilon = 1e-06,
                                     # rho (float, optional) – moving average coefficient for the adadelta learning schedule.
                                     # epsilon (float, optional) – conditioning parameter for the adadelta learning schedule.
                                     max_sampled = 10
                                     # max_sampled (int, optional) – maximum number of negative samples used during WARP fitting.
                                    )



    def fit(self, epochs = 300, loss = "bpr", sgd_mode = "adagrad", n_components = 10,
            item_alpha = 0.0, user_alpha = 0.0,
            learning_rate = 0.05, num_threads = 4, **earlystopping_kwargs):

        if loss not in self.LOSS_VALUES:
           raise ValueError("Value for 'loss' not recognized. Acceptable values are {}, provided was '{}'".format(self.LOSS_VALUES, loss))

        if sgd_mode not in self.SGD_MODE_VALUES:
           raise ValueError("Value for 'sgd_mode' not recognized. Acceptable values are {}, provided was '{}'".format(self.SGD_MODE_VALUES, sgd_mode))

        self._init_model(loss, sgd_mode, n_components, item_alpha, user_alpha, learning_rate)

        self.num_threads = num_threads

        ########################### Earlystopping

        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.lightFM_model = self.lightFM_model_best



        # self.lightFM_model = self.lightFM_model.fit(self.URM_train,
        #                                item_features = self.ICM_train,
        #                                user_features = self.UCM_train,
        #                                epochs=epochs,
        #                                num_threads = num_threads,
        #                                verbose = self.verbose)



    def _prepare_model_for_validation(self):
        pass


    def _update_best_model(self):
        self.lightFM_model_best = deepcopy(self.lightFM_model)


    def _run_epoch(self, num_epoch):

        self.lightFM_model = self.lightFM_model.fit_partial(self.URM_train,
                                                            item_features = self.ICM_train,
                                                            user_features = self.UCM_train,
                                                            epochs = 1,
                                                            num_threads = self.num_threads,
                                                            verbose = False)



    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {
                            "item_embeddings": self.lightFM_model.item_embeddings,
                            "item_embedding_gradients": self.lightFM_model.item_embedding_gradients,
                            "item_embedding_momentum": self.lightFM_model.item_embedding_momentum,
                            "item_biases": self.lightFM_model.item_biases,
                            "item_bias_gradients": self.lightFM_model.item_bias_gradients,
                            "item_bias_momentum": self.lightFM_model.item_bias_momentum,
                            "user_embeddings": self.lightFM_model.user_embeddings,
                            "user_embedding_gradients": self.lightFM_model.user_embedding_gradients,
                            "user_embedding_momentum": self.lightFM_model.user_embedding_momentum,
                            "user_biases": self.lightFM_model.user_biases,
                            "user_bias_gradients": self.lightFM_model.user_bias_gradients,
                            "user_bias_momentum": self.lightFM_model.user_bias_momentum,
                            }


        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")




    def load_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        self.lightFM_model = LightFM()

        for attrib_name in data_dict.keys():
             self.lightFM_model.__setattr__(attrib_name, data_dict[attrib_name])

        self._print("Loading complete")










class LightFMCFRecommender(_BaseLightFMWrapper):
    """LightFMRecommender"""

    RECOMMENDER_NAME = "LightFMCFRecommender"

    def __init__(self, URM_train, verbose = True):
        super(LightFMCFRecommender, self).__init__(URM_train, verbose = verbose)
        self.ICM_train = None
        self.UCM_train = None


class LightFMItemHybridRecommender(BaseItemCBFRecommender, _BaseLightFMWrapper):
    """LightFMItemHybridRecommender"""

    RECOMMENDER_NAME = "LightFMItemHybridRecommender"

    def __init__(self, URM_train, ICM_train, verbose = True):
        super(LightFMItemHybridRecommender, self).__init__(URM_train, ICM_train, verbose = verbose)
        self.UCM_train = None

        # Need to hstack item_features to ensure each ItemIDs are present in the model
        eye = sps.eye(self.n_items, self.n_items).tocsr()
        self.ICM_train = sps.hstack((eye, self.ICM_train)).tocsr()



class LightFMUserHybridRecommender(BaseUserCBFRecommender, _BaseLightFMWrapper):
    """LightFMUserHybridRecommender"""

    RECOMMENDER_NAME = "LightFMUserHybridRecommender"

    def __init__(self, URM_train, UCM_train, verbose = True):
        super(LightFMUserHybridRecommender, self).__init__(URM_train, UCM_train, verbose = verbose)
        self.ICM_train = None

        # Need to hstack user_features to ensure each UserIDs are present in the model
        eye = sps.eye(self.n_users, self.n_users).tocsr()
        self.UCM_train = sps.hstack((eye, self.UCM_train)).tocsr()