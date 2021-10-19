#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/02/2019

"""

import subprocess, os, shutil
import numpy as np
from tqdm import tqdm

from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Recommender_utils import check_matrix


class SVDFeature(BaseRecommender):

    RECOMMENDER_NAME = "SVDFeature"
    FILE_MODEL_NAME = "svd_feature_train"
    FILE_TEST_NAME = "svd_feature_test"
    FILE_PREDICTION_NAME = "svd_feature_predicted"
    DEFAULT_TEMP_FILE_FOLDER = './result_experiments/__Temp_SVDFeature/'


    def __init__(self, URM_train, ICM=None, UCM=None):

        super(SVDFeature, self).__init__()

        self.URM_train = check_matrix(URM_train, "csr")
        self.ICM = ICM
        self.UCM = UCM
        self.n_users, self.n_items = URM_train.shape
        self.normalize = False


    def __dealloc__(self):

        if self.temp_file_folder == self.DEFAULT_TEMP_FILE_FOLDER:
            print("{}: cleaning temporary files".format(self.RECOMMENDER_NAME))
            shutil.rmtree(self.DEFAULT_TEMP_FILE_FOLDER, ignore_errors=True)



    def _compute_item_score(self, user_id_array, items_to_compute=None):

        if items_to_compute is None:
            items_to_compute = np.arange(self.n_items)

        item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)*np.inf


        with open(self.temp_file_folder + self.FILE_TEST_NAME, "w") as fileout:
            for userid in user_id_array.tolist():
                for itemid in items_to_compute.tolist():
                    print(self._get_feature_format(userid, itemid), file=fileout)

        args = ["svd_feature_infer",
                "pred=0",
                "test:input_type=1",
                "test:data_in={}".format(self.temp_file_folder + self.FILE_TEST_NAME),
                "name_pred={}".format(self.temp_file_folder + self.FILE_PREDICTION_NAME)]

        subprocess.run(args)

        # item_scores = np.zeros((len(user_id_array), len(items_to_compute)))

        with open(self.temp_file_folder + self.FILE_PREDICTION_NAME, "r") as filein:
            for userid in user_id_array.tolist():
                for itemid in range(len(items_to_compute)):
                    item_scores[userid, itemid] = float(filein.readline())

        return item_scores


    def _get_feature_format(self, userid, itemid):

        output = "{:.2f}\t".format(self.URM_train[userid, itemid])

        if self.UCM is not None:
            userfeatures = self.UCM[userid]
            output += "{:d}\t".format(userfeatures.nnz + 1)
        else:
            userfeatures = None
            output += "1\t"

        if self.ICM is not None:
            itemfeatures = self.ICM[itemid]
            output += "{:d}\t".format(itemfeatures.nnz + 1)
        else:
            itemfeatures = None
            output += "1\t"

        output += "{:d}:1\t".format(userid)
        if userfeatures is not None:
            for j in range(len(userfeatures.indices)):
                output += "{:d}:{:.6f}\t".format(userfeatures.indices[j] + self.n_users,
                                                 userfeatures.data[j])

        output += "{:d}:1\t".format(itemid)
        if itemfeatures is not None:
            for j in range(len(itemfeatures.indices)):
                output += "{:d}:{:.6f}\t".format(itemfeatures.indices[j] + self.n_items,
                                                 itemfeatures.data[j])

        return output


    def _write_feature_format_file(self):

        self.n_item_features = self.n_items
        if self.ICM is not None:
            self.ICM = check_matrix(self.ICM, "csr")
            self.n_item_features += self.ICM.shape[1]

        self.n_user_features = self.n_users
        if self.UCM is not None:
            self.UCM = check_matrix(self.UCM, "csr")
            self.n_user_features += self.UCM.shape[1]

        nnz_rows, nnz_cols = self.URM_train.nonzero()

        with open(self.temp_file_folder + self.FILE_MODEL_NAME, "w") as fileout:

            for i in tqdm(range(len(nnz_rows))):

                userid, itemid = nnz_rows[i], nnz_cols[i]
                output = self._get_feature_format(userid, itemid)

                print(output, file=fileout)


    def fit(self, epochs=30, num_factors=32, learning_rate=0.01,
            user_reg=0.0, item_reg=0.0, user_bias_reg=0.0, item_bias_reg=0.0,
            temp_file_folder = None):


        if temp_file_folder is None:
            print("{}: Using default Temp folder '{}'".format(self.RECOMMENDER_NAME, self.DEFAULT_TEMP_FILE_FOLDER))
            self.temp_file_folder = self.DEFAULT_TEMP_FILE_FOLDER
        else:
            print("{}: Using Temp folder '{}'".format(self.RECOMMENDER_NAME, temp_file_folder))
            self.temp_file_folder = temp_file_folder

        if not os.path.isdir(self.temp_file_folder):
            os.makedirs(self.temp_file_folder)


        print("SVDFeature: Writing input file in feature format")

        self._write_feature_format_file()

        print("SVDFeature: Fit starting")

        args = ["svd_feature",
                #"active_type=3",
                "input_type=1",
                "data_in={}".format(self.temp_file_folder + self.FILE_MODEL_NAME),
                "model_out_folder={}".format(self.temp_file_folder),
                "num_item={:d}".format(self.n_item_features),
                "num_user={:d}".format(self.n_user_features),
                "num_global=0",
                "num_factor={:d}".format(num_factors),
                "base_score={:.2f}".format(self.URM_train.data.mean()),
                "learning_rate={}".format(learning_rate),
                "wd_user={}".format(user_reg),
                "wd_item={}".format(item_reg),
                "wd_user_bias={}".format(user_bias_reg),
                "wd_item_bias={}".format(item_bias_reg),
                "num_round=1",
                "train_repeat={:d}".format(epochs)]

        subprocess.run(args)
