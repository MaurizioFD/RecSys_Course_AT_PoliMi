#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/06/18

@author: Maurizio Ferrari Dacrema
"""

from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender

from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sps


class PureSVDRecommender(Recommender):
    """ PureSVDRecommender"""

    RECOMMENDER_NAME = "PureSVDRecommender"

    def __init__(self, URM_train):
        super(PureSVDRecommender, self).__init__()

        # CSR is faster during evaluation
        self.URM_train = check_matrix(URM_train, 'csr')

        self.compute_item_score = self.compute_score_SVD


    def fit(self, num_factors=100):

        from sklearn.utils.extmath import randomized_svd

        print(self.RECOMMENDER_NAME + " Computing SVD decomposition...")

        self.U, self.Sigma, self.VT = randomized_svd(self.URM_train,
                                      n_components=num_factors,
                                      #n_iter=5,
                                      random_state=None)

        self.s_Vt = sps.diags(self.Sigma)*self.VT

        print(self.RECOMMENDER_NAME + " Computing SVD decomposition... Done!")

        # truncatedSVD = TruncatedSVD(n_components = num_factors)
        #
        # truncatedSVD.fit(self.URM_train)
        #
        # truncatedSVD

        #U, s, Vt =



    def compute_score_SVD(self, user_id_array):

        try:

            item_weights = self.U[user_id_array, :].dot(self.s_Vt)

        except:
            pass

        return item_weights













    def saveModel(self, folder_path, file_name = None, forceSparse = True):

        import pickle

        print("{}: Saving model in folder '{}'".format(self.RECOMMENDER_NAME, folder_path))

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        file_name += "_"

        data_dict = {
            "U":self.U,
            "Sigma":self.Sigma,
            "VT":self.VT,
            "s_Vt":self.s_Vt
        }

        pickle.dump(data_dict,
                    open(folder_path + file_name + "model", "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)



    def loadModel(self, folderPath, namePrefix = None, forceSparse = True):

        import pickle

        print("{}: Loading model from folder '{}'".format(self.RECOMMENDER_NAME, folderPath))

        if namePrefix is None:
            namePrefix = self.RECOMMENDER_NAME

        namePrefix += "_"

        data_dict = pickle.load(open(folderPath + namePrefix + "model", "rb"))

        for attrib_name in data_dict.keys():
             self.__setattr__(attrib_name, data_dict[attrib_name])
