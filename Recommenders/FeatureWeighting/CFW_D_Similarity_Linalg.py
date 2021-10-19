#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/09/17

@author: Maurizio Ferrari Dacrema
"""

from FeatureWeighting.Cython.CFW_D_Similarity_Cython import CFW_D_Similarity_Cython


from scipy.sparse import linalg




class CFW_D_Similarity_Linalg(CFW_D_Similarity_Cython):

    RECOMMENDER_NAME = "CFW_D_Similarity_Linalg"


    def __init__(self, URM_train, ICM_train, S_matrix_target):
        super(CFW_D_Similarity_Linalg, self).__init__(URM_train, ICM_train, S_matrix_target, recompile_cython=False)


    def fit(self, show_max_performance = False,
            logFile = None,
            loss_tolerance = 1e-6,
            iteration_limit = 50000,
            damp_coeff=0.0,
            topK = 300,
            add_zeros_quota = 0.0,
            normalize_similarity = False):


        self.logFile = logFile

        self.show_max_performance = show_max_performance
        self.add_zeros_quota = add_zeros_quota
        self.normalize_similarity = normalize_similarity
        self.topK = topK

        self._generate_train_data()

        commonFeatures = self.ICM[self.row_list].multiply(self.ICM[self.col_list])

        assert False, "check consistency of train data"

        linalg_result = linalg.lsqr(commonFeatures,
                                    self.data_list,
                                    show = False,
                                    atol=loss_tolerance,
                                    btol=loss_tolerance,
                                    iter_lim = iteration_limit,
                                    damp=damp_coeff)

        # res = linalg.lsmr(commonFeatures, self.data_list, show = False, atol=loss_tolerance, btol=loss_tolerance,
        #                   maxiter = iteration_limit, damp=damp_coeff)



        self.D_incremental = linalg_result[0].copy()
        self.D_best = linalg_result[0].copy()
        self.loss = linalg_result[3]

        self.compute_W_sparse()

