#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Massimo Quadrana
"""

import logging

import numpy as np
from Base.Recommender_utils import check_matrix

from Base.Recommender import Recommender
from MatrixFactorization.Cython.MF_RMSE import FunkSVD_sgd, AsySVD_sgd, AsySVD_compute_user_factors, BPRMF_sgd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")




class FunkSVD(Recommender):
    '''
    FunkSVD model
    Reference: http://sifter.org/~simon/journal/20061211.html

    Factorizes the rating matrix R into the dot product of two matrices U and V of latent factors.
    U represent the user latent factors, V the item latent factors.
    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin} \limits_{U,V}\frac{1}{2}||R - UV^T||^2_2 + \frac{\lambda}{2}(||U||^2_F + ||V||^2_F)
    Latent factors are initialized from a Normal distribution with given mean and std.
    '''

    RECOMMENDER_NAME = "FunkSVD"

    # TODO: add global effects
    def __init__(self, URM_train):

        super(FunkSVD, self).__init__()

        self.URM_train = check_matrix(URM_train, 'csr', dtype=np.float32)



    def __str__(self):
        return "FunkSVD(num_factors={}, lrate={}, reg={}, iters={}, init_mean={}, " \
               "init_std={}, lrate_decay={}, rnd_seed={})".format(
            self.num_factors, self.learning_rate, self.reg, self.epochs, self.init_mean, self.init_std, self.lrate_decay,
            self.rnd_seed
        )


    def fit(self, num_factors=50,
                 learning_rate=0.01,
                 reg=0.015,
                 epochs=10,
                 init_mean=0.0,
                 init_std=0.1,
                 lrate_decay=1.0,
                 rnd_seed=42):
        """

        Initialize the model
        :param num_factors: number of latent factors
        :param learning_rate: initial learning rate used in SGD
        :param reg: regularization term
        :param epochs: number of iterations in training the model with SGD
        :param init_mean: mean used to initialize the latent factors
        :param init_std: standard deviation used to initialize the latent factors
        :param lrate_decay: learning rate decay
        :param rnd_seed: random seed
        """

        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.epochs = epochs
        self.init_mean = init_mean
        self.init_std = init_std
        self.lrate_decay = lrate_decay
        self.rnd_seed = rnd_seed

        self.U, self.V = FunkSVD_sgd(self.URM_train, self.num_factors, self.learning_rate, self.reg, self.epochs, self.init_mean,
                                     self.init_std,
                                     self.lrate_decay, self.rnd_seed)

    # def recommend(self, user_id, n=None, exclude_seen=True):
    #     scores = np.dot(self.U[user_id], self.V.T)
    #     ranking = scores.argsort()[::-1]
    #     # rank items
    #     if exclude_seen:
    #         ranking = self._filter_seen(user_id, ranking)
    #     return ranking[:n]
    #
    #
    # def _get_user_ratings(self, user_id):
    #     return self.dataset[user_id]
    #
    # def _get_item_ratings(self, item_id):
    #     return self.dataset[:, item_id]
    #
    #
    # def _filter_seen(self, user_id, ranking):
    #     user_profile = self._get_user_ratings(user_id)
    #     seen = user_profile.indices
    #     unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
    #     return ranking[unseen_mask]





    def recommendBatch(self, users_in_batch, n=None, exclude_seen=True, filterTopPop = False, filterCustomItems = False):

        # compute the scores using the dot product
        user_profile_batch = self.URM_train[users_in_batch]

        scores_array = np.dot(self.U[users_in_batch], self.V.T)

        if self.normalize:
            raise ValueError("Not implemented")

        # To exclude seen items perform a boolean indexing and replace their score with -inf
        # Seen items will be at the bottom of the list but there is no guarantee they'll NOT be
        # recommended
        if exclude_seen:
            scores_array[user_profile_batch.nonzero()] = -np.inf

        if filterTopPop:
            scores_array[:,self.filterTopPop_ItemsID] = -np.inf

        if filterCustomItems:
            scores_array[:, self.filterCustomItems_ItemsID] = -np.inf


        # rank items and mirror column to obtain a ranking in descending score
        #ranking = (-scores_array).argsort(axis=1)
        #ranking = np.fliplr(ranking)
        #ranking = ranking[:,0:n]

        ranking = np.zeros((scores_array.shape[0],n), dtype=np.int)

        for row_index in range(scores_array.shape[0]):
            scores = scores_array[row_index]

            relevant_items_partition = (-scores).argpartition(n)[0:n]
            relevant_items_partition_sorting = np.argsort(-scores[relevant_items_partition])
            ranking[row_index] = relevant_items_partition[relevant_items_partition_sorting]


        return ranking



    def recommend(self, user_id, cutoff=None, remove_seen_flag=True, remove_top_pop_flag = False, remove_CustomItems = False):


        if cutoff==None:
            cutoff= self.URM_train.shape[1] - 1

        scores_array = np.dot(self.U[user_id], self.V.T)

        if self.normalize:
            raise ValueError("Not implemented")


        if remove_seen_flag:
            scores = self._remove_seen_on_scores(user_id, scores_array)

        if remove_top_pop_flag:
            scores = self._remove_TopPop_on_scores(scores_array)

        if remove_CustomItems:
            scores = self._remove_CustomItems_on_scores(scores_array)


        # rank items and mirror column to obtain a ranking in descending score
        #ranking = scores.argsort()
        #ranking = np.flip(ranking, axis=0)

        # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
        # - Partition the data to extract the set of relevant items
        # - Sort only the relevant items
        # - Get the original item index
        relevant_items_partition = (-scores_array).argpartition(cutoff)[0:cutoff]
        relevant_items_partition_sorting = np.argsort(-scores_array[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]


        return ranking



    def saveModel(self, folderPath, namePrefix = None, forceSparse = True):

        print("{}: Saving model in folder '{}'".format(self.RECOMMENDER_NAME, folderPath))

        if namePrefix is None:
            namePrefix = self.RECOMMENDER_NAME

        namePrefix += "_"

        np.savez(folderPath + "{}.npz".format(namePrefix), W = self.U, H = self.V)



    def loadModel(self, folderPath, namePrefix = None, forceSparse = True):


        print("{}: Loading model from folder '{}'".format(self.RECOMMENDER_NAME, folderPath))

        if namePrefix is None:
            namePrefix = self.RECOMMENDER_NAME

        namePrefix += "_"

        npzfile = np.load(folderPath + "{}.npz".format(namePrefix))

        for attrib_name in npzfile.files:
             self.__setattr__(attrib_name, npzfile[attrib_name])



class AsySVD(Recommender):
    '''
    AsymmetricSVD model
    Reference: Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model (Koren, 2008)

    Factorizes the rating matrix R into two matrices X and Y of latent factors, which both represent item latent features.
    Users are represented by aggregating the latent features in Y of items they have already rated.
    Rating prediction is performed by computing the dot product of this accumulated user profile with the target item's
    latent factor in X.

    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin}\limits_{x*,y*}\frac{1}{2}\sum_{i,j \in R}(r_{ij} - x_j^T \sum_{l \in R(i)} r_{il}y_l)^2 + \frac{\lambda}{2}(\sum_{i}{||x_i||^2} + \sum_{j}{||y_j||^2})
    '''

    # TODO: add global effects
    # TODO: recommendation for new-users. Update the precomputed profiles online
    def __init__(self,
                 num_factors=50,
                 lrate=0.01,
                 reg=0.015,
                 iters=10,
                 init_mean=0.0,
                 init_std=0.1,
                 lrate_decay=1.0,
                 rnd_seed=42):
        '''
        Initialize the model
        :param num_factors: number of latent factors
        :param lrate: initial learning rate used in SGD
        :param reg: regularization term
        :param iters: number of iterations in training the model with SGD
        :param init_mean: mean used to initialize the latent factors
        :param init_std: standard deviation used to initialize the latent factors
        :param lrate_decay: learning rate decay
        :param rnd_seed: random seed
        '''
        super(AsySVD, self).__init__()
        self.num_factors = num_factors
        self.lrate = lrate
        self.reg = reg
        self.iters = iters
        self.init_mean = init_mean
        self.init_std = init_std
        self.lrate_decay = lrate_decay
        self.rnd_seed = rnd_seed

    def __str__(self):
        return "AsySVD(num_factors={}, lrate={}, reg={}, iters={}, init_mean={}, " \
               "init_std={}, lrate_decay={}, rnd_seed={})".format(
            self.num_factors, self.lrate, self.reg, self.iters, self.init_mean, self.init_std, self.lrate_decay,
            self.rnd_seed
        )

    def fit(self, R):
        self.dataset = R
        R = check_matrix(R, 'csr', dtype=np.float32)
        self.X, self.Y = AsySVD_sgd(R, self.num_factors, self.lrate, self.reg, self.iters, self.init_mean,
                                    self.init_std,
                                    self.lrate_decay, self.rnd_seed)
        # precompute the user factors
        M = R.shape[0]
        self.U = np.vstack([AsySVD_compute_user_factors(R[i], self.Y) for i in range(M)])

    def recommend(self, user_id, cutoff=None, remove_seen_flag=True):
        scores = np.dot(self.X, self.U[user_id].T)
        ranking = scores.argsort()[::-1]
        # rank items
        if remove_seen_flag:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:cutoff]


    def _get_user_ratings(self, user_id):
        return self.dataset[user_id]

    def _get_item_ratings(self, item_id):
        return self.dataset[:, item_id]


    def _filter_seen(self, user_id, ranking):
        user_profile = self._get_user_ratings(user_id)
        seen = user_profile.indices
        unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
        return ranking[unseen_mask]



class IALS_numpy(Recommender):
    '''
    binary Alternating Least Squares model (or Weighed Regularized Matrix Factorization)
    Reference: Collaborative Filtering for binary Feedback Datasets (Hu et al., 2008)

    Factorization model for binary feedback.
    First, splits the feedback matrix R as the element-wise a Preference matrix P and a Confidence matrix C.
    Then computes the decomposition of them into the dot product of two matrices X and Y of latent factors.
    X represent the user latent factors, Y the item latent factors.

    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin}\limits_{x*,y*}\frac{1}{2}\sum_{i,j}{c_{ij}(p_{ij}-x_i^T y_j) + \lambda(\sum_{i}{||x_i||^2} + \sum_{j}{||y_j||^2})}
    '''

    # TODO: Add support for multiple confidence scaling functions (e.g. linear and log scaling)
    def __init__(self,
                 num_factors=50,
                 reg=0.015,
                 iters=10,
                 scaling='linear',
                 alpha=40,
                 epsilon=1.0,
                 init_mean=0.0,
                 init_std=0.1,
                 rnd_seed=42):
        '''
        Initialize the model
        :param num_factors: number of latent factors
        :param reg: regularization term
        :param iters: number of iterations in training the model with SGD
        :param scaling: supported scaling modes for the observed values: 'linear' or 'log'
        :param alpha: scaling factor to compute confidence scores
        :param epsilon: epsilon used in log scaling only
        :param init_mean: mean used to initialize the latent factors
        :param init_std: standard deviation used to initialize the latent factors
        :param rnd_seed: random seed
        '''

        super(IALS_numpy, self).__init__()
        assert scaling in ['linear', 'log'], 'Unsupported scaling: {}'.format(scaling)

        self.num_factors = num_factors
        self.reg = reg
        self.iters = iters
        self.scaling = scaling
        self.alpha = alpha
        self.epsilon = epsilon
        self.init_mean = init_mean
        self.init_std = init_std
        self.rnd_seed = rnd_seed

    def __str__(self):
        return "WRMF-iALS(num_factors={},  reg={}, iters={}, scaling={}, alpha={}, episilon={}, init_mean={}, " \
               "init_std={}, rnd_seed={})".format(
            self.num_factors, self.reg, self.iters, self.scaling, self.alpha, self.epsilon, self.init_mean,
            self.init_std, self.rnd_seed
        )

    def _linear_scaling(self, R):
        C = R.copy().tocsr()
        C.data *= self.alpha
        C.data += 1.0
        return C

    def _log_scaling(self, R):
        C = R.copy().tocsr()
        C.data = 1.0 + self.alpha * np.log(1.0 + C.data / self.epsilon)
        return C

    def fit(self, R):
        self.dataset = R
        # compute the confidence matrix
        if self.scaling == 'linear':
            C = self._linear_scaling(R)
        else:
            C = self._log_scaling(R)

        Ct = C.T.tocsr()
        M, N = R.shape

        # set the seed
        np.random.seed(self.rnd_seed)

        # initialize the latent factors
        self.X = np.random.normal(self.init_mean, self.init_std, size=(M, self.num_factors))
        self.Y = np.random.normal(self.init_mean, self.init_std, size=(N, self.num_factors))

        for it in range(self.iters):
            self.X = self._lsq_solver_fast(C, self.X, self.Y, self.reg)
            self.Y = self._lsq_solver_fast(Ct, self.Y, self.X, self.reg)
            logger.debug('Finished iter {}'.format(it + 1))

    def recommend(self, user_id, cutoff=None, remove_seen_flag=True):
        scores = np.dot(self.X[user_id], self.Y.T)
        ranking = scores.argsort()[::-1]
        # rank items
        if remove_seen_flag:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:cutoff]

    def _lsq_solver(self, C, X, Y, reg):
        # precompute YtY
        rows, factors = X.shape
        YtY = np.dot(Y.T, Y)

        for i in range(rows):
            # accumulate YtCiY + reg*I in A
            A = YtY + reg * np.eye(factors)

            # accumulate Yt*Ci*p(i) in b
            b = np.zeros(factors)

            for j, cij in self._nonzeros(C, i):
                vj = Y[j]
                A += (cij - 1.0) * np.outer(vj, vj)
                b += cij * vj

            X[i] = np.linalg.solve(A, b)
        return X

    def _lsq_solver_fast(self, C, X, Y, reg):
        # precompute YtY
        rows, factors = X.shape
        YtY = np.dot(Y.T, Y)

        for i in range(rows):
            # accumulate YtCiY + reg*I in A
            A = YtY + reg * np.eye(factors)

            start, end = C.indptr[i], C.indptr[i + 1]
            j = C.indices[start:end]  # indices of the non-zeros in Ci
            ci = C.data[start:end]  # non-zeros in Ci

            Yj = Y[j]  # only the factors with non-zero confidence
            # compute Yt(Ci-I)Y
            aux = np.dot(Yj.T, np.diag(ci - 1.0))
            A += np.dot(aux, Yj)
            # compute YtCi
            b = np.dot(Yj.T, ci)

            X[i] = np.linalg.solve(A, b)
        return X

    def _nonzeros(self, R, row):
        for i in range(R.indptr[row], R.indptr[row + 1]):
            yield (R.indices[i], R.data[i])


    def _get_user_ratings(self, user_id):
        return self.dataset[user_id]

    def _get_item_ratings(self, item_id):
        return self.dataset[:, item_id]


    def _filter_seen(self, user_id, ranking):
        user_profile = self._get_user_ratings(user_id)
        seen = user_profile.indices
        unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
        return ranking[unseen_mask]



class BPRMF(Recommender):
    '''
    BPRMF model
    '''

    # TODO: add global effects
    def __init__(self,
                 num_factors=50,
                 lrate=0.01,
                 user_reg=0.015,
                 pos_reg=0.015,
                 neg_reg=0.0015,
                 iters=10,
                 sampling_type='user_uniform_item_uniform',
                 sample_with_replacement=True,
                 use_resampling=True,
                 sampling_pop_alpha=1.0,
                 init_mean=0.0,
                 init_std=0.1,
                 lrate_decay=1.0,
                 rnd_seed=42,
                 verbose=True):
        '''
        Initialize the model
        :param num_factors: number of latent factors
        :param lrate: initial learning rate used in SGD
        :param user_reg: regularization for the user factors
        :param pos_reg: regularization for the factors of the positive sampled items
        :param neg_reg: regularization for the factors of the negative sampled items
        :param iters: number of iterations in training the model with SGD
        :param sampling_type: type of sampling. Supported types are 'user_uniform_item_uniform' and 'user_uniform_item_pop'
        :param sample_with_replacement: `True` to sample positive items with replacement (doesn't work with 'user_uniform_item_pop')
        :param use_resampling: `True` to resample at each iteration during training
        :param sampling_pop_alpha: float smoothing factor for popularity based samplers (e.g., 'user_uniform_item_pop')
        :param init_mean: mean used to initialize the latent factors
        :param init_std: standard deviation used to initialize the latent factors
        :param lrate_decay: learning rate decay
        :param rnd_seed: random seed
        :param verbose: controls verbosity in output
        '''
        super(BPRMF, self).__init__()
        self.num_factors = num_factors
        self.lrate = lrate
        self.user_reg = user_reg
        self.pos_reg = pos_reg
        self.neg_reg = neg_reg
        self.iters = iters
        self.sampling_type = sampling_type
        self.sample_with_replacement = sample_with_replacement
        self.use_resampling = use_resampling
        self.sampling_pop_alpha = sampling_pop_alpha
        self.init_mean = init_mean
        self.init_std = init_std
        self.lrate_decay = lrate_decay
        self.rnd_seed = rnd_seed
        self.verbose = verbose

    def __str__(self):
        return "BPRMF(num_factors={}, lrate={}, user_reg={}. pos_reg={}, neg_reg={}, iters={}, " \
               "sampling_type={}, sample_with_replacement={}, use_resampling={}, sampling_pop_alpha={}, init_mean={}, " \
               "init_std={}, lrate_decay={}, rnd_seed={}, verbose={})".format(
            self.num_factors, self.lrate, self.user_reg, self.pos_reg, self.neg_reg, self.iters,
            self.sampling_type, self.sample_with_replacement, self.use_resampling, self.sampling_pop_alpha,
            self.init_mean,
            self.init_std,
            self.lrate_decay,
            self.rnd_seed,
            self.verbose
        )

    def fit(self, R):
        self.dataset = R
        R = check_matrix(R, 'csr', dtype=np.float32)
        self.X, self.Y = BPRMF_sgd(R,
                                   num_factors=self.num_factors,
                                   lrate=self.lrate,
                                   user_reg=self.user_reg,
                                   pos_reg=self.pos_reg,
                                   neg_reg=self.neg_reg,
                                   iters=self.iters,
                                   sampling_type=self.sampling_type,
                                   sample_with_replacement=self.sample_with_replacement,
                                   use_resampling=self.use_resampling,
                                   sampling_pop_alpha=self.sampling_pop_alpha,
                                   init_mean=self.init_mean,
                                   init_std=self.init_std,
                                   lrate_decay=self.lrate_decay,
                                   rnd_seed=self.rnd_seed,
                                   verbose=self.verbose)

    def recommend(self, user_id, cutoff=None, remove_seen_flag=True):
        scores = np.dot(self.X[user_id], self.Y.T)
        ranking = scores.argsort()[::-1]
        # rank items
        if remove_seen_flag:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:cutoff]



    def _get_user_ratings(self, user_id):
        return self.dataset[user_id]

    def _get_item_ratings(self, item_id):
        return self.dataset[:, item_id]


    def _filter_seen(self, user_id, ranking):
        user_profile = self._get_user_ratings(user_id)
        seen = user_profile.indices
        unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
        return ranking[unseen_mask]

