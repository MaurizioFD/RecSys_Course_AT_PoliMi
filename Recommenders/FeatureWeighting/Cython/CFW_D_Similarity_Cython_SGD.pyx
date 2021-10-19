"""
Created on 09/09/17

@author: Maurizio Ferrari Dacrema
"""

#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: language_level=3
#cython: nonecheck=False
#cython: cdivision=True
#cython: unpack_method_calls=True
#cython: overflowcheck=False

#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
import time
import sys

from libc.math cimport exp, sqrt
from libc.stdlib cimport rand, RAND_MAX


cdef class CFW_D_Similarity_Cython_SGD:

    cdef double[:] data_list
    cdef int[:] row_list, col_list
    cdef double[:] D
    cdef int n_features


    cdef double learning_rate, l1_reg, l2_reg
    cdef int precompute_common_features
    cdef int[:] icm_indices, icm_indptr
    cdef int[:] common_features_id, common_features_flag
    cdef int[:] commonFeatures_indices, commonFeatures_indptr
    cdef double[:] icm_data, commonFeatures_data, common_features_data, common_features_data_i

    cdef int use_dropout
    cdef double dropout_perc
    cdef int[:] dropout_mask


    cdef int useAdaGrad, useRmsprop, useAdam, verbose
    cdef int positive_only_D

    cdef double [:] sgd_cache_D
    cdef double gamma

    cdef double [:] sgd_cache_D_momentum_1, sgd_cache_D_momentum_2
    cdef double beta_1, beta_2, beta_1_power_t, beta_2_power_t
    cdef double momentum_1, momentum_2


    def __init__(self, row_list, col_list, data_list,
                 n_features, ICM, precompute_common_features = True,
                 positive_only_D = True,
                 weights_initialization_D = None,
                 use_dropout = False,
                 dropout_perc = 0.3,
                 learning_rate = 0.05,
                 l1_reg = 0.0,
                 l2_reg = 0.0,
                 sgd_mode='adam',
                 verbose = False,
                 gamma=0.995, beta_1=0.9, beta_2=0.999):

        super(CFW_D_Similarity_Cython_SGD, self).__init__()

        self.row_list = np.array(row_list, dtype=np.int32)
        self.col_list = np.array(col_list, dtype=np.int32)
        self.data_list = np.array(data_list, dtype=np.float64)

        self.n_features = n_features
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

        self.positive_only_D = positive_only_D

        if weights_initialization_D is not None:
            self.D = np.array(weights_initialization_D, dtype=np.float64)
        else:
            self.D = np.zeros(self.n_features, dtype=np.float64)


        # RUN TEST
        self.run_tests()


        self.common_features_id = np.zeros((self.n_features), dtype=np.int32)
        self.common_features_data = np.zeros((self.n_features), dtype=np.float64)
        self.common_features_data_i = np.zeros((self.n_features), dtype=np.float64)
        self.common_features_flag = np.zeros((self.n_features), dtype=np.int32)

        self.precompute_common_features = precompute_common_features

        if self.precompute_common_features:
            self.precompute_common_features_function(ICM)

        else:
            self.icm_indices = np.array(ICM.indices, dtype=np.int32)
            self.icm_indptr = np.array(ICM.indptr, dtype=np.int32)
            self.icm_data = np.array(ICM.data, dtype=np.float64)







        self.use_dropout = use_dropout
        self.dropout_perc = dropout_perc
        self.dropout_mask = np.ones(self.n_features, dtype=np.int32)


        self.useAdaGrad = False
        self.useRmsprop = False
        self.useAdam = False


        if sgd_mode=='adagrad':
            self.useAdaGrad = True
            self.sgd_cache_D = np.zeros((self.n_features), dtype=np.float64)

        elif sgd_mode=='rmsprop':
            self.useRmsprop = True
            self.sgd_cache_D = np.zeros((self.n_features), dtype=np.float64)

            # Gamma default value suggested by Hinton
            # self.gamma = 0.9
            self.gamma = gamma

        elif sgd_mode=='adam':
            self.useAdam = True
            self.sgd_cache_D_momentum_1 = np.zeros((self.n_features), dtype=np.float64)
            self.sgd_cache_D_momentum_2 = np.zeros((self.n_features), dtype=np.float64)

            # Default value suggested by the original paper
            # beta_1=0.9, beta_2=0.999
            self.beta_1 = beta_1
            self.beta_2 = beta_2
            self.beta_1_power_t = beta_1
            self.beta_2_power_t = beta_2

        elif sgd_mode=='sgd':
            pass
        else:
            raise ValueError(
                "SGD_mode not valid. Acceptable values are: 'sgd', 'adagrad', 'rmsprop', 'adam'. Provided value was '{}'".format(
                    sgd_mode))


    def run_tests(self):

        n_test_features = 50

        self.common_features_id = np.zeros((n_test_features), dtype=np.int32)
        self.common_features_data = np.zeros((n_test_features), dtype=np.float64)
        self.common_features_data_i = np.zeros((n_test_features), dtype=np.float64)
        self.common_features_flag = np.zeros((n_test_features), dtype=np.int32)

        self.feature_common_test()



    def precompute_common_features_function(self, ICM):

        # Compute common features, to keep memory requirements low process one at a time
        commonFeatures = ICM[np.array(self.row_list)].multiply(ICM[np.array(self.col_list)])

        # Init Common features
        self.commonFeatures_indices = np.array(commonFeatures.indices, dtype=np.int32)
        self.commonFeatures_indptr = np.array(commonFeatures.indptr, dtype=np.int32)
        self.commonFeatures_data = np.array(commonFeatures.data, dtype=np.float64)



    def fit(self):

        cdef double similarity_value_target, similarity_value_weighted, similarity_value_unweighted, gradient, adaptive_gradient, error
        cdef long sample_num, n_samples
        cdef double cum_loss

        cdef int[:] f_row, f_col
        cdef double[:] f_row_data, f_col_data
        cdef int num_common_features

        cdef long sample_index
        cdef long weight_index

        cdef long feature_id, feature_index

        start_time_epoch = time.time()

        # Get number of available interactions
        n_samples = len(self.data_list)
        cum_loss = 0

        # Shuffle data
        cdef long[:] newOrdering = np.arange(n_samples)
        np.random.shuffle(newOrdering)

        cdef long dropout_threshold = long(RAND_MAX * self.dropout_perc)

        # Renew dropout mask
        if self.use_dropout:
            for feature_id in range(self.n_features):
                self.dropout_mask[feature_id] = rand() > dropout_threshold


        start_time_batch = time.time()

        for sample_num in range(n_samples):

            # Get next sample and compute its prediction using the current model
            sample_index = newOrdering[sample_num]

            similarity_value_target = self.data_list[sample_index]

            if self.precompute_common_features:
                num_common_features = self.feature_common_precomputed(sample_index)
            else:
                f_row = self.get_features_vector(self.row_list[sample_index])
                f_row_data = self.get_features_vector_data(self.row_list[sample_index])
                f_col = self.get_features_vector(self.col_list[sample_index])
                f_col_data = self.get_features_vector_data(self.col_list[sample_index])

                num_common_features = self.feature_common_unordered(f_row, f_col, f_row_data, f_col_data)


            similarity_value_weighted = 0.0
            similarity_value_unweighted = 0.0

            for feature_index in range(num_common_features):
                feature_id = self.common_features_id[feature_index]
                similarity_value_weighted += self.D[feature_id] * self.common_features_data[feature_index] * self.dropout_mask[feature_id]
                #similarity_value_unweighted += self.common_features_data[feature_index] * self.dropout_mask[feature_id]

            # The gradient is the prediction error
            error = similarity_value_weighted - similarity_value_target
            cum_loss += error**2


            # For every common feature update the corresponding weight
            for feature_index in range(num_common_features):

                feature_id = self.common_features_id[feature_index]

                if self.dropout_mask[feature_id]:

                    gradient = error * self.common_features_data[feature_index]

                    adaptive_gradient = self.compute_adaptive_gradient(feature_id, gradient)

                    self.D[feature_id] -= self.learning_rate * (adaptive_gradient + self.l1_reg + 2*self.l2_reg * self.D[feature_id])

                    # Clamp weight if needed
                    if self.positive_only_D and self.D[feature_id] < 0.0:
                        self.D[feature_id] = 0.0


            # Exponentiation of beta at the end of each sample
            if self.useAdam:

                self.beta_1_power_t *= self.beta_1
                self.beta_2_power_t *= self.beta_2


            if self.verbose and ((sample_num % 10000000 == 0 and sample_num!=0) or sample_num==n_samples-1):

                print("CFW_D_Similarity_Cython_SGD: Processed {} out of {} samples ( {:.2f}% ) in {:.2f} sec. Loss is {:.4E}. Sample per second: {:.0f}".format(
                      sample_num, n_samples,
                      100.0* float(sample_num)/n_samples,
                      time.time()-start_time_batch,
                      cum_loss / sample_num,
                      float(sample_num+1)/(time.time()-start_time_epoch)))

                # Flush buffer to ensure update progress is written on linux nohup file
                sys.stdout.flush()
                sys.stderr.flush()

                start_time_batch = time.time()


        return cum_loss / sample_num


    def get_weights(self):

        return np.array(self.D).copy()




    cdef double compute_adaptive_gradient(self, int feature_id, double gradient):

        cdef double adaptive_gradient


        if self.useAdaGrad:
            self.sgd_cache_D[feature_id] += gradient ** 2
            adaptive_gradient = self.learning_rate / (sqrt(self.sgd_cache_D[feature_id]) + 1e-8)



        elif self.useRmsprop:
            self.sgd_cache_D[feature_id] = self.sgd_cache_D[feature_id] * self.gamma + (1 - self.gamma) * gradient ** 2
            adaptive_gradient = self.learning_rate / (sqrt(self.sgd_cache_D[feature_id]) + 1e-8)


        elif self.useAdam:

            self.sgd_cache_D_momentum_1[feature_id] = \
                self.sgd_cache_D_momentum_1[feature_id] * self.beta_1 + (1 - self.beta_1) * gradient

            self.sgd_cache_D_momentum_2[feature_id] = \
                self.sgd_cache_D_momentum_2[feature_id] * self.beta_2 + (1 - self.beta_2) * gradient**2


            self.momentum_1 = self.sgd_cache_D_momentum_1[feature_id]/ (1 - self.beta_1_power_t)
            self.momentum_2 = self.sgd_cache_D_momentum_2[feature_id]/ (1 - self.beta_2_power_t)

            adaptive_gradient = self.momentum_1/ (sqrt(self.momentum_2) + 1e-8)

        else:

            adaptive_gradient = gradient



        return adaptive_gradient




















    # Using memoryview instead of the sparse matrix itself allows for much faster access
    cdef int feature_common_precomputed(self, long sample_index):

        self.common_features_id = self.commonFeatures_indices[self.commonFeatures_indptr[sample_index]:self.commonFeatures_indptr[sample_index + 1]]
        self.common_features_data = self.commonFeatures_data[self.commonFeatures_indptr[sample_index]:self.commonFeatures_indptr[sample_index + 1]]

        return len(self.common_features_id)


    cdef int[:] get_features_vector(self, long index):
        return self.icm_indices[self.icm_indptr[index]:self.icm_indptr[index + 1]]

    cdef double[:] get_features_vector_data(self, long index):
        return self.icm_data[self.icm_indptr[index]:self.icm_indptr[index + 1]]


    cdef int feature_common_unordered(self, int[:] feature_i, int[:] feature_j, double[:] data_i, double[:] data_j):

        cdef int common_features_count = 0
        cdef int feature_index, feature_id

        for feature_index in range(len(feature_i)):

            feature_id = feature_i[feature_index]
            self.common_features_flag[feature_id] = True
            self.common_features_data_i[feature_id] = data_i[feature_index]

        for feature_index in range(len(feature_j)):

            feature_id = feature_j[feature_index]

            if self.common_features_flag[feature_id]:
                self.common_features_id[common_features_count] = feature_id
                self.common_features_data[common_features_count] = self.common_features_data_i[feature_id] * data_j[feature_index]
                common_features_count += 1

        # Clear flag data structure
        for feature_index in range(len(feature_i)):
            feature_id = feature_i[feature_index]
            self.common_features_flag[feature_id] = False

        return common_features_count






    def feature_common_test(self):

        if self.verbose:
            print("CFW_D_Similarity_Cython: feature_common_test")

        # No duplicates
        f_i = np.array([0, 5, 8, 10, 19], dtype=np.int32)
        f_j = np.array([1, 7, 9, 11, 15, 18], dtype=np.int32)

        common_np = np.intersect1d(f_i, f_j)
        count = self.feature_common_unordered(f_i, f_j, np.ones_like(f_i, dtype=np.float64), np.ones_like(f_j, dtype=np.float64))
        common_cy = self.common_features_id[0:count]
        common_cy = np.sort(np.array(common_cy))

        assert np.array_equal(common_np, common_cy), "Common unordered is different. F_i: {}, F_j: {}. Union Numpy: {}, Common Cython: {}".format(
            f_i, f_j, common_np, common_cy)


        # Duplicates
        f_i = np.array([0, 5, 8, 10, 19], dtype=np.int32)
        f_j = np.array([1, 10, 11], dtype=np.int32)

        common_np = np.intersect1d(f_i, f_j)
        count = self.feature_common_unordered(f_i, f_j, np.ones_like(f_i, dtype=np.float64), np.ones_like(f_j, dtype=np.float64))
        common_cy = self.common_features_id[0:count]
        common_cy = np.sort(np.array(common_cy))

        assert np.array_equal(common_np, common_cy), "Common unordered is different. F_i: {}, F_j: {}. Union Numpy: {}, Common Cython: {}".format(
            f_i, f_j, common_np, common_cy)



        f_i = np.array([0, 5, 8, 10, 19], dtype=np.int32)
        f_j = np.array([1, 2, 5, 10, 15], dtype=np.int32)

        common_np = np.intersect1d(f_i, f_j)
        count = self.feature_common_unordered(f_i, f_j, np.ones_like(f_i, dtype=np.float64), np.ones_like(f_j, dtype=np.float64))
        common_cy = self.common_features_id[0:count]
        common_cy = np.sort(np.array(common_cy))

        assert np.array_equal(common_np, common_cy), "Common unordered is different. F_i: {}, F_j: {}. Union Numpy: {}, Common Cython: {}".format(
            f_i, f_j, common_np, common_cy)



        for _ in range(100):

            size_f_i = np.random.randint(1,10)
            f_i = np.random.randint(0,50, size=size_f_i, dtype=np.int32)
            f_i = np.unique(f_i)

            size_f_j = np.random.randint(1,10)
            f_j = np.random.randint(0,50, size=size_f_j, dtype=np.int32)
            f_j = np.unique(f_j)

            common_np = np.intersect1d(f_i, f_j)
            count = self.feature_common_unordered(f_i, f_j, np.ones_like(f_i, dtype=np.float64), np.ones_like(f_j, dtype=np.float64))
            common_cy = self.common_features_id[0:count]
            common_cy = np.sort(np.array(common_cy))

            assert np.array_equal(common_np, common_cy), "Common unordered is different. F_i: {}, F_j: {}. Common Numpy: {}, Common Cython: {}".format(
                f_i, f_j, common_np, common_cy)



        if self.verbose:
            print("CFW_D_Similarity_Cython: All tests passed")

