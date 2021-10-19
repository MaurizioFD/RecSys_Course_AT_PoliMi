"""
Created on 10/17

@author: Alberto Gasparin, Maurizio Ferrari Dacrema
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
import scipy.sparse as sps
from Recommenders.Recommender_utils import check_matrix

from libc.math cimport sqrt
from cpython.array cimport array, clone


cdef class CFW_DVV_Similarity_Cython_SGD:

    cdef double[:] data_list
    cdef int[:] row_list, col_list
    cdef int positive_only_D, positive_only_V, precompute_common_features

    cdef double[:] D
    cdef double[:,:] V

    cdef double learning_rate, l2_reg_D, l2_reg_V, add_zeros_quota
    cdef int[:] icm_indices, icm_indptr
    cdef int n_features, n_factors, n_samples, n_items

    cdef int[:] union_features_id, union_features_flag
    cdef int[:] common_features_id, common_features_flag
    cdef int[:] commonFeatures_indices, commonFeatures_indptr

    cdef int useAdaGrad, useRmsprop, useAdam, verbose

    cdef double [:] sgd_cache_D
    cdef double [:,:] sgd_cache_V
    cdef double gamma

    cdef double [:] sgd_cache_D_momentum_1, sgd_cache_D_momentum_2
    cdef double [:,:] sgd_cache_V_momentum_1, sgd_cache_V_momentum_2
    cdef double beta_1, beta_2, beta_1_power_t, beta_2_power_t
    cdef double momentum_1, momentum_2


    cdef ICM


    def __init__(self, row_list, col_list, data_list,
                 ICM, add_zeros_quota = 0.0, n_factors = 1,
                 precompute_common_features = True,
                 weights_initialization_D = None,
                 weights_initialization_V = None,
                 positive_only_D = True,
                 positive_only_V = True,
                 verbose = False,
                 learning_rate = 0.001, l2_reg_D = 0.0, l2_reg_V = 0.0, sgd_mode='adagrad',
                 gamma = 0.995, beta_1=0.9, beta_2=0.999, mean_init = 0.0, std_init = 0.001):


        self.row_list = np.array(row_list, dtype=np.int32)
        self.col_list = np.array(col_list, dtype=np.int32)
        self.data_list = np.array(data_list, dtype=np.float64)

        self.n_samples = len(self.row_list)

        self.ICM = check_matrix(ICM, 'csr').astype(np.bool)


        self.n_items = ICM.shape[0]
        self.n_features = ICM.shape[1]
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.add_zeros_quota = add_zeros_quota
        self.l2_reg_D = l2_reg_D
        self.l2_reg_V = l2_reg_V
        self.positive_only_D = positive_only_D
        self.positive_only_V = positive_only_V
        self.verbose = verbose

        self.precompute_common_features = precompute_common_features


        if self.precompute_common_features:
            self.precompute_common_features_function()


        if weights_initialization_D is not None:
            assert np.array(weights_initialization_D).ravel().shape == (self.n_features,), \
                "CFW_DVV_Similarity_Cython_SGD: Wrong shape for weights_initialization_D, received was {}, expected was {}.".format(
                    np.array(weights_initialization_D).ravel().shape, (self.n_features,))

            self.D = np.array(weights_initialization_D, dtype=np.float64)
        else:
            self.D = np.zeros(self.n_features, dtype=np.float64)

        if weights_initialization_V is not None:
            assert np.array(weights_initialization_V).ravel().shape == (self.n_factors, self.n_features), \
                "CFW_DVV_Similarity_Cython_SGD: Wrong shape for weights_initialization_V, received was {}, expected was {}.".format(
                    np.array(weights_initialization_V).ravel().shape, (self.n_factors, self.n_features))

            self.V = np.array(weights_initialization_V, dtype=np.float64)
        else:
            self.V = np.random.normal(mean_init, std_init, (self.n_factors, self.n_features)).astype(np.float64)



        # Init ICM indices
        self.icm_indices = self.ICM.indices
        self.icm_indptr = self.ICM.indptr


        self.union_features_id = np.zeros((self.n_features), dtype=np.int32)
        self.union_features_flag = np.zeros((self.n_features), dtype=np.int32)

        self.common_features_id = np.zeros((self.n_features), dtype=np.int32)
        self.common_features_flag = np.zeros((self.n_features), dtype=np.int32)

        self.feature_union_test()
        self.feature_common_test()
        #self.precompute_common_features_function_test()


        self.useAdaGrad = False
        self.useRmsprop = False
        self.useAdam = False

        if sgd_mode=='adagrad':
            self.useAdaGrad = True
            self.sgd_cache_D = np.zeros((self.n_features), dtype=np.float64)
            self.sgd_cache_V = np.zeros((self.n_factors, self.n_features), dtype=np.float64)

        elif sgd_mode=='rmsprop':
            self.useRmsprop = True
            self.sgd_cache_D = np.zeros((self.n_features), dtype=np.float64)
            self.sgd_cache_V = np.zeros((self.n_factors, self.n_features), dtype=np.float64)

            # Gamma default value suggested by Hinton
            # self.gamma = 0.9
            self.gamma = gamma

        elif sgd_mode=='adam':
            self.useAdam = True
            self.sgd_cache_D_momentum_1 = np.zeros((self.n_features), dtype=np.float64)
            self.sgd_cache_D_momentum_2 = np.zeros((self.n_features), dtype=np.float64)

            self.sgd_cache_V_momentum_1 = np.zeros((self.n_factors, self.n_features), dtype=np.float64)
            self.sgd_cache_V_momentum_2 = np.zeros((self.n_factors, self.n_features), dtype=np.float64)

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




    def precompute_common_features_function(self):

        # Compute common features, to keep memory requirements low process one at a time
        commonFeatures = self.ICM[np.array(self.row_list)].multiply(self.ICM[np.array(self.col_list)])

        # Init Common features
        self.commonFeatures_indices = commonFeatures.indices
        self.commonFeatures_indptr = commonFeatures.indptr






    def fit(self):

        cdef int index, feature_id, feature_index, factor_index, num_common_features, num_union_features

        cdef double cum_loss = 0.0, s_ij=0.0, s_ij_hat=0.0, loss=0.0, loss_deriv=0.0, learning_rate_current_feature=0.0, gradient_update = 0.0

        cdef double D_hat, V_hat, error

        cdef double V_fi_k, V_fj_k

        cdef int[:] fi, fj

        cdef int[:] samples = np.arange(self.n_samples).astype(np.intc)
        np.random.shuffle(samples)

        cdef int sample_index

        start_time = time.time()

        # Start SGD
        for index in range(self.n_samples):

            # Draw sample

            # Initialize samples
            sample_index = samples[index]
            s_ij = self.data_list[sample_index]

            # Initialize needed variables
            fi = self.get_features_vector(self.row_list[sample_index])
            fj = self.get_features_vector(self.col_list[sample_index])

            # indices of the common features
            if self.precompute_common_features:
                num_common_features = self.feature_common_precomputed(sample_index)
            else:
                num_common_features = self.feature_common_unordered(fi, fj)

            # indices of the features appearing in fi or fj
            num_union_features = self.feature_union_unordered(fi, fj)

            D_hat = 0.0

            # Compute predicted similarity (s_ij_hat)
            for feature_index in range(num_common_features):
                feature_id = self.common_features_id[feature_index]
                D_hat += self.D[feature_id]


            V_hat = 0.0

            for factor_index in range(self.n_factors):
                V_fi_k = 0.0
                V_fj_k = 0.0

                for feature_index in range(len(fi)):
                    feature_id = fi[feature_index]
                    V_fi_k += self.V[factor_index, feature_id]

                for feature_index in range(len(fj)):
                    feature_id = fj[feature_index]
                    V_fj_k += self.V[factor_index, feature_id]

                V_hat += V_fi_k*V_fj_k

            s_ij_hat = D_hat + V_hat

            #Compute loss
            loss = 0.5 * (s_ij_hat - s_ij)**2
            cum_loss += loss

            #Compute derivatives
            error = s_ij_hat - s_ij


            # Parameters update for d
            for feature_index in range(num_common_features):

                feature_id = self.common_features_id[feature_index]

                adaptive_gradient = self.compute_adaptive_gradient_D(feature_id, error)
                self.D[feature_id] -= self.learning_rate * (adaptive_gradient + 2*self.l2_reg_D * self.D[feature_id])


                #Limit weights values between 0 and 1
                if self.positive_only_D and self.D[feature_id] < 0.0:
                    self.D[feature_id] = 0.0


            # Parameters update for V
            for feature_index in range(num_union_features):

                feature_id = self.union_features_id[feature_index] # The features that need updates for each factor

                for factor_index in range(self.n_factors):

                    #gradient_update = loss_deriv * self.deriv_V(k, c, fi, fj) + self.l1_reg + 2*self.l2_reg * self.V[k,c]
                    loss_deriv = error * self.deriv_V(factor_index, feature_id, fi, fj)

                    adaptive_gradient = self.compute_adaptive_gradient_V(factor_index, feature_id, loss_deriv)
                    self.V[factor_index, feature_id] -= self.learning_rate * (adaptive_gradient + 2 *self.l2_reg_V * self.V[factor_index, feature_id])

                    if self.positive_only_V and self.V[factor_index, feature_id] < 0.0:
                        self.V[factor_index, feature_id] = 0.0


            # Exponentiation of beta at the end of each sample
            if self.useAdam:

                self.beta_1_power_t *= self.beta_1
                self.beta_2_power_t *= self.beta_2



            if self.verbose and ((index % 500000 == 0 and index!=0) or index == self.n_samples-1):
                print ("CFW_DVV_Similarity_Cython_SGD: Processed {} out of {} samples ( {:.2f}%). Loss is {:.4E}. Sample per second {:.0f}".format(
                    index, self.n_samples, 100*float(index)/self.n_samples, cum_loss/index, float(index)/(time.time() - start_time)))




        cum_loss /= self.n_samples


        return cum_loss


    def get_D(self):

        return np.array(self.D).copy()

    def get_V(self):

        return np.array(self.V).copy()





    cdef double deriv_V(self, int factor_index, int c, int[:] fi, int[:] fj):
        """
        s_ij_hat partial derivative w.r.t V[k,c]
        dV[k,c] = sum over f { V[k,f] * (fi[c]*fj[f] + fj[c]*fi[f]) }
        """
        cdef int feature_index, feature_id
        cdef double res = 0.0, res_c_in_fi = 0.0, res_c_in_fj = 0.0
        cdef int c_in_fi = False, c_in_fj = False

        for feature_index in range(len(fj)):
            feature_id = fj[feature_index]
            res_c_in_fi += self.V[factor_index, feature_id]

            if feature_id == c:
                c_in_fj = True


        for feature_index in range(len(fi)):
            feature_id = fi[feature_index]
            res_c_in_fj += self.V[factor_index, feature_id]

            if feature_id == c:
                c_in_fi = True


        if c_in_fi:
            res += res_c_in_fi

        if c_in_fj:
            res += res_c_in_fj

        return res



    cdef double compute_adaptive_gradient_D(self, int feature_id, double gradient):

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




    cdef double compute_adaptive_gradient_V(self, int factor_index, int feature_id, double gradient):

        cdef double adaptive_gradient

        if self.useAdaGrad:

            self.sgd_cache_V[factor_index, feature_id] += gradient ** 2
            adaptive_gradient = self.learning_rate / (sqrt(self.sgd_cache_V[factor_index, feature_id]) + 1e-8)

        elif self.useRmsprop:
            self.sgd_cache_V[factor_index, feature_id] = self.sgd_cache_V[factor_index, feature_id] * self.gamma + (1 - self.gamma) * gradient ** 2
            adaptive_gradient = self.learning_rate / (sqrt(self.sgd_cache_V[factor_index, feature_id]) + 1e-8)

        elif self.useAdam:

            self.sgd_cache_V_momentum_1[factor_index, feature_id] = \
                self.sgd_cache_V_momentum_1[factor_index, feature_id] * self.beta_1 + (1 - self.beta_1) * gradient

            self.sgd_cache_V_momentum_2[factor_index, feature_id] = \
                self.sgd_cache_V_momentum_2[factor_index, feature_id] * self.beta_2 + (1 - self.beta_2) * gradient**2


            self.momentum_1 = self.sgd_cache_V_momentum_1[factor_index, feature_id]/ (1 - self.beta_1_power_t)
            self.momentum_2 = self.sgd_cache_V_momentum_2[factor_index, feature_id]/ (1 - self.beta_2_power_t)

            adaptive_gradient = self.momentum_1/ (sqrt(self.momentum_2) + 1e-8)

        else:

            adaptive_gradient = gradient


        return adaptive_gradient











    # Using memoryview instead of the sparse matrix itself allows for much faster access
    cdef int feature_common_precomputed(self, long sample_index):

        self.common_features_id = self.commonFeatures_indices[self.commonFeatures_indptr[sample_index]:self.commonFeatures_indptr[sample_index + 1]]

        return len(self.common_features_id)



    cdef int[:] get_features_vector(self, long index):
        return self.icm_indices[self.icm_indptr[index]:self.icm_indptr[index + 1]]






    cdef int[:] feature_union(self, int[:] feature_i, int[:] feature_j):

        cdef int index_i, index_j, index_result, current_max_value
        cdef int len_f_i = len(feature_i)
        cdef int len_f_j = len(feature_j)
        cdef int terminate = False, add_i_values, add_j_values

        cdef array[int] template = array('i')

        cdef int[:] result = clone(template, len_f_i + len_f_j, zero=True)

        index_i = 0
        index_j = 0
        index_result = 0

        while not terminate:

            add_i_values = True
            # Add values from i, keeping ordering intact

            while add_i_values and index_i < len_f_i:

                # If j contains something else, chech for value
                if index_j < len_f_j:
                    if feature_i[index_i] == feature_j[index_j]:
                        index_j += 1

                    elif feature_i[index_i] > feature_j[index_j]:
                        add_i_values = False

                if add_i_values:

                    if index_result != 0 and current_max_value > feature_i[index_i]:
                        raise ValueError("Array is not sorted! The function return array is undefined")

                    result[index_result] = feature_i[index_i]
                    current_max_value = result[index_result]
                    index_result += 1
                    index_i += 1




            # Either I stopped because there are no more values in i
            # or because the value in i is > of the value in j

            add_j_values = True
            # Add values from j, keeping ordering intact

            while add_j_values and index_j < len_f_j:

                # If i contains something else, chech for value
                if index_i < len_f_i:
                    if feature_j[index_j] == feature_i[index_i]:
                        index_i += 1

                    elif feature_j[index_j] > feature_i[index_i]:
                        add_j_values = False

                if add_j_values:

                    if index_result != 0 and current_max_value > feature_j[index_j]:
                        raise ValueError("Array is not sorted! The function return array is undefined")

                    result[index_result] = feature_j[index_j]
                    current_max_value = result[index_result]
                    index_result += 1
                    index_j += 1



            if index_j == len_f_j and index_i == len_f_i:
                terminate = True


        return result[0:index_result]



    cdef int feature_union_unordered(self, int[:] feature_i, int[:] feature_j):

        cdef int union_features_count = 0
        cdef int feature_index, feature_id

        for feature_index in range(len(feature_i)):

            feature_id = feature_i[feature_index]

            if not self.union_features_flag[feature_id]:
                self.union_features_flag[feature_id] = True
                self.union_features_id[union_features_count] = feature_id
                union_features_count += 1


        for feature_index in range(len(feature_j)):

            feature_id = feature_j[feature_index]

            if not self.union_features_flag[feature_id]:
                self.union_features_flag[feature_id] = True
                self.union_features_id[union_features_count] = feature_id
                union_features_count += 1

        # Clear flag data structure
        for feature_index in range(union_features_count):
            feature_id = self.union_features_id[feature_index]
            self.union_features_flag[feature_id] = False



        return union_features_count




    cdef int feature_common_unordered(self, int[:] feature_i, int[:] feature_j):

        cdef int common_features_count = 0
        cdef int feature_index, feature_id

        #print("New feature_common_unordered")

        for feature_index in range(len(feature_i)):

            feature_id = feature_i[feature_index]
            self.common_features_flag[feature_id] = True

        for feature_index in range(len(feature_j)):

            feature_id = feature_j[feature_index]

            if self.common_features_flag[feature_id]:
                self.common_features_id[common_features_count] = feature_id
                common_features_count += 1

        # Clear flag data structure
        for feature_index in range(len(feature_i)):
            feature_id = feature_i[feature_index]
            self.common_features_flag[feature_id] = False




        return common_features_count




    def feature_union_test(self):

        if self.verbose:
            print("CFW_DVV_Similarity_Cython_SGD: feature_union_test")

        # No duplicates
        f_i = np.array([0, 5, 8, 10, 19], dtype=np.int32)
        f_j = np.array([1, 7, 9, 11, 15, 18], dtype=np.int32)

        union_np = np.union1d(f_i, f_j)
        union_cy = self.feature_union(f_i, f_j)
        union_cy = np.array(union_cy)

        assert np.array_equal(union_np, union_cy), "Union is different. F_i: {}, F_j: {}. Union Numpy: {}, Union Cython: {}".format(
            f_i, f_j, union_np, union_cy)


        count = self.feature_union_unordered(f_i, f_j)
        union_cy = self.union_features_id[0:count]
        union_cy = np.sort(np.array(union_cy))

        assert np.array_equal(union_np, union_cy), "Union unordered is different. F_i: {}, F_j: {}. Union Numpy: {}, Union Cython: {}".format(
            f_i, f_j, union_np, union_cy)


        # Duplicates
        f_i = np.array([0, 5, 8, 10, 19], dtype=np.int32)
        f_j = np.array([1, 10, 11], dtype=np.int32)

        union_np = np.union1d(f_i, f_j)
        union_cy = self.feature_union(f_i, f_j)
        union_cy = np.array(union_cy)

        assert np.array_equal(union_np, union_cy), "Union is different. F_i: {}, F_j: {}. Union Numpy: {}, Union Cython: {}".format(
            f_i, f_j, union_np, union_cy)


        count = self.feature_union_unordered(f_i, f_j)
        union_cy = self.union_features_id[0:count]
        union_cy = np.sort(np.array(union_cy))

        assert np.array_equal(union_np, union_cy), "Union unordered is different. F_i: {}, F_j: {}. Union Numpy: {}, Union Cython: {}".format(
            f_i, f_j, union_np, union_cy)



        f_i = np.array([0, 5, 8, 10, 19], dtype=np.int32)
        f_j = np.array([1, 2, 5, 10, 15], dtype=np.int32)

        union_np = np.union1d(f_i, f_j)
        union_cy = self.feature_union(f_i, f_j)
        union_cy = np.array(union_cy)

        assert np.array_equal(union_np, union_cy), "Union is different. F_i: {}, F_j: {}. Union Numpy: {}, Union Cython: {}".format(
            f_i, f_j, union_np, union_cy)


        count = self.feature_union_unordered(f_i, f_j)
        union_cy = self.union_features_id[0:count]
        union_cy = np.sort(np.array(union_cy))

        assert np.array_equal(union_np, union_cy), "Union unordered is different. F_i: {}, F_j: {}. Union Numpy: {}, Union Cython: {}".format(
            f_i, f_j, union_np, union_cy)



        for _ in range(100):

            size_f_i = np.random.randint(1,10)
            f_i = np.random.randint(0,50, size=size_f_i, dtype=np.int32)
            f_i = np.unique(f_i)

            size_f_j = np.random.randint(1,10)
            f_j = np.random.randint(0,50, size=size_f_j, dtype=np.int32)
            f_j = np.unique(f_j)

            union_np = np.union1d(f_i, f_j)
            union_cy = self.feature_union(f_i, f_j)
            union_cy = np.array(union_cy)

            assert np.array_equal(union_np, union_cy), "Union is different. F_i: {}, F_j: {}. Union Numpy: {}, Union Cython: {}".format(
                f_i, f_j, union_np, union_cy)


            count = self.feature_union_unordered(f_i, f_j)
            union_cy = self.union_features_id[0:count]
            union_cy = np.sort(np.array(union_cy))

            assert np.array_equal(union_np, union_cy), "Union unordered is different. F_i: {}, F_j: {}. Union Numpy: {}, Union Cython: {}".format(
                f_i, f_j, union_np, union_cy)



        if self.verbose:
            print("CFW_DVV_Similarity_Cython_SGD: All tests passed")



    def feature_common_test(self):

        if self.verbose:
            print("CFW_DVV_Similarity_Cython_SGD: feature_common_test")

        # No duplicates
        f_i = np.array([0, 5, 8, 10, 19], dtype=np.int32)
        f_j = np.array([1, 7, 9, 11, 15, 18], dtype=np.int32)

        common_np = np.intersect1d(f_i, f_j)
        count = self.feature_common_unordered(f_i, f_j)
        common_cy = self.common_features_id[0:count]
        common_cy = np.sort(np.array(common_cy))

        assert np.array_equal(common_np, common_cy), "Common unordered is different. F_i: {}, F_j: {}. Union Numpy: {}, Common Cython: {}".format(
            f_i, f_j, common_np, common_cy)


        # Duplicates
        f_i = np.array([0, 5, 8, 10, 19], dtype=np.int32)
        f_j = np.array([1, 10, 11], dtype=np.int32)

        common_np = np.intersect1d(f_i, f_j)
        count = self.feature_common_unordered(f_i, f_j)
        common_cy = self.common_features_id[0:count]
        common_cy = np.sort(np.array(common_cy))

        assert np.array_equal(common_np, common_cy), "Common unordered is different. F_i: {}, F_j: {}. Union Numpy: {}, Common Cython: {}".format(
            f_i, f_j, common_np, common_cy)



        f_i = np.array([0, 5, 8, 10, 19], dtype=np.int32)
        f_j = np.array([1, 2, 5, 10, 15], dtype=np.int32)

        common_np = np.intersect1d(f_i, f_j)
        count = self.feature_common_unordered(f_i, f_j)
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
            count = self.feature_common_unordered(f_i, f_j)
            common_cy = self.common_features_id[0:count]
            common_cy = np.sort(np.array(common_cy))

            assert np.array_equal(common_np, common_cy), "Common unordered is different. F_i: {}, F_j: {}. Common Numpy: {}, Common Cython: {}".format(
                f_i, f_j, common_np, common_cy)



        if self.verbose:
            print("CFW_DVV_Similarity_Cython_SGD: All tests passed")


    def precompute_common_features_function_test(self):

        commonFeatures = self.ICM[np.array(self.row_list)].multiply(self.ICM[np.array(self.col_list)])

        # Init Common features
        commonFeatures_indices_test = commonFeatures.indices
        commonFeatures_indptr_test = commonFeatures.indptr


        self.precompute_common_features_function()


        assert np.array_equal(commonFeatures_indices_test, self.commonFeatures_indices), "commonFeatures_indices differ"
        assert np.array_equal(commonFeatures_indptr_test, self.commonFeatures_indptr), "commonFeatures_indptr differ"