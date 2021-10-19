"""
Created on 03/02/2018

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

cimport numpy as np
import numpy as np


from libc.stdlib cimport rand
from libc.math cimport exp, log, sqrt
from cpython.array cimport array, clone


from Recommenders.Recommender_utils import check_matrix
import time


cdef struct BPR_sample:
    long user
    long pos_item
    long neg_item
    long seen_items_start_pos
    long seen_items_end_pos



cdef class FBSM_Rating_Cython_SGD:

    cdef double[:] data_list
    cdef int[:] row_list, col_list

    cdef double[:] D
    cdef double[:,:] V

    cdef double learning_rate, l2_reg_D, l2_reg_V

    cdef int[:] URM_mask_indices, URM_mask_indptr
    cdef int[:] ICM_indices, ICM_indptr
    cdef int[:] user_feature_count_indices, user_feature_count_indptr, user_feature_count_data

    # Data strucures for fast user_feature_count
    cdef int[:] user_feature_count_id, user_feature_count_counter, user_feature_count_flag, user_feature_count_counter_temp
    cdef int user_feature_count_len
    cdef int precompute_user_feature_count


    cdef int n_features, n_factors, n_samples, n_items, n_users

    cdef int useAdaGrad, useRmsprop, useAdam, verbose
    cdef int positive_only_D, positive_only_V

    cdef double [:] sgd_cache_D
    cdef double [:,:] sgd_cache_V
    cdef double gamma


    cdef double [:] sgd_cache_D_momentum_1, sgd_cache_D_momentum_2
    cdef double [:,:] sgd_cache_V_momentum_1, sgd_cache_V_momentum_2
    cdef double beta_1, beta_2, beta_1_power_t, beta_2_power_t
    cdef double momentum_1, momentum_2




    def __init__(self, URM, ICM,
                 n_factors = 1, precompute_user_feature_count = False,
                 learning_rate = 0.05,
                 l2_reg_D = 0.0, l2_reg_V = 0.0,
                 weights_initialization_D = None,
                 weights_initialization_V = None,
                 positive_only_D = True,
                 positive_only_V = True,
                 verbose = False,
                 sgd_mode='adagrad',
                 gamma = 0.995, beta_1=0.9, beta_2=0.999, mean_init = 0.0, std_init = 0.001):


        URM = check_matrix(URM, 'csr').astype(np.int32)
        self.URM_mask_indices = np.array(URM.indices, dtype=np.int32)
        self.URM_mask_indptr = np.array(URM.indptr, dtype=np.int32)

        ICM = check_matrix(ICM, 'csr').astype(np.int32)
        self.ICM_indices = np.array(ICM.indices, dtype=np.int32)
        self.ICM_indptr = np.array(ICM.indptr, dtype=np.int32)



        self.n_users, self.n_items = URM.shape
        self.n_features = ICM.shape[1]
        self.n_factors = n_factors
        self.n_samples = URM.nnz
        self.verbose = verbose

        self.learning_rate = learning_rate
        self.l2_reg_D = l2_reg_D
        self.l2_reg_V = l2_reg_V

        self.positive_only_D = positive_only_D
        self.positive_only_V = positive_only_V

        # Initialization of vector d and matrix V

        if weights_initialization_D is not None:
            assert np.array(weights_initialization_D).ravel().shape == (self.n_features,), \
                "FBSM_Rating_Cython_SGD: Wrong shape for weights_initialization_D, received was {}, expected was {}.".format(
                    np.array(weights_initialization_D).ravel().shape, (self.n_features,))

            self.D = np.array(weights_initialization_D, dtype=np.float64)
        else:
            self.D = np.zeros(self.n_features, dtype=np.float64)

        if weights_initialization_V is not None:
            assert np.array(weights_initialization_V).ravel().shape == (self.n_factors, self.n_features), \
                "FBSM_Rating_Cython_SGD: Wrong shape for weights_initialization_V, received was {}, expected was {}.".format(
                    np.array(weights_initialization_V).ravel().shape, (self.n_factors, self.n_features))

            self.V = np.array(weights_initialization_V, dtype=np.float64)
        else:
            self.V = np.random.normal(mean_init, std_init, (self.n_factors, self.n_features)).astype(np.float64)





        self.precompute_user_feature_count = precompute_user_feature_count

        if self.precompute_user_feature_count:

            if self.verbose:
                print("FBSM_Rating_Cython: Precomputing user_feature_count...")

            # Compute sum of occurrencies of features in items each user interacted with
            # self.User_feature_count = #users, #features
            URM.data = np.ones_like(URM.data, np.int32)
            user_feature_count = (URM * ICM).astype(np.int32)

            user_feature_count = check_matrix(user_feature_count, 'csr')
            self.user_feature_count_indices = np.array(user_feature_count.indices, dtype=np.int32)
            self.user_feature_count_indptr = np.array(user_feature_count.indptr, dtype=np.int32)
            self.user_feature_count_data = np.array(user_feature_count.data, dtype=np.int32)

            if self.verbose:
                print("FBSM_Rating_Cython: Precomputing user_feature_count... Done. "
                      "Nonzero elements are {}, density is {:.4E}.".format(
                    int(user_feature_count.nnz),
                    int(user_feature_count.nnz)/(URM.shape[0]*ICM.shape[1])))

        else:
            self.user_feature_count_counter_temp = np.zeros(self.n_features, dtype=np.int32)
            self.user_feature_count_flag = np.zeros(self.n_features, dtype=np.int32)

        self.user_feature_count_len = 0
        self.user_feature_count_id = np.zeros(self.n_features, dtype=np.int32)
        self.user_feature_count_counter = np.zeros(self.n_features, dtype=np.int32)





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
                "SGD_mode not valid. Acceptable values are: 'sgd', 'adagrad', 'rmsprop'. Provided value was '{}'".format(
                    sgd_mode))





    def fit(self):


        cdef int num_sample, n_block = 10

        cdef double cum_loss=0.0, r_uij_hat, sigma_deriv, loss, loss_deriv, adaptive_gradient, learning_rate_current_feature

        ### FAST data structures
        cdef int[:] feature_i, feature_j
        cdef int feature_index, feature_id, feature_count
        cdef int factor_index

        cdef BPR_sample sample

        cdef int [:] user_feature_count_u_full_vector = np.zeros(self.n_features, dtype=np.int32)
        cdef double [:] D_update = np.zeros(self.n_features, dtype=np.float64)
        cdef double [:,:] V_update = np.zeros((self.n_factors, self.n_features), dtype=np.float64)
        cdef int [:] updated_features_id = np.zeros(self.n_features, dtype=np.int32)
        cdef int [:] updated_features_flag = np.zeros(self.n_features, dtype=np.int32)
        cdef int updated_features_count

        # cdef array[double] template_zero = array('d')
        # cdef array[double] V_f_u, Vdelta_ij, Vfi, Ddelta_ij
        cdef double [:] V_f_u = np.zeros(self.n_factors, dtype=np.float64)
        cdef double [:] Vdelta_ij = np.zeros(self.n_factors, dtype=np.float64)
        cdef double [:] Vfi = np.zeros(self.n_factors, dtype=np.float64)
        cdef double [:] Ddelta_ij = np.zeros(self.n_features, dtype=np.float64)

        start_time = time.time()


        for num_sample in range(self.n_users):

            sample = self.sampleBPR_Cython()

            #Get features of items i and j and items clicked by user u
            # ALL of the following are (1,#features)
            feature_i = self.get_features_vector(sample.pos_item)
            feature_j = self.get_features_vector(sample.neg_item)

            if self.precompute_user_feature_count:
                self.get_user_feature_count(sample.user)
            else:
                self.get_user_feature_count_compute(sample)

            # Add D component
            r_uij_hat = 0.0


            # Ddelta_ij = D (1,#Features) dot delta_ij (#Features,1)
            for feature_index in range(len(feature_i)):
                feature_id = feature_i[feature_index]
                Ddelta_ij[feature_id] += self.D[feature_id]

            for feature_index in range(len(feature_j)):
                feature_id = feature_j[feature_index]
                Ddelta_ij[feature_id] -= self.D[feature_id]


            # Ddelta_ij (#Features,1) dot user_feature_count_u (#Features,1)
            for feature_index in range(self.user_feature_count_len):
                feature_id = self.user_feature_count_id[feature_index]
                feature_count = self.user_feature_count_counter[feature_index]

                r_uij_hat += Ddelta_ij[feature_id] * feature_count


            # r_uij_hat -= feature_i * D * feature_i
            for feature_index in range(len(feature_i)):
                feature_id = feature_i[feature_index]

                r_uij_hat -= self.D[feature_id]


            # Clear Ddelta_ij
            for feature_index in range(len(feature_i)):
                feature_id = feature_i[feature_index]
                Ddelta_ij[feature_id] = 0.0

            for feature_index in range(len(feature_j)):
                feature_id = feature_j[feature_index]
                Ddelta_ij[feature_id] = 0.0


            # ALL of the following are (#factors,1)


            # delta_ij = (fi - fj)
            # Vdelta_ij = V (#factors,#Features) dot delta_ij (#Features,1)
            # Vdelta_ij (#factors,1)
            # Vdelta_ij = clone(template_zero, self.n_factors, zero=True)
            for factor_index in range(self.n_factors):
                Vdelta_ij[factor_index] = 0.0

            for feature_index in range(len(feature_i)):
                feature_id = feature_i[feature_index]

                for factor_index in range(self.n_factors):
                    Vdelta_ij[factor_index] += self.V[factor_index, feature_id]

            for feature_index in range(len(feature_j)):
                feature_id = feature_j[feature_index]

                for factor_index in range(self.n_factors):
                    Vdelta_ij[factor_index] -= self.V[factor_index, feature_id]




            # V_f_u = user_feature_count (1,#Features) dot V.t (#factors,#Features).t
            # V_f_u is (#factors,1)
            for factor_index in range(self.n_factors):
                V_f_u[factor_index] = 0.0

            for feature_index in range(self.user_feature_count_len):
                feature_id = self.user_feature_count_id[feature_index]
                feature_count = self.user_feature_count_counter[feature_index]

                for factor_index in range(self.n_factors):
                    V_f_u[factor_index] += self.V[factor_index, feature_id] * feature_count




            # Vfi = V (#factors,#Features) dot fi (#Features,1)
            # Vfi (#factors,1)
            for factor_index in range(self.n_factors):
                Vfi[factor_index] = 0.0

            for feature_index in range(len(feature_i)):
                feature_id = feature_i[feature_index]

                for factor_index in range(self.n_factors):
                    Vfi[factor_index] += self.V[factor_index, feature_id]



            # Compute np.dot(Vdelta_ij, Vrf_u.T)
            for factor_index in range(self.n_factors):
                r_uij_hat += Vdelta_ij[factor_index] * V_f_u[factor_index]

            # Compute np.dot(Vfi, Vfi.T)
            for factor_index in range(self.n_factors):
                r_uij_hat -= Vfi[factor_index] * Vfi[factor_index]




            #Compute derivatives
            # Use alternate form for sigma e^-x / (1+ e^-x)
            sigma_deriv = 1.0 / (1.0 + exp(r_uij_hat))

            #Compute loss
            # loss = log(1. / (1. + exp(-r_uij_hat)))
            cum_loss += r_uij_hat


            # Perform gradient ascent ONLY on relevant features

            ### UPDATE D

            updated_features_count = 0


            # loss_deriv_D = sigma_deriv * (delta_ij.multiply(user_feature_count_u) - fi.multiply(fi)).toarray().ravel()
            # Problem, delta_ij and user_feature_count_u are two independent data structures and the ordering is not consistent
            # Use a big array to avoid manipulating indices
            for feature_index in range(self.user_feature_count_len):
                feature_id = self.user_feature_count_id[feature_index]
                feature_count = self.user_feature_count_counter[feature_index]

                user_feature_count_u_full_vector[feature_id] = feature_count


            for feature_index in range(len(feature_i)):
                feature_id = feature_i[feature_index]

                D_update[feature_id] += user_feature_count_u_full_vector[feature_id]

                if not updated_features_flag[feature_id]:
                    updated_features_flag[feature_id] = True
                    updated_features_id[updated_features_count] = feature_id
                    updated_features_count += 1


            for feature_index in range(len(feature_j)):
                feature_id = feature_j[feature_index]

                D_update[feature_id] -= user_feature_count_u_full_vector[feature_id]

                if not updated_features_flag[feature_id]:
                    updated_features_flag[feature_id] = True
                    updated_features_id[updated_features_count] = feature_id
                    updated_features_count += 1


            # Clear array full
            for feature_index in range(self.user_feature_count_len):
                feature_id = self.user_feature_count_id[feature_index]

                user_feature_count_u_full_vector[feature_id] = 0


            # loss_deriv_D += sigma_deriv * (- fi.multiply(fi))
            for feature_index in range(len(feature_i)):
                feature_id = feature_i[feature_index]

                D_update[feature_id] -= 1

                # Features i already set to update




            # Apply update and regularization only to modified features
            if updated_features_count > 0:

                for feature_index in range(updated_features_count):
                    feature_id = updated_features_id[feature_index]
                    updated_features_flag[feature_id] = False

                    loss_deriv = sigma_deriv*D_update[feature_id]

                    adaptive_gradient = self.compute_adaptive_gradient_D(feature_id, loss_deriv)

                    self.D[feature_id] += self.learning_rate * (adaptive_gradient - 2 * self.l2_reg_D * self.D[feature_id])


                    D_update[feature_id] = 0.0

                    if self.positive_only_D and self.D[feature_id] < 0.0:
                        self.D[feature_id] = 0.0






            ### UPDATE V

            updated_features_count = 0

            #loss_deriv_V = sigma_deriv * (Vrf_u.T * delta_ij  -  Vdelta_ij.T * user_feature_count_u  - Vfi.T * fi * 2)

            #V_f_u.T (#factors,1).T * delta_ij(#Features,1)         Is an outer product

            for factor_index in range(self.n_factors):

                for feature_index in range(len(feature_i)):
                    feature_id = feature_i[feature_index]

                    V_update[factor_index, feature_id] += V_f_u[factor_index]

                    if not updated_features_flag[feature_id]:
                        updated_features_flag[feature_id] = True
                        updated_features_id[updated_features_count] = feature_id
                        updated_features_count += 1


                for feature_index in range(len(feature_j)):
                    feature_id = feature_j[feature_index]

                    V_update[factor_index, feature_id] -= V_f_u[factor_index]

                    if not updated_features_flag[feature_id]:
                        updated_features_flag[feature_id] = True
                        updated_features_id[updated_features_count] = feature_id
                        updated_features_count += 1



            # + Vdelta_ij.T (#factors,1).T * user_feature_count_u(#Features,1)
            for factor_index in range(self.n_factors):

                for feature_index in range(self.user_feature_count_len):
                    feature_id = self.user_feature_count_id[feature_index]
                    feature_count = self.user_feature_count_counter[feature_index]

                    V_update[factor_index, feature_id] += Vdelta_ij[factor_index] * feature_count

                    if not updated_features_flag[feature_id]:
                        updated_features_flag[feature_id] = True
                        updated_features_id[updated_features_count] = feature_id
                        updated_features_count += 1



            # - Vfi.T * fi * 2
            for factor_index in range(self.n_factors):

                for feature_index in range(len(feature_i)):
                    feature_id = feature_i[feature_index]

                    V_update[factor_index, feature_id] -= 2 * Vfi[factor_index]

                    # Features i already set to update


            # Apply regularization only to modified features and reset data structure
            if updated_features_count > 0:

                for feature_index in range(updated_features_count):
                    feature_id = updated_features_id[feature_index]
                    updated_features_flag[feature_id] = False

                    for factor_index in range(self.n_factors):

                        loss_deriv = sigma_deriv * V_update[factor_index, feature_id]
                        adaptive_gradient = self.compute_adaptive_gradient_V(factor_index, feature_id, loss_deriv)


                        self.V[factor_index, feature_id] += self.learning_rate * (adaptive_gradient - 2 *self.l2_reg_V * self.V[factor_index, feature_id])

                        V_update[factor_index, feature_id] = 0.0

                        if self.positive_only_V and self.V[factor_index, feature_id] < 0.0:
                            self.V[factor_index, feature_id] = 0.0


            if self.useAdam:

                self.beta_1_power_t *= self.beta_1
                self.beta_2_power_t *= self.beta_2



            if self.verbose and ((num_sample % 500000 == 0 and num_sample!=0) or num_sample == self.n_samples-1):
                print ("FBSM_Rating_Cython: Processed {} out of {} samples ( {:.2f}%). BPR loss is {:.4E}. Sample per second {:.0f}".format(
                    num_sample, self.n_samples, 100*float(num_sample)/self.n_samples, np.sqrt(cum_loss/num_sample),
                    float(num_sample)/(time.time() - start_time)))




        return np.sqrt(cum_loss/num_sample)





    def get_D(self):
        return np.array(self.D)

    def get_V(self):
        return  np.array(self.V)



    # Using memoryview instead of the sparse matrix itself allows for much faster access
    cdef int[:] get_features_vector(self, long item_id):
        return self.ICM_indices[self.ICM_indptr[item_id]:self.ICM_indptr[item_id + 1]]



    cdef get_user_feature_count(self, long user_id):

        cdef int start_pos = self.user_feature_count_indptr[user_id]
        cdef int end_pos = self.user_feature_count_indptr[user_id + 1]
        cdef int feature_index, feature_id

        self.user_feature_count_len = 0

        feature_index = start_pos

        while feature_index < end_pos:

            self.user_feature_count_id[self.user_feature_count_len] = self.user_feature_count_indices[feature_index]
            self.user_feature_count_counter[self.user_feature_count_len] = self.user_feature_count_data[feature_index]
            self.user_feature_count_len += 1

            feature_index += 1



    cdef get_user_feature_count_compute(self, BPR_sample sample):

        cdef int item_index, item_id
        cdef int feature_index, feature_id


        # Clean data structure
        for feature_index in range(self.user_feature_count_len):
            feature_id = self.user_feature_count_id[feature_index]

            # Get all feature count with the same ordering as the user_feature_count_id array
            self.user_feature_count_counter_temp[feature_id] = 0
            self.user_feature_count_flag[feature_id] = False


        # Compute the occurence of features for all seen items
        self.user_feature_count_len = 0

        item_index = sample.seen_items_start_pos


        while item_index < sample.seen_items_end_pos:

            item_id = self.URM_mask_indices[item_index]
            feature_index = self.ICM_indptr[item_id]
            item_index += 1

            while feature_index < self.ICM_indptr[item_id+1]:

                feature_id = self.ICM_indices[feature_index]
                feature_index += 1
                self.user_feature_count_counter_temp[feature_id] += 1

                if not self.user_feature_count_flag[feature_id]:

                    self.user_feature_count_flag[feature_id] = True
                    self.user_feature_count_id[self.user_feature_count_len] = feature_id
                    self.user_feature_count_len += 1


        # Collect counters
        for feature_index in range(self.user_feature_count_len):
            feature_id = self.user_feature_count_id[feature_index]

            # Get all feature count with the same ordering as the user_feature_count_id array
            self.user_feature_count_counter[feature_index] = self.user_feature_count_counter_temp[feature_id]






    cdef BPR_sample sampleBPR_Cython(self):

        cdef BPR_sample sample = BPR_sample(-1,-1,-1,-1,-1)

        cdef long index

        cdef int negItemSelected, numSeenItems = 0


        # Skip users with no interactions or with no negative items
        while numSeenItems == 0 or numSeenItems == self.n_items:

            sample.user = rand() % self.n_users

            sample.seen_items_start_pos = self.URM_mask_indptr[sample.user]
            sample.seen_items_end_pos = self.URM_mask_indptr[sample.user + 1]

            numSeenItems = sample.seen_items_end_pos - sample.seen_items_start_pos


        index = rand() % numSeenItems

        sample.pos_item = self.URM_mask_indices[sample.seen_items_start_pos + index]


        negItemSelected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        # for every user
        while (not negItemSelected):

            sample.neg_item = rand() % self.n_items

            index = 0
            while index < numSeenItems and self.URM_mask_indices[sample.seen_items_start_pos + index]!=sample.neg_item:
                index+=1

            if index == numSeenItems:
                negItemSelected = True


        return sample





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


