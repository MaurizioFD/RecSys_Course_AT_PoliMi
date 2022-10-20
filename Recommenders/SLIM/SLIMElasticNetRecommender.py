#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Massimo Quadrana, Cesare Bernardis
"""


import numpy as np
import scipy.sparse as sps
from Recommenders.Recommender_utils import check_matrix
from sklearn.linear_model import ElasticNet
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.Similarity.Compute_Similarity_Python import Incremental_Similarity_Builder
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit
import time, sys
from tqdm import tqdm
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# os.environ["PYTHONWARNINGS"] = ('ignore::exceptions.ConvergenceWarning:sklearn.linear_model')
# os.environ["PYTHONWARNINGS"] = ('ignore:Objective did not converge:ConvergenceWarning:')

class SLIMElasticNetRecommender(BaseItemSimilarityMatrixRecommender):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.
    NOTE: ElasticNet solver is parallel, a single intance of SLIM_ElasticNet will
          make use of half the cores available

    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.

        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """

    RECOMMENDER_NAME = "SLIMElasticNetRecommender"

    def __init__(self, URM_train, verbose = True):
        super(SLIMElasticNetRecommender, self).__init__(URM_train, verbose = verbose)

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, l1_ratio=0.1, alpha = 1.0, positive_only=True, topK = 100):

        assert l1_ratio>= 0 and l1_ratio<=1, "{}: l1_ratio must be between 0 and 1, provided value was {}".format(self.RECOMMENDER_NAME, l1_ratio)

        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        self.topK = topK


        # initialize the ElasticNet model
        self.model = ElasticNet(alpha=alpha,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=100,
                                tol=1e-4)

        URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)

        n_items = URM_train.shape[1]

        similarity_builder = Incremental_Similarity_Builder(self.n_items, initial_data_block=self.n_items*self.topK, dtype = np.float32)

        start_time = time.time()
        start_time_printBatch = start_time

        # fit each item's factors sequentially (not in parallel)
        for currentItem in range(n_items):

            # get the target column
            y = URM_train[:, currentItem].toarray()

            # set the j-th column of X to zero
            start_pos = URM_train.indptr[currentItem]
            end_pos = URM_train.indptr[currentItem + 1]

            current_item_data_backup = URM_train.data[start_pos: end_pos].copy()
            URM_train.data[start_pos: end_pos] = 0.0

            # fit one ElasticNet model per column
            self.model.fit(URM_train, y)

            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values
            nonzero_model_coef_index = self.model.sparse_coef_.indices
            nonzero_model_coef_value = self.model.sparse_coef_.data

            # Check if there are more data points than topK, if so, extract the set of K best values
            if len(nonzero_model_coef_value) > self.topK:
                # Partition the data because this operation does not require to fully sort the data
                relevant_items_partition = np.argpartition(-np.abs(nonzero_model_coef_value), self.topK-1, axis=0)[0:self.topK]
                nonzero_model_coef_index = nonzero_model_coef_index[relevant_items_partition]
                nonzero_model_coef_value = nonzero_model_coef_value[relevant_items_partition]

            similarity_builder.add_data_lists(row_list_to_add=nonzero_model_coef_index,
                                              col_list_to_add=np.ones(len(nonzero_model_coef_index), dtype = np.int) * currentItem,
                                              data_list_to_add=nonzero_model_coef_value)


            # finally, replace the original values of the j-th column
            URM_train.data[start_pos:end_pos] = current_item_data_backup

            elapsed_time = time.time() - start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)


            if time.time() - start_time_printBatch > 300 or currentItem == n_items-1:
                self._print("Processed {} ({:4.1f}%) in {:.2f} {}. Items per second: {:.2f}".format(
                    currentItem+1,
                    100.0* float(currentItem+1)/n_items,
                    new_time_value,
                    new_time_unit,
                    float(currentItem)/elapsed_time))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        self.W_sparse = similarity_builder.get_SparseMatrix()



from multiprocessing import Pool, cpu_count, shared_memory
from functools import partial


def create_shared_memory(a):
    shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
    b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
    b[:] = a[:]
    return shm


@ignore_warnings(category=ConvergenceWarning)
def _partial_fit(items, topK, alpha, l1_ratio, urm_shape, positive_only=True, shm_names=None, shm_shapes=None, shm_dtypes=None):

    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        positive=positive_only,
        fit_intercept=False,
        copy_X=False,
        precompute=True,
        selection='random',
        max_iter=100,
        tol=1e-4
    )

    indptr_shm = shared_memory.SharedMemory(name=shm_names[0], create=False)
    indices_shm = shared_memory.SharedMemory(name=shm_names[1], create=False)
    data_shm = shared_memory.SharedMemory(name=shm_names[2], create=False)

    X_j = sps.csc_matrix((
            np.ndarray(shm_shapes[2], dtype=shm_dtypes[2], buffer=data_shm.buf).copy(),
            np.ndarray(shm_shapes[1], dtype=shm_dtypes[1], buffer=indices_shm.buf),
            np.ndarray(shm_shapes[0], dtype=shm_dtypes[0], buffer=indptr_shm.buf),
        ), shape=urm_shape)

    values, rows, cols = [], [], []

    for currentItem in items:

        y = X_j[:, currentItem].toarray()

        backup = X_j.data[X_j.indptr[currentItem]:X_j.indptr[currentItem + 1]]
        X_j.data[X_j.indptr[currentItem]:X_j.indptr[currentItem + 1]] = 0.0

        model.fit(X_j, y)

        nonzero_model_coef_index = model.sparse_coef_.indices
        nonzero_model_coef_value = model.sparse_coef_.data

        # Check if there are more data points than topK, if so, extract the set of K best values
        if len(nonzero_model_coef_value) > topK:
            # Partition the data because this operation does not require to fully sort the data
            relevant_items_partition = np.argpartition(-np.abs(nonzero_model_coef_value), topK-1, axis=0)[0:topK]
            nonzero_model_coef_index = nonzero_model_coef_index[relevant_items_partition]
            nonzero_model_coef_value = nonzero_model_coef_value[relevant_items_partition]

        values.extend(nonzero_model_coef_value)
        rows.extend(nonzero_model_coef_index)
        cols.extend([currentItem] * len(nonzero_model_coef_index))

        X_j.data[X_j.indptr[currentItem]:X_j.indptr[currentItem + 1]] = backup

    indptr_shm.close()
    indices_shm.close()
    data_shm.close()

    return values, rows, cols




class MultiThreadSLIM_SLIMElasticNetRecommender(SLIMElasticNetRecommender):

    def fit(self, alpha=1.0, l1_ratio=0.1, positive_only=True, topK=100,
            verbose=True, workers=int(cpu_count()*0.3)):

        assert l1_ratio>= 0 and l1_ratio<=1, \
            "ElasticNet: l1_ratio must be between 0 and 1, provided value was {}".format(l1_ratio)

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        self.topK = topK

        self.workers = workers

        self.URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)

        indptr_shm = create_shared_memory(self.URM_train.indptr)
        indices_shm = create_shared_memory(self.URM_train.indices)
        data_shm = create_shared_memory(self.URM_train.data)

        _pfit = partial(_partial_fit, topK=self.topK, alpha=self.alpha, urm_shape=self.URM_train.shape,
                        l1_ratio=self.l1_ratio, positive_only=self.positive_only,
                        shm_names=[indptr_shm.name, indices_shm.name, data_shm.name],
                        shm_shapes=[self.URM_train.indptr.shape, self.URM_train.indices.shape, self.URM_train.data.shape],
                        shm_dtypes=[self.URM_train.indptr.dtype, self.URM_train.indices.dtype, self.URM_train.data.dtype])

        with Pool(processes=self.workers) as pool:

            pool_chunksize = 4
            item_chunksize = 8

            itemchunks = np.array_split(np.arange(self.n_items), int(self.n_items / item_chunksize))
            if verbose:
                pbar = tqdm(total=self.n_items)

            # res contains a vector of (values, rows, cols) tuples
            values, rows, cols = [], [], []
            for values_, rows_, cols_ in pool.imap_unordered(_pfit, itemchunks, pool_chunksize):
                values.extend(values_)
                rows.extend(rows_)
                cols.extend(cols_)
                if verbose:
                    pbar.update(item_chunksize)

        indptr_shm.close()
        indices_shm.close()
        data_shm.close()

        indptr_shm.unlink()
        indices_shm.unlink()
        data_shm.unlink()

        # generate the sparse weight matrix
        self.W_sparse = sps.csr_matrix((values, (rows, cols)), shape=(self.n_items, self.n_items), dtype=np.float32)
        self.URM_train = self.URM_train.tocsr()
