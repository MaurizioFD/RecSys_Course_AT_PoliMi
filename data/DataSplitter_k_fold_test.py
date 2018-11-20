#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 02/08/18

@author: Maurizio Ferrari Dacrema
"""


import unittest


from data.DataReader import DataReader
import scipy.sparse as sps
import numpy as np


class DummyDatasetReader(object):

    DATASET_SUBFOLDER = "DummyDataset/"
    AVAILABLE_ICM = ["ICM"]


    def __init__(self, apply_k_cores = None, n_items = 1000, n_users = 2000, n_features = 500):

        super(DummyDatasetReader, self).__init__()

        self.n_items = n_items
        self.n_users = n_users
        self.n_features = n_features

        self.load_from_original_file()



    def get_URM_all(self):
        return self.URM_all.copy()


    def load_from_original_file(self):

        print("DummyDataset: Creating data...")

        self.URM_all = np.arange(1, self.n_items+1).repeat(self.n_users).reshape(self.n_users, self.n_items)
        self.ICM = np.arange(1, self.n_features+1).repeat(self.n_items).reshape(self.n_items, self.n_features)

        self.URM_all = sps.csr_matrix(self.URM_all, shape=(self.n_users, self.n_items))
        self.ICM = sps.csr_matrix(self.ICM, shape=(self.n_items, self.n_features))


        print("DummyDataset: Creating data... done!")





class MyTestCase(unittest.TestCase):


    def test_dataSplitter_warm_k_fold(self):


        from data.DataSplitter_k_fold import DataSplitter_Warm_k_fold

        # Clean temp files
        import shutil
        shutil.rmtree("../../RecSysFramework/data/data", ignore_errors=True)


        for n_folds in [2, 5, 10]:

            print("Testing n_folds {}".format(n_folds))

            dataSplitter = DataSplitter_Warm_k_fold(DummyDatasetReader, n_folds = n_folds, ICM_to_load="ICM", forbid_new_split = False)

            dataReader = DummyDatasetReader()

            URM_all = dataReader.get_URM_all()



            for fold_index, (URM_train, URM_test) in dataSplitter:

                # URM train does not contain interactions of URM_test
                URM_sum = URM_train + URM_test

                assert URM_sum.nnz == URM_train.nnz + URM_test.nnz, "DataSplitter_Warm_k_fold: URM train and test contain overlapping interactions"

                assert URM_sum.nnz == URM_all.nnz, "DataSplitter_Warm_k_fold: URM train and test contain a different number of elements than the original matrix"


                URM_delta = URM_sum - URM_all
                URM_delta.eliminate_zeros()

                assert URM_delta.nnz == 0,  "DataSplitter_Warm_k_fold: URM train and test data is different than the original data"


        # Clean temp files
        import shutil
        shutil.rmtree("../../RecSysFramework/data/data", ignore_errors=True)





    def test_dataSplitter_coldItem_k_fold(self):


        from data.DataSplitter_k_fold import DataSplitter_ColdItems_k_fold
        import numpy as np

        # Clean temp files
        import shutil
        shutil.rmtree("../../RecSysFramework/data/data", ignore_errors=True)


        for n_folds in [2, 5, 10]:

            print("Testing n_folds {}".format(n_folds))

            dataSplitter = DataSplitter_ColdItems_k_fold(DummyDatasetReader, n_folds = n_folds, ICM_to_load="ICM", forbid_new_split = False)

            dataReader = DummyDatasetReader()

            URM_all = dataReader.get_URM_all()



            for fold_index, (URM_train, URM_test) in dataSplitter:

                # URM train does not contain interactions of URM_test
                URM_sum = URM_train + URM_test

                assert URM_sum.nnz == URM_train.nnz + URM_test.nnz, "DataSplitter_Warm_k_fold: URM train and test contain overlapping interactions"

                assert URM_sum.nnz == URM_all.nnz, "DataSplitter_Warm_k_fold: URM train and test contain a different number of elements than the original matrix"

                URM_train = sps.csc_matrix(URM_train)
                URM_test = sps.csc_matrix(URM_test)

                train_item_interactions = np.ediff1d(URM_train.indptr)!=0
                test_item_interactions = np.ediff1d(URM_test.indptr)!=0

                assert not np.any(np.logical_and(train_item_interactions, test_item_interactions)), "DataSplitter_Warm_k_fold: URM train and test contain interactions for the same items"

                URM_delta = URM_sum - URM_all
                URM_delta.eliminate_zeros()

                assert URM_delta.nnz == 0,  "DataSplitter_Warm_k_fold: URM train and test data is different than the original data"



        # Clean temp files
        import shutil
        shutil.rmtree("../../RecSysFramework/data/data", ignore_errors=True)











    def test_get_holdout(self):


        from data.DataSplitter_k_fold import DataSplitter_Warm_k_fold

        # Clean temp files
        import shutil
        shutil.rmtree("../../RecSysFramework/data/data", ignore_errors=True)

        for n_folds in [5, 10]:

            dataSplitter = DataSplitter_Warm_k_fold(DummyDatasetReader, n_folds = n_folds, ICM_to_load="ICM", forbid_new_split = False)

            dataReader = DummyDatasetReader()

            URM_all = dataReader.get_URM_all()



            URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

            URM_sum = URM_train + URM_test + URM_validation

            assert URM_sum.nnz == URM_train.nnz + URM_test.nnz + URM_validation.nnz, "DataSplitter_Warm_k_fold: URM train and test contain overlapping interactions"

            assert URM_sum.nnz == URM_all.nnz, "DataSplitter_Warm_k_fold: URM train and test contain a different number of elements than the original matrix"

            URM_delta = URM_sum - URM_all
            URM_delta.eliminate_zeros()

            assert URM_delta.nnz == 0,  "DataSplitter_Warm_k_fold: URM train and test data is different than the original data"



        # Clean temp files
        import shutil
        shutil.rmtree("../../RecSysFramework/data/data", ignore_errors=True)













if __name__ == '__main__':


    unittest.main()
