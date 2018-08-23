#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/09/17

@author: Maurizio Ferrari Dacrema
"""

from Base.Recommender_utils import similarityMatrixTopK

import numpy as np
import scipy.sparse as sps
import unittest


class MyTestCase(unittest.TestCase):

    def test_similarityMatrixTopK_denseToDense(self):

        numRows = 100

        TopK = 20

        dense_input = np.random.random((numRows, numRows))
        dense_output = similarityMatrixTopK(dense_input, k=TopK, forceSparseOutput=False)

        numExpectedNonZeroCells = TopK*numRows

        numNonZeroCells = np.sum(dense_output!=0)

        self.assertEqual(numExpectedNonZeroCells, numNonZeroCells, "DenseToDense incorrect")


    def test_similarityMatrixTopK_denseToSparse(self):

        numRows = 100

        TopK = 20

        dense = np.random.random((numRows, numRows))

        sparse = similarityMatrixTopK(dense, k=TopK, forceSparseOutput=True)
        dense = similarityMatrixTopK(dense, k=TopK, forceSparseOutput=False)


        self.assertTrue(np.equal(dense, sparse.todense()).all(), "denseToSparse incorrect")


    def test_similarityMatrixTopK_sparseToSparse(self):

        numRows = 20

        TopK = 5

        dense_input = np.random.random((numRows, numRows))
        sparse_input = sps.csr_matrix(dense_input)

        dense_output = similarityMatrixTopK(dense_input, k=TopK, forceSparseOutput=False, inplace=False)
        sparse_output = similarityMatrixTopK(sparse_input, k=TopK, forceSparseOutput=True)

        self.assertTrue(np.allclose(dense_output, sparse_output.todense()), "sparseToSparse CSR incorrect")

        sparse_input = sps.csc_matrix(dense_input)
        sparse_output = similarityMatrixTopK(sparse_input, k=TopK, forceSparseOutput=True)
        self.assertTrue(np.allclose(dense_output, sparse_output.todense()), "sparseToSparse CSC incorrect")

if __name__ == '__main__':

    unittest.main()

