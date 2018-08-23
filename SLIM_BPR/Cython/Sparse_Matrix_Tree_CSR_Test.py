#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 27/09/17

@author: Maurizio Ferrari Dacrema
"""

import subprocess
import os
import numpy as np
import scipy.sparse as sps

import unittest



def runCompilationScript():
    # Run compile script setting the working directory to ensure the compiled file are contained in the
    # appropriate subfolder and not the project root

    compiledModuleSubfolder = "/SLIM_BPR/Cython"
    fileToCompile = 'Sparse_Matrix_Tree_CSR.pyx'

    command = ['python',
               'compileCython.py',
               fileToCompile,
               'build_ext',
               '--inplace'
               ]

    output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd())

    try:

        command = ['cython',
                   fileToCompile,
                   '-a'
                   ]

        output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd())

    except:
        pass

    print("Compiled module saved in subfolder: {}".format(compiledModuleSubfolder))

    # Command to run compilation script
    # python compileCython.py Sparse_Matrix_Tree_CSR.pyx build_ext --inplace


class MyTestCase(unittest.TestCase):



    def test_initialization(self):

        numCols = 10
        numRows = 5

        sparseMatrix = Sparse_Matrix_Tree_CSR(numRows, numCols)
        dense = sparseMatrix.get_scipy_csr().todense()

        self.assertTrue(np.equal(dense, np.zeros((numRows, numCols))).all(), "Initialization incorrect")


    def test_get_value_in_empty(self):

        numCols = 10
        numRows = 5

        sparseMatrix = Sparse_Matrix_Tree_CSR(numRows, numCols)

        value = sparseMatrix.get_value(1, 4)

        self.assertEqual(0.0, value, "get_value_in_empty incorrect")


    def test_add_element_in_empty(self):
        numCols = 10
        numRows = 5

        increment = 5.0

        sparseMatrix = Sparse_Matrix_Tree_CSR(numRows, numCols)

        value = sparseMatrix.add_value(1, 4, increment)
        self.assertEqual(increment, value, "add_value incorrect")

        value = sparseMatrix.get_value(1, 4)
        self.assertEqual(increment, value, "get_value incorrect")


        dense = np.zeros((numRows, numCols))
        dense[1,4]+=increment

        self.assertTrue(np.equal(dense, sparseMatrix.get_scipy_csr().todense()).all(), "Dense matrix incorrect")




    def test_add_element_new_row(self):
        numCols = 10
        numRows = 5

        increment = 5.0

        sparseMatrix = Sparse_Matrix_Tree_CSR(numRows, numCols)

        sparseMatrix.add_value(1, 4, increment)

        value = sparseMatrix.add_value(3, 3, increment+1)
        self.assertEqual(increment+1, value, "add_value incorrect")

        value = sparseMatrix.get_value(3, 3)
        self.assertEqual(increment+1, value, "get_value incorrect")


        dense = np.zeros((numRows, numCols))
        dense[1, 4] += increment
        dense[3, 3] += increment+1

        self.assertTrue(np.equal(dense, sparseMatrix.get_scipy_csr().todense()).all(), "Dense matrix incorrect")



    def test_add_element_after(self):
        numCols = 10
        numRows = 5

        increment = 5.0

        sparseMatrix = Sparse_Matrix_Tree_CSR(numRows, numCols)

        sparseMatrix.add_value(1, 2, increment)

        value = sparseMatrix.add_value(1, 4, increment+1)
        self.assertEqual(increment+1, value, "add_value incorrect")

        value = sparseMatrix.get_value(1, 4)
        self.assertEqual(increment+1, value, "get_value incorrect")


        dense = np.zeros((numRows, numCols))
        dense[1, 2] += increment
        dense[1, 4] += increment+1

        self.assertTrue(np.equal(dense, sparseMatrix.get_scipy_csr().todense()).all(), "Dense matrix incorrect")







    def test_rebuild_tree(self):
        numCols = 100
        numRows = 1

        sparseMatrix = Sparse_Matrix_Tree_CSR(numRows, numCols)
        randomData = np.random.random(numCols)

        for index in range(len(randomData)):

            sparseMatrix.add_value(0, index, randomData[index])


        dense_matrix_original = sparseMatrix.get_scipy_csr().todense()

        sparseMatrix.rebalance_tree(TopK = False)
        dense_matrix_rebalanced = sparseMatrix.get_scipy_csr().todense()

        self.assertTrue(np.equal(dense_matrix_original, dense_matrix_rebalanced).all(), "Dense matrix incorrect")


    def test_rebuild_tree_top_k(self):
        numCols = 100
        numRows = 1

        TopK = 20

        sparseMatrix = Sparse_Matrix_Tree_CSR(numRows, numCols)
        randomData = np.random.random(numCols)

        for index in range(len(randomData)):

            sparseMatrix.add_value(0, index, randomData[index])


        dense_matrix_original = sparseMatrix.get_scipy_csr().todense()
        dense_matrix_original_topK = np.zeros_like(dense_matrix_original)


        for row in range(dense_matrix_original.shape[0]):
            idx_sorted = np.argsort(dense_matrix_original[row])  # sort by column
            top_k_idx = np.array(idx_sorted)[0,-TopK:]

            dense_matrix_original_topK[row, top_k_idx] = dense_matrix_original[row, top_k_idx]



        sparseMatrix.rebalance_tree(TopK = TopK)
        dense_matrix_rebalanced_topK = sparseMatrix.get_scipy_csr().todense()

        self.assertTrue(np.equal(dense_matrix_original_topK, dense_matrix_rebalanced_topK).all(), "Dense matrix incorrect")



    def test_add_complex_data(self):
        numCols = 100
        numRows = 10

        sparseMatrix = Sparse_Matrix_Tree_CSR(numRows, numCols)

        randomData = np.random.random((numRows, numCols))
        randomData = randomData.round(decimals=4)

        incrementalMatrix = np.zeros_like(randomData)

        randomData_coo = sps.coo_matrix(randomData)

        newOrdering = np.arange(len(randomData_coo.data))
        np.random.shuffle(newOrdering)


        for index in range(len(newOrdering)):

            data = randomData_coo.data[newOrdering[index]]
            row = randomData_coo.row[newOrdering[index]]
            col = randomData_coo.col[newOrdering[index]]

            data_return = sparseMatrix.add_value(row, col, data)
            self.assertEqual(data, data_return, "add_value incorrect")

            data_return_2 = sparseMatrix.get_value(row, col)
            self.assertEqual(data, data_return_2, "get_value incorrect")

            incrementalMatrix[row, col] = data
            #incrementalMatrix = incrementalMatrix.round(decimals=4)

            dense_return = sparseMatrix.get_scipy_csr().todense()
            #dense_return = dense_return.round(decimals=4)

            #self.assertTrue(np.equal(incrementalMatrix, dense_return).all(), "Dense matrix incorrect")
            self.assertTrue(np.allclose(incrementalMatrix, dense_return), "Dense matrix incorrect")

            #print("Added row:{}, col:{}, data:{}".format(row, col, data))


    def test_speed_access(self):
        numCols = 100
        numRows = numCols

        increment = 5.0

        sparseMatrix_non_rebalanced = Sparse_Matrix_Tree_CSR(numRows, numCols)
        sparseMatrix_rebalanced = Sparse_Matrix_Tree_CSR(numRows, numCols)

        randomData = np.random.random((numRows, numCols))
        incrementalMatrix = np.zeros_like(randomData)

        import time

        start_time = time.time()

        for row_index in range(numRows):
            for column_index in range(numCols):
                incrementalMatrix[row_index,column_index] = randomData[row_index,column_index]
                val = incrementalMatrix[row_index,column_index]

        elapsed_time = time.time()-start_time
        print("Dense matrix access requires:  {:.3} sec, avg per insert is {:.3} sec".format(elapsed_time, elapsed_time/(numRows*numCols)))

        start_time = time.time()

        for row_index in range(numRows):
            for column_index in range(numCols):
                sparseMatrix_non_rebalanced.add_value(row_index, column_index, randomData[row_index, column_index])
                sparseMatrix_rebalanced.add_value(row_index, column_index, randomData[row_index, column_index])

                val = sparseMatrix_non_rebalanced.get_value(row_index, column_index)
                val = sparseMatrix_rebalanced.get_value(row_index, column_index)

        elapsed_time = time.time() - start_time
        elapsed_time = elapsed_time/2
        print("Sparse matrix access requires: {:.3} sec, avg per insert is {:.3} sec".format(elapsed_time, elapsed_time / (numRows * numCols)))

        start_time = time.time()
        sparseMatrix_rebalanced.rebalance_tree(TopK=20)

        print("Rebalance takes {} sec".format(time.time() - start_time))





    def test_flat_list(self):
        numCols = 100
        numRows = 1

        sparseMatrix = Sparse_Matrix_Tree_CSR(numRows, numCols)
        randomData = np.random.random(numCols)

        for index in range(len(randomData)):

            sparseMatrix.add_value(0, index, randomData[index])

        sparseMatrix.test_list_tree_conversion(0)


    def test_top_k(self):
        numCols = 100
        numRows = 1

        sparseMatrix = Sparse_Matrix_Tree_CSR(numRows, numCols)
        randomData = np.random.random(numCols)

        for index in range(len(randomData)):

            sparseMatrix.add_value(0, index, randomData[index])

        sparseMatrix.test_topK_from_list_selection(0, 10)



from SLIM_BPR.Cython.Sparse_Matrix_Tree_CSR import Sparse_Matrix_Tree_CSR


if __name__ == '__main__':

    runCompilationScript()

    from SLIM_BPR.Cython.Sparse_Matrix_Tree_CSR import Sparse_Matrix_Tree_CSR

    unittest.main()


