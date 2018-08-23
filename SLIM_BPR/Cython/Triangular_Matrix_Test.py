#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 04/01/18

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
    fileToCompile = 'Triangular_Matrix.pyx'

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
    # python compileCython.py Triangular_Matrix.pyx build_ext --inplace


class MyTestCase(unittest.TestCase):



    def test_initialization(self):

        numCols = 10

        triangularMatrix = Triangular_Matrix(numCols)
        dense = triangularMatrix.get_scipy_csr().todense()

        self.assertTrue(np.equal(dense, np.zeros((numCols, numCols))).all(), "Initialization incorrect")


    def test_get_value_in_empty(self):

        numCols = 10

        triangularMatrix = Triangular_Matrix(numCols)

        value = triangularMatrix.get_value(4, 1)
        self.assertEqual(0.0, value, "get_value_in_empty incorrect")

        value = triangularMatrix.get_value(8, 5)
        self.assertEqual(0.0, value, "get_value_in_empty incorrect")


    def test_add_element_in_empty(self):

        numCols = 10

        increment = 5.0

        triangularMatrix = Triangular_Matrix(numCols)

        value = triangularMatrix.add_value(4, 1, increment)
        self.assertEqual(increment, value, "add_value incorrect")

        value = triangularMatrix.get_value(4, 1)
        self.assertEqual(increment, value, "get_value incorrect")


        dense = np.zeros((numCols, numCols))
        dense[4,1] += increment

        self.assertTrue(np.equal(dense, triangularMatrix.get_scipy_csr().todense()).all(), "Dense matrix incorrect")


    def test_add_element_new_row(self):

        numCols = 10

        increment = 5.0

        triangularMatrix = Triangular_Matrix(numCols)

        triangularMatrix.add_value(4, 1, increment)

        value = triangularMatrix.add_value(3, 3, increment+1)
        self.assertEqual(increment+1, value, "add_value incorrect")

        value = triangularMatrix.get_value(3, 3)
        self.assertEqual(increment+1, value, "get_value incorrect")


        dense = np.zeros((numCols, numCols))
        dense[4, 1] += increment
        dense[3, 3] += increment+1

        self.assertTrue(np.equal(dense, triangularMatrix.get_scipy_csr().todense()).all(), "Dense matrix incorrect")



    def test_add_element_after(self):
        numCols = 10

        increment = 5.0

        triangularMatrix = Triangular_Matrix(numCols)

        triangularMatrix.add_value(2, 1, increment)

        value = triangularMatrix.add_value(4, 1, increment+1)
        self.assertEqual(increment+1, value, "add_value incorrect")

        value = triangularMatrix.get_value(4, 1)
        self.assertEqual(increment+1, value, "get_value incorrect")


        dense = np.zeros((numCols, numCols))
        dense[2, 1] += increment
        dense[4, 1] += increment+1

        self.assertTrue(np.equal(dense, triangularMatrix.get_scipy_csr().todense()).all(), "Dense matrix incorrect")


    def test_symmetry(self):

        numCols = 10

        increment = 5.0

        triangularMatrix = Triangular_Matrix(numCols, isSymmetric = True)

        triangularMatrix.add_value(3, 1, increment)
        self.assertEqual(increment, triangularMatrix.get_value(1, 3), "add_value incorrect")

        triangularMatrix.add_value(8, 2, increment+1)
        self.assertEqual(increment+1, triangularMatrix.get_value(2, 8), "add_value incorrect")

        triangularMatrix.add_value(5, 3, increment+2)
        self.assertEqual(increment+2, triangularMatrix.get_value(3, 5), "add_value incorrect")


        dense = np.zeros((numCols, numCols))
        dense[3, 1] += increment
        dense[1, 3] = dense[3, 1]
        dense[8, 2] += increment+1
        dense[2, 8] = dense[8, 2]
        dense[5, 3] += increment+2
        dense[3, 5] = dense[5, 3]

        self.assertTrue(np.equal(dense, triangularMatrix.get_scipy_csr().todense()).all(), "Dense matrix incorrect")




    def test_add_complex_data(self):
        numCols = 100

        triangularMatrix = Triangular_Matrix(numCols, isSymmetric = True)

        randomData = np.random.random((numCols, numCols))
        randomData = randomData.round(decimals=4)

        incrementalMatrix = np.zeros_like(randomData)

        randomData_coo = sps.coo_matrix(randomData)

        newOrdering = np.arange(len(randomData_coo.data))
        np.random.shuffle(newOrdering)


        for index in range(len(newOrdering)):

            data = randomData_coo.data[newOrdering[index]]
            row = randomData_coo.row[newOrdering[index]]
            col = randomData_coo.col[newOrdering[index]]

            if col<= row:

                data_return = triangularMatrix.add_value(row, col, data)
                self.assertEqual(data, data_return, "add_value incorrect")

                data_return_2 = triangularMatrix.get_value(row, col)
                self.assertEqual(data, data_return_2, "get_value incorrect")

                incrementalMatrix[row, col] += data
                incrementalMatrix[col, row] = incrementalMatrix[row, col]
                #incrementalMatrix = incrementalMatrix.round(decimals=4)

                dense_return = triangularMatrix.get_scipy_csr().todense()
                #dense_return = dense_return.round(decimals=4)

                #self.assertTrue(np.equal(incrementalMatrix, dense_return).all(), "Dense matrix incorrect")
                self.assertTrue(np.allclose(incrementalMatrix, dense_return), "Dense matrix incorrect")

                #print("Added row:{}, col:{}, data:{}".format(row, col, data))



    def test_TopK(self):

        numCols = 4

        TopK = 2

        triangularMatrix = Triangular_Matrix(numCols, isSymmetric = True)
        randomData = np.random.random((numCols,numCols))

        for row in range(randomData.shape[0]):

            for col in range(randomData.shape[1]):
                triangularMatrix.add_value(row, col, randomData[row, col])


        dense_matrix = triangularMatrix.get_scipy_csr().todense()
        dense_matrix_topK = np.zeros_like(dense_matrix)


        for row in range(dense_matrix.shape[0]):
            idx_sorted = np.argsort(dense_matrix[row])  # sort by column
            top_k_idx = np.array(idx_sorted)[0,-TopK:]

            dense_matrix_topK[row, top_k_idx] = dense_matrix[row, top_k_idx]


        dense_matrix_cython_topK = triangularMatrix.get_scipy_csr(TopK = TopK).todense()

        self.assertTrue(np.allclose(dense_matrix_topK, dense_matrix_cython_topK), "Dense matrix incorrect")






from SLIM_BPR.Cython.Triangular_Matrix import Triangular_Matrix

if __name__ == '__main__':

    runCompilationScript()

    from SLIM_BPR.Cython.Triangular_Matrix import Triangular_Matrix

    unittest.main()


