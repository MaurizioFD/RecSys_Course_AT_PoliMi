"""
Created on 04/01/18

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

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
"""
Determine the operative system. The interface of numpy returns a different type for argsort under windows and linux

http://docs.cython.org/en/latest/src/userguide/language_basics.html#conditional-compilation
"""
IF UNAME_SYSNAME == "linux":
    DEF LONG_t = "long"
ELIF  UNAME_SYSNAME == "Windows":
    DEF LONG_t = "long long"
ELSE:
    DEF LONG_t = "long long"


import scipy.sparse as sps

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc

from cpython.array cimport array, clone


#################################
#################################       CLASS DECLARATION
#################################

cdef class Triangular_Matrix:

    cdef long num_rows, num_cols
    cdef int isSymmetric

    cdef double** row_pointer





    def __init__(self, long num_rows, int isSymmetric = False):

        cdef int numRow, numCol

        self.num_rows = num_rows
        self.num_cols = num_rows
        self.isSymmetric = isSymmetric

        self.row_pointer = <double **> malloc(self.num_rows * sizeof(double*))



        # Initialize all rows to empty
        for numRow in range(self.num_rows):
            self.row_pointer[numRow] = < double *> malloc((numRow+1) * sizeof(double))

            for numCol in range(numRow+1):
                self.row_pointer[numRow][numCol] = 0.0


    cpdef double add_value(self, long row, long col, double value):
        """
        The function adds a value to the specified cell. A new cell is created if necessary.         
        
        :param row: cell coordinates
        :param col:  cell coordinates
        :param value: value to add
        :return double: resulting cell value
        """

        if row >= self.num_rows or col >= self.num_cols or row < 0 or col < 0:
            raise ValueError("Cell is outside matrix. Matrix shape is ({},{}),"
                             " coordinates given are ({},{})".format(
                             self.num_rows, self.num_cols, row, col))

        elif col > row:

            if self.isSymmetric:
                self.row_pointer[col][row] += value

                return self.row_pointer[col][row]

            else:
                raise ValueError("Cell is in the upper triangular of the matrix,"
                                 " current matrix is lower triangular."
                                 " Coordinates given are ({},{})".format(row, col))
        else:

            self.row_pointer[row][col] += value

            return self.row_pointer[row][col]



    cpdef double get_value(self, long row, long col):
        """
        The function returns the value of the specified cell.         
        
        :param row: cell coordinates
        :param col:  cell coordinates
        :return double: cell value
        """

        if row >= self.num_rows or col >= self.num_cols or row < 0 or col < 0:
            raise ValueError("Cell is outside matrix. Matrix shape is ({},{}), coordinates given are ({},{})".format(
                self.num_rows, self.num_cols, row, col))

        elif col > row:

            if self.isSymmetric:
                return self.row_pointer[col][row]

            else:
                raise ValueError("Cell is in the upper triangular of the matrix,"
                                 " current matrix is lower triangular."
                                 " Coordinates given are ({},{})".format(row, col))
        else:

            return self.row_pointer[row][col]




    cpdef get_scipy_csr(self, long TopK = False):
        """
        The function returns the current sparse matrix as a scipy_csr object         
   
        :return double: scipy_csr object
        """
        cdef int terminate
        cdef long row, col, index

        cdef array[double] template_zero = array('d')
        cdef array[double] currentRowArray = clone(template_zero, self.num_cols, zero=True)

        # Declare numpy data type to use vetor indexing and simplify the topK selection code
        cdef np.ndarray[LONG_t, ndim=1] top_k_partition, top_k_partition_sorting
        cdef np.ndarray[np.float64_t, ndim=1] currentRowArray_np


        data = []
        indices = []
        indptr = []

        # Loop the rows
        for row in range(self.num_rows):

            #Always set indptr
            indptr.append(len(data))

            # Fill RowArray
            for col in range(self.num_cols):

                if col <= row:
                    currentRowArray[col] = self.row_pointer[row][col]
                else:
                    if self.isSymmetric:
                        currentRowArray[col] = self.row_pointer[col][row]
                    else:
                        currentRowArray[col] = 0.0


            if TopK:

                # Sort indices and select TopK
                # Using numpy implies some overhead, unfortunately the plain C qsort function is even slower
                #top_k_idx = np.argsort(this_item_weights) [-self.TopK:]

                # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                # because we avoid sorting elements we already know we don't care about
                # - Partition the data to extract the set of TopK items, this set is unsorted
                # - Sort only the TopK items, discarding the rest
                # - Get the original item index

                currentRowArray_np = - np.array(currentRowArray)
                #
                # Get the unordered set of topK items
                top_k_partition = np.argpartition(currentRowArray_np, TopK-1)[0:TopK]
                # Sort only the elements in the partition
                top_k_partition_sorting = np.argsort(currentRowArray_np[top_k_partition])
                # Get original index
                top_k_idx = top_k_partition[top_k_partition_sorting]

                for index in range(len(top_k_idx)):

                    col = top_k_idx[index]

                    if currentRowArray[col] != 0.0:
                        indices.append(col)
                        data.append(currentRowArray[col])

            else:

                for index in range(self.num_cols):

                    if currentRowArray[index] != 0.0:
                        indices.append(index)
                        data.append(currentRowArray[index])


        #Set terminal indptr
        indptr.append(len(data))

        return sps.csr_matrix((data, indices, indptr), shape=(self.num_rows, self.num_cols))

