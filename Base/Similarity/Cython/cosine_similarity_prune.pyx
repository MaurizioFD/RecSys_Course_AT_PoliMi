"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

#cython: boundscheck=False
#cython: wraparound=True
#cython: initializedcheck=False
#cython: language_level=3
#cython: nonecheck=False
#cython: cdivision=True
#cython: unpack_method_calls=True
#cython: overflowcheck=False


import time, sys

import numpy as np
cimport numpy as np

from cpython.array cimport array, clone

from libc.math cimport sqrt



import scipy.sparse as sps
from Base.Recommender_utils import check_matrix

#
# ctypedef struct data_pointer_s:
#     long start_position
#     long num_elements




cdef class Cosine_Similarity:

    cdef int TopK
    cdef long n_columns, n_rows

    cdef double[:] this_item_weights
    cdef int[:] this_item_weights_mask, this_item_weights_id
    cdef int this_item_weights_counter

    cdef int[:] user_to_item_row_ptr, user_to_item_cols
    cdef int[:] item_to_user_rows, item_to_user_col_ptr
    cdef double[:] user_to_item_data, item_to_user_data
    cdef double[:] sumOfSquared
    cdef int shrink, normalize, adjusted_cosine, pearson_correlation, tanimoto_coefficient

    cdef int[:] interactions_per_col
    cdef long[:] new_to_original_id_mapper

    cdef int use_row_weights
    cdef double[:] row_weights

    cdef double[:,:] W_dense

    def __init__(self, dataMatrix, topK = 100, shrink=0, normalize = True,
                 mode = "cosine", row_weights = None):
        """
        Computes the cosine similarity on the columns of dataMatrix
        If it is computed on URM=|users|x|items|, pass the URM as is.
        If it is computed on ICM=|items|x|features|, pass the ICM transposed.
        :param dataMatrix:
        :param topK:
        :param shrink:
        :param normalize:           If True divide the dot product by the product of the norms
        :param row_weights:         Multiply the values in each row by a specified value. Array
        :param mode:    "cosine"    computes Cosine similarity
                        "adjusted"  computes Adjusted Cosine, removing the average of the users
                        "pearson"   computes Pearson Correlation, removing the average of the items
                        "jaccard"   computes Jaccard similarity for binary interactions using Tanimoto
                        "tanimoto"  computes Tanimoto coefficient for binary interactions

        """

        super(Cosine_Similarity, self).__init__()

        self.n_columns = dataMatrix.shape[1]
        self.n_rows = dataMatrix.shape[0]
        self.shrink = shrink
        self.normalize = normalize

        self.adjusted_cosine = False
        self.pearson_correlation = False
        self.tanimoto_coefficient = False

        if mode == "adjusted":
            self.adjusted_cosine = True
        elif mode == "pearson":
            self.pearson_correlation = True
        elif mode == "jaccard" or mode == "tanimoto":
            self.tanimoto_coefficient = True
            # Tanimoto has a specific kind of normalization
            self.normalize = False

        elif mode == "cosine":
            pass
        else:
            raise ValueError("Cosine_Similarity: value for paramether 'mode' not recognized."
                             " Allowed values are: 'cosine', 'pearson', 'adjusted', 'jaccard', 'tanimoto'."
                             " Passed value was '{}'".format(mode))


        self.TopK = min(topK, self.n_columns)
        self.this_item_weights = np.zeros(self.n_columns, dtype=np.float64)
        self.this_item_weights_id = np.zeros(self.n_columns, dtype=np.int32)
        self.this_item_weights_mask = np.zeros(self.n_columns, dtype=np.int32)
        self.this_item_weights_counter = 0

        # Copy data to avoid altering the original object
        dataMatrix = dataMatrix.copy()




        # Order the items with decreasing popularity
        print("Re-Ordering dataMatrix...")
        dataMatrix = check_matrix(dataMatrix, 'csc')
        self.interactions_per_col = np.diff(dataMatrix.indptr)

        self.new_to_original_id_mapper = np.flip(np.argsort(self.interactions_per_col), axis=0)
        self.interactions_per_col =  np.flip(np.sort(self.interactions_per_col), axis=0)

        original_id_to_new = np.zeros(self.n_columns, dtype=np.int32)

        for new_id in range(self.n_columns):
            original_id = self.new_to_original_id_mapper[new_id]
            original_id_to_new[original_id] = new_id


        dataMatrix = check_matrix(dataMatrix, 'coo')
        dataMatrix.col = original_id_to_new[dataMatrix.col]
        dataMatrix = check_matrix(dataMatrix, 'csr')
        print("Re-Ordering dataMatrix... complete")



        if self.adjusted_cosine:
            dataMatrix = self.applyAdjustedCosine(dataMatrix)
        elif self.pearson_correlation:
            dataMatrix = self.applyPearsonCorrelation(dataMatrix)
        elif self.tanimoto_coefficient:
            dataMatrix = self.useOnlyBooleanInteractions(dataMatrix)



        # Compute sum of squared values to be used in normalization
        self.sumOfSquared = np.array(dataMatrix.power(2).sum(axis=0), dtype=np.float64).ravel()

        # Tanimoto does not require the square root to be applied
        if not self.tanimoto_coefficient:
            self.sumOfSquared = np.sqrt(self.sumOfSquared)


        # Apply weight after sumOfSquared has been computed but before the matrix is
        # split in its inner data structures
        self.use_row_weights = False

        if row_weights is not None:

            if dataMatrix.shape[0] != len(row_weights):
                raise ValueError("Cosine_Similarity: provided row_weights and dataMatrix have different number of rows."
                                 "Row_weights has {} rows, dataMatrix has {}.".format(len(row_weights), dataMatrix.shape[0]))


            self.use_row_weights = True
            self.row_weights = np.array(row_weights, dtype=np.float64)





        dataMatrix = check_matrix(dataMatrix, 'csr')

        self.user_to_item_row_ptr = dataMatrix.indptr
        self.user_to_item_cols = dataMatrix.indices
        self.user_to_item_data = np.array(dataMatrix.data, dtype=np.float64)

        dataMatrix = check_matrix(dataMatrix, 'csc')
        self.item_to_user_rows = dataMatrix.indices
        self.item_to_user_col_ptr = dataMatrix.indptr
        self.item_to_user_data = np.array(dataMatrix.data, dtype=np.float64)




        if self.TopK == 0:
            self.W_dense = np.zeros((self.n_columns,self.n_columns))





    cdef useOnlyBooleanInteractions(self, dataMatrix):
        """
        Set to 1 all data points
        :return:
        """

        cdef long index

        for index in range(len(dataMatrix.data)):
            dataMatrix.data[index] = 1

        return dataMatrix



    cdef applyPearsonCorrelation(self, dataMatrix):
        """
        Remove from every data point the average for the corresponding column
        :return:
        """

        cdef double[:] sumPerCol
        cdef int[:] interactionsPerCol
        cdef long colIndex, innerIndex, start_pos, end_pos
        cdef double colAverage


        dataMatrix = check_matrix(dataMatrix, 'csc')


        sumPerCol = np.array(dataMatrix.sum(axis=0), dtype=np.float64).ravel()
        interactionsPerCol = np.diff(dataMatrix.indptr)


        #Remove for every row the corresponding average
        for colIndex in range(self.n_columns):

            if interactionsPerCol[colIndex]>0:

                colAverage = sumPerCol[colIndex] / interactionsPerCol[colIndex]

                start_pos = dataMatrix.indptr[colIndex]
                end_pos = dataMatrix.indptr[colIndex+1]

                innerIndex = start_pos

                while innerIndex < end_pos:

                    dataMatrix.data[innerIndex] -= colAverage
                    innerIndex+=1


        return dataMatrix



    cdef applyAdjustedCosine(self, dataMatrix):
        """
        Remove from every data point the average for the corresponding row
        :return:
        """

        cdef double[:] sumPerRow
        cdef int[:] interactionsPerRow
        cdef long rowIndex, innerIndex, start_pos, end_pos
        cdef double rowAverage

        dataMatrix = check_matrix(dataMatrix, 'csr')

        sumPerRow = np.array(dataMatrix.sum(axis=1), dtype=np.float64).ravel()
        interactionsPerRow = np.diff(dataMatrix.indptr)


        #Remove for every row the corresponding average
        for rowIndex in range(self.n_rows):

            if interactionsPerRow[rowIndex]>0:

                rowAverage = sumPerRow[rowIndex] / interactionsPerRow[rowIndex]

                start_pos = dataMatrix.indptr[rowIndex]
                end_pos = dataMatrix.indptr[rowIndex+1]

                innerIndex = start_pos

                while innerIndex < end_pos:

                    dataMatrix.data[innerIndex] -= rowAverage
                    innerIndex+=1


        return dataMatrix





    cdef int[:] getUsersThatRatedItem(self, long item_id):
        return self.item_to_user_rows[self.item_to_user_col_ptr[item_id]:self.item_to_user_col_ptr[item_id+1]]

    cdef int[:] getItemsRatedByUser(self, long user_id):
        return self.user_to_item_cols[self.user_to_item_row_ptr[user_id]:self.user_to_item_row_ptr[user_id+1]]





    cdef double[:] computeItemSimilarities(self, long item_id_input, long start_item, long end_item):
        """
        For every item the cosine similarity against other items depends on whether they have users in common. The more
        common users the higher the similarity.
        
        The basic implementation is:
        - Select the first item
        - Loop through all other items
        -- Given the two items, get the users they have in common
        -- Update the similarity for all common users
        
        That is VERY slow due to the common user part, in which a long data structure is looped multiple times.
        
        A better way is to use the data structure in a different way skipping the search part, getting directly the
        information we need.
        
        The implementation here used is:
        - Select the first item
        - Initialize a zero valued array for the similarities
        - Get the users who rated the first item
        - Loop through the users
        -- Given a user, get the items he rated (second item)
        -- Update the similarity of the items he rated
        
        
        """

        # Create template used to initialize an array with zeros
        # Much faster than np.zeros(self.n_columns)
        cdef array[double] template_zero = array('d')
        cdef array[double] result = clone(template_zero, end_item-start_item, zero=True)


        cdef long user_index, user_id, item_index, item_id, item_id_second

        cdef int[:] users_that_rated_item = self.getUsersThatRatedItem(item_id_input)
        cdef int[:] items_rated_by_user

        cdef double rating_item_input, rating_item_second, row_weight


        # Get users that rated the items
        for user_index in range(len(users_that_rated_item)):

            user_id = users_that_rated_item[user_index]
            rating_item_input = self.item_to_user_data[self.item_to_user_col_ptr[item_id_input]+user_index]

            if self.use_row_weights:
                row_weight = self.row_weights[user_id]
            else:
                row_weight = 1.0

            # Get all items rated by that user
            items_rated_by_user = self.getItemsRatedByUser(user_id)

            for item_index in range(len(items_rated_by_user)):

                item_id_second = items_rated_by_user[item_index]

                # Stop when final item reached
                if item_id_second >= end_item:
                    break

                # Do not compute the similarity on the diagonal
                # Or for item before start
                if item_id_second != item_id_input and item_id_second >= start_item:
                    # Increment similairty
                    rating_item_second = self.user_to_item_data[self.user_to_item_row_ptr[user_id]+item_index]

                    result[item_id_second-start_item] += rating_item_input*rating_item_second*row_weight


                    # Update global data structure
                    if not self.this_item_weights_mask[item_id_second]:

                        self.this_item_weights_mask[item_id_second] = True
                        self.this_item_weights_id[self.this_item_weights_counter] = item_id_second
                        self.this_item_weights_counter += 1

        return result




    def compute_similarity(self, start_col=None, end_col=None):
        """
        Compute the similarity for the given dataset
        :param self:
        :param start_col: column to begin with
        :param end_col: column to stop before, end_col is excluded
        :return:
        """

        cdef int print_block_size = 500
        cdef int item_pruned = 0

        cdef int itemIndex, innerItemIndex, item_id, local_topK
        cdef long long topKItemIndex

        cdef long long[:] top_k_idx
        cdef double min_dot_product_computed = 0.0

        # Declare numpy data type to use vetor indexing and simplify the topK selection code
        cdef np.ndarray[long, ndim=1] top_k_partition, top_k_partition_sorting
        cdef np.ndarray[np.float64_t, ndim=1] this_item_weights_np
        cdef double[:] this_item_weights
        cdef int[:] this_item_weights_id

        cdef long start_item, end_item, item_block_size
        cdef int prune, processedItems = 0

        # Data structure to incrementally build sparse matrix
        # Preinitialize max possible length
        cdef double[:] values = np.zeros((self.n_columns*self.TopK))
        cdef int[:] rows = np.zeros((self.n_columns*self.TopK,), dtype=np.int32)
        cdef int[:] cols = np.zeros((self.n_columns*self.TopK,), dtype=np.int32)
        cdef long sparse_data_pointer = 0

        cdef int start_col_local = 0, end_col_local = self.n_columns


        if start_col is not None and start_col>0 and start_col<self.n_columns:
            start_col_local = start_col

        if end_col is not None and end_col>start_col_local and end_col<self.n_columns:
            end_col_local = end_col



        start_time = time.time()
        last_print_time = start_time

        itemIndex = start_col_local


        # Compute all similarities for each item
        while itemIndex < end_col_local:

            processedItems += 1
            prune = False

            item_block_size = 100000 + self.TopK

            #print("PROCESSING ITEM {}".format(itemIndex))

            if processedItems % print_block_size==0 or processedItems == end_col_local:

                current_time = time.time()

                # Set block size to the number of items necessary in order to print every 30 seconds
                itemPerSec = processedItems/(time.time()-start_time)

                print_block_size = int(itemPerSec*30)

                if current_time - last_print_time > 30  or processedItems == end_col_local:

                    print("Similarity column {} ( {:2.0f} % ), pruned {} ( {:2.0f} % ), {:.2f} column/sec, elapsed time {:.2f} min".format(
                        processedItems, processedItems*1.0/(end_col_local-start_col_local)*100,
                        item_pruned, item_pruned*1.0/processedItems*100,
                        itemPerSec, (time.time()-start_time) / 60))

                    last_print_time = current_time

                    sys.stdout.flush()
                    sys.stderr.flush()


            # Computed similarities go in self.this_item_weights
            # Compute block by block
            start_item = 0
            this_item_weights_np = np.zeros(item_block_size, dtype=np.float64)
            this_item_weights_id = np.zeros(item_block_size, dtype=np.int32)

            while not prune and start_item<end_col_local:

                end_item = start_item + item_block_size

                if end_item >= end_col_local:
                    end_item = end_col_local
                    prune = True

                # After the first block, add only topK values items
                if start_item == 0:
                    item_block_size -= self.TopK


                #print("Block start_item {} end_item {}".format(start_item, end_item))

                # Compute a block of similarities adding it to self.this_item_weights
                this_item_weights = self.computeItemSimilarities(itemIndex, start_item, end_item)

                # build data structure containing
                # the topK items plus the new block
                for innerItemIndex in range(len(this_item_weights)):
                    item_id = start_item + innerItemIndex


                    if start_item == 0:
                        #print("Adding {} in cell {}".format(this_item_weights[innerItemIndex], innerItemIndex))

                        this_item_weights_np[innerItemIndex] = - this_item_weights[innerItemIndex]
                        this_item_weights_id[innerItemIndex] = item_id
                    else:
                        #print("Adding {} in cell {}".format(this_item_weights[innerItemIndex], innerItemIndex + self.TopK -1))
                        this_item_weights_np[innerItemIndex + self.TopK -1] = - this_item_weights[innerItemIndex]
                        this_item_weights_id[innerItemIndex + self.TopK -1] = item_id


                start_item += item_block_size


                top_k_idx = np.argsort(this_item_weights_np[0:self.TopK + len(this_item_weights)])

                min_dot_product_computed = -this_item_weights_np[top_k_idx[self.TopK-1]]


                if start_item<end_col_local:
                    if self.interactions_per_col[start_item] <= min_dot_product_computed:
                        prune = True
                        item_pruned += 1
                        #print("PRUNED")

                    # else:
                    #     print("self.interactions_per_col[start_item] {}, min_dot_product_computed {}".format(
                    #     self.interactions_per_col[start_item], min_dot_product_computed))

                # Build new data ordering for next step
                for innerItemIndex in range(self.TopK):
                    this_item_weights_id[innerItemIndex] = top_k_idx[innerItemIndex]
                    this_item_weights_np[innerItemIndex] = self.this_item_weights[top_k_idx[innerItemIndex]]





            # Apply normalization and shrinkage, ensure denominator != 0
            if self.normalize:
                for innerItemIndex in range(self.TopK):
                    item_id = this_item_weights_id[innerItemIndex]
                    this_item_weights_np[innerItemIndex] /= self.sumOfSquared[itemIndex] * self.sumOfSquared[item_id]\
                                                         + self.shrink + 1e-6

            # Apply the specific denominator for Tanimoto
            elif self.tanimoto_coefficient:
                for innerItemIndex in range(self.TopK):
                    item_id = this_item_weights_id[innerItemIndex]
                    this_item_weights_np[innerItemIndex] /= self.sumOfSquared[itemIndex] + self.sumOfSquared[item_id] -\
                                                         this_item_weights_np[innerItemIndex] + self.shrink + 1e-6

            elif self.shrink != 0:
                for innerItemIndex in range(self.TopK):
                    this_item_weights_np[innerItemIndex] /= self.shrink



            # Incrementally build sparse matrix, do not add zeros
            for innerItemIndex in range(self.TopK):

                if this_item_weights_np[innerItemIndex] != 0.0:

                    item_id = self.new_to_original_id_mapper[this_item_weights_id[innerItemIndex]]

                    values[sparse_data_pointer] = -this_item_weights_np[innerItemIndex]
                    rows[sparse_data_pointer] = item_id
                    cols[sparse_data_pointer] = itemIndex

                    sparse_data_pointer += 1


            itemIndex += 1

        # End while on columns


        if self.TopK == 0:

            return np.array(self.W_dense)

        else:

            values = np.array(values[0:sparse_data_pointer])
            rows = np.array(rows[0:sparse_data_pointer])
            cols = np.array(cols[0:sparse_data_pointer])

            W_sparse = sps.csr_matrix((values, (rows, cols)),
                                    shape=(self.n_columns, self.n_columns),
                                    dtype=np.float32)

            return W_sparse





def cosine_common(X):
    """
    Function that pairwise cosine similarity of the columns in X.
    It takes only the values in common between each pair of columns
    :param X: instance of scipy.sparse.csc_matrix
    :return:
        the result of co_prodsum
        the number of co_rated elements for every column pair
    """

    X = check_matrix(X, 'csc')

    # use Cython MemoryViews for fast access to the sparse structure of X
    cdef int [:] indices = X.indices
    cdef int [:] indptr = X.indptr
    cdef float [:] data = X.data

    # initialize the result variables
    cdef int n_cols = X.shape[1]
    cdef np.ndarray[np.float32_t, ndim=2] result = np.zeros([n_cols, n_cols], dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=2] common = np.zeros([n_cols, n_cols], dtype=np.int32)

    # let's declare all the variables that we'll use in the loop here
    # NOTE: declaring the type of your variables makes your Cython code run MUCH faster
    # NOTE: Cython allows cdef's only in the main scope
    # cdef's in nested codes will result in compilation errors
    cdef int current_col, second_col, n_i, n_j, ii, jj, n_common
    cdef float ii_sum, jj_sum, ij_sum, x_i, x_j

    for current_col in range(n_cols):
        n_i = indptr[current_col+1] - indptr[current_col]
        # the correlation matrix is symmetric,
        # let's compute only the values for the upper-right triangle
        for second_col in range(current_col+1, n_cols):
            n_j = indptr[second_col+1] - indptr[second_col]

            ij_sum, ii_sum, jj_sum = 0.0, 0.0, 0.0
            ii, jj = 0, 0
            n_common = 0

            # here we exploit the fact that the two subvectors in indices are sorted
            # to compute the dot product of the rows in common between i and j in linear time.
            # (indices[indptr[i]:indptr[i]+n_i] and indices[indptr[j]:indptr[j]+n_j]
            # contain the row indices of the non-zero items in columns i and j)
            while ii < n_i and jj < n_j:
                if indices[indptr[current_col] + ii] < indices[indptr[second_col] + jj]:
                    ii += 1
                elif indices[indptr[current_col] + ii] > indices[indptr[second_col] + jj]:
                    jj += 1
                else:
                    x_i = data[indptr[current_col] + ii]
                    x_j = data[indptr[second_col] + jj]
                    ij_sum += x_i * x_j
                    ii_sum += x_i ** 2
                    jj_sum += x_j ** 2
                    ii += 1
                    jj += 1
                    n_common += 1

            if n_common > 0:
                result[current_col, second_col] = ij_sum / np.sqrt(ii_sum * jj_sum)
                result[second_col, current_col] = result[current_col, second_col]
                common[current_col, second_col] = n_common
                common[second_col, current_col] = n_common

    return result, common



###################################################################################################################
#########################
#########################       ARGSORT
#########################




from libc.stdlib cimport malloc, free#, qsort

# Declaring QSORT as "gil safe", appending "nogil" at the end of the declaration
# Otherwise I will not be able to pass the comparator function pointer
# https://stackoverflow.com/questions/8353076/how-do-i-pass-a-pointer-to-a-c-function-in-cython
cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compar)(const_void *, const_void *)) nogil



# Node struct
ctypedef struct matrix_element_s:
    long coordinate
    double data


cdef int compare_struct_on_data(const void * a_input, const void * b_input):
    """
    The function compares the data contained in the two struct passed.
    If a.data > b.data returns >0  
    If a.data < b.data returns <0      
    
    :return int: +1 or -1
    """

    cdef matrix_element_s * a_casted = <matrix_element_s *> a_input
    cdef matrix_element_s * b_casted = <matrix_element_s *> b_input

    if (a_casted.data - b_casted.data) > 0.0:
        return +1
    else:
        return -1





cdef long[:] argsort(double[:] this_item_weights, int TopK):

    #start_time = time.time()
    cdef array[long] template_zero = array('l')
    cdef array[long] result = clone(template_zero, TopK, zero=False)
    #print("clone {} sec".format(time.time()-start_time))

    cdef matrix_element_s *matrix_element_array
    cdef int index, num_elements

    num_elements = len(this_item_weights)

    # Allocate vector that will be used for sorting
    matrix_element_array = < matrix_element_s *> malloc(num_elements * sizeof(matrix_element_s))

    #start_time = time.time()

    # Fill vector wit pointers to list elements
    for index in range(num_elements):
        matrix_element_array[index].coordinate = index
        matrix_element_array[index].data = this_item_weights[index]

    #print("Init {} sec".format(time.time()-start_time))

    #start_time = time.time()
    # Sort array elements on their data field
    qsort(matrix_element_array, num_elements, sizeof(matrix_element_s), compare_struct_on_data)
    #print("qsort {} sec".format(time.time()-start_time))

    #start_time = time.time()
    # Sort is from lower to higher, therefore the elements to be considered are from len-topK to len
    for index in range(TopK):

        result[index] = matrix_element_array[num_elements - index - 1].coordinate
    #print("result {} sec".format(time.time()-start_time))

    free(matrix_element_array)



    return result

