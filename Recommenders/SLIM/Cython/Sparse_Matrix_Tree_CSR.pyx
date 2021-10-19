"""
Created on 26/09/17

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

import scipy.sparse as sps

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free#, qsort

# Declaring QSORT as "gil safe", appending "nogil" at the end of the declaration
# Otherwise I will not be able to pass the comparator function pointer
# https://stackoverflow.com/questions/8353076/how-do-i-pass-a-pointer-to-a-c-function-in-cython
cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compar)(const_void *, const_void *)) nogil


# Node struct
ctypedef struct matrix_element_tree_s:
    long column
    double data
    matrix_element_tree_s *higher
    matrix_element_tree_s *lower

ctypedef struct head_pointer_tree_s:
    matrix_element_tree_s *head


# Function to allocate a new node
cdef matrix_element_tree_s * pointer_new_matrix_element_tree_s(long column, double data, matrix_element_tree_s *higher,  matrix_element_tree_s *lower):

    cdef matrix_element_tree_s * new_element

    new_element = < matrix_element_tree_s * > malloc(sizeof(matrix_element_tree_s))
    new_element.column = column
    new_element.data = data
    new_element.higher = higher
    new_element.lower = lower

    return new_element


# Functions to compare structs to be used in C qsort
cdef int compare_struct_on_column(const void *a_input, const void *b_input):
    """
    The function compares the column contained in the two struct passed.
    If a.column > b.column returns >0  
    If a.column < b.column returns <0      
    
    :return int: a.column - b.column
    """

    cdef head_pointer_tree_s *a_casted = <head_pointer_tree_s *> a_input
    cdef head_pointer_tree_s *b_casted = <head_pointer_tree_s *> b_input

    return a_casted.head.column  - b_casted.head.column



cdef int compare_struct_on_data(const void * a_input, const void * b_input):
    """
    The function compares the data contained in the two struct passed.
    If a.data > b.data returns >0  
    If a.data < b.data returns <0      
    
    :return int: +1 or -1
    """

    cdef head_pointer_tree_s * a_casted = <head_pointer_tree_s *> a_input
    cdef head_pointer_tree_s * b_casted = <head_pointer_tree_s *> b_input

    if (a_casted.head.data - b_casted.head.data) > 0.0:
        return +1
    else:
        return -1



#################################
#################################       CLASS DECLARATION
#################################

cdef class Sparse_Matrix_Tree_CSR:

    cdef long num_rows, num_cols

    # Array containing the struct (object, not pointer) corresponding to the root of the tree
    cdef head_pointer_tree_s* row_pointer

    def __init__(self, long num_rows, long num_cols):

        self.num_rows = num_rows
        self.num_cols = num_cols

        self.row_pointer = < head_pointer_tree_s *> malloc(self.num_rows * sizeof(head_pointer_tree_s))

        # Initialize all rows to empty
        for index in range(self.num_rows):
            self.row_pointer[index].head = NULL


    cpdef double add_value(self, long row, long col, double value):
        """
        The function adds a value to the specified cell. A new cell is created if necessary.         
        
        :param row: cell coordinates
        :param col:  cell coordinates
        :param value: value to add
        :return double: resulting cell value
        """

        if row >= self.num_rows or col >= self.num_cols or row < 0 or col < 0:
            raise ValueError("Cell is outside matrix. Matrix shape is ({},{}), coordinates given are ({},{})".format(
                self.num_rows, self.num_cols, row, col))

        cdef matrix_element_tree_s* current_element, new_element, * old_element
        cdef int stopSearch = False


        # If the row is empty, create a new element
        if self.row_pointer[row].head == NULL:

            # row_pointer is a python object, so I need the object itself and not the address
            self.row_pointer[row].head = pointer_new_matrix_element_tree_s(col, value, NULL, NULL)

            return value


        # If the row is not empty, look for the cell
        # row_pointer contains the struct itself, but I just want its address
        current_element = self.row_pointer[row].head

        # Follow the tree structure
        while not stopSearch:

            if current_element.column < col and current_element.higher != NULL:
                current_element = current_element.higher

            elif current_element.column > col and current_element.lower != NULL:
                current_element = current_element.lower

            else:
                stopSearch = True

        # If the cell exist, update its value
        if current_element.column == col:
            current_element.data += value

            return current_element.data


        # The cell is not found, create new Higher element
        elif current_element.column < col and current_element.higher == NULL:

            current_element.higher = pointer_new_matrix_element_tree_s(col, value, NULL, NULL)

            return value

        # The cell is not found, create new Lower element
        elif current_element.column > col and current_element.lower == NULL:

            current_element.lower = pointer_new_matrix_element_tree_s(col, value, NULL, NULL)

            return value

        else:
            assert False, 'ERROR - Current insert operation is not implemented'




    cpdef double get_value(self, long row, long col):
        """
        The function returns the value of the specified cell.         
        
        :param row: cell coordinates
        :param col:  cell coordinates
        :return double: cell value
        """


        if row >= self.num_rows or col >= self.num_cols or row < 0 or col < 0:
            raise ValueError(
                "Cell is outside matrix. Matrix shape is ({},{}), coordinates given are ({},{})".format(
                    self.num_rows, self.num_cols, row, col))


        cdef matrix_element_tree_s* current_element
        cdef int stopSearch = False

        # If the row is empty, return default
        if self.row_pointer[row].head == NULL:
            return 0.0


        # If the row is not empty, look for the cell
        # row_pointer contains the struct itself, but I just want its address
        current_element = self.row_pointer[row].head

        # Follow the tree structure
        while not stopSearch:

            if current_element.column < col and current_element.higher != NULL:
                current_element = current_element.higher

            elif current_element.column > col and current_element.lower != NULL:
                current_element = current_element.lower

            else:
                stopSearch = True


        # If the cell exist, return its value
        if current_element.column == col:
            return current_element.data

        # The cell is not found, return default
        else:
            return 0.0




    cpdef get_scipy_csr(self, long TopK = False):
        """
        The function returns the current sparse matrix as a scipy_csr object         
   
        :return double: scipy_csr object
        """
        cdef int terminate
        cdef long row

        data = []
        indices = []
        indptr = []

        # Loop the rows
        for row in range(self.num_rows):

            #Always set indptr
            indptr.append(len(data))

            # row contains data
            if self.row_pointer[row].head != NULL:

                # Flatten the data structure
                self.row_pointer[row].head = self.subtree_to_list_flat(self.row_pointer[row].head)
                #print("subtree_to_list_flat {} sec".format(time.time() - start_time))

                if TopK:
                    self.row_pointer[row].head = self.topK_selection_from_list(self.row_pointer[row].head, TopK)
                    #print("topK_selection_from_list {} sec".format(time.time() - start_time))


                # Flatten the tree data
                subtree_column, subtree_data = self.from_linked_list_to_python_list(self.row_pointer[row].head)
                data.extend(subtree_data)
                indices.extend(subtree_column)

                # Rebuild the tree
                self.row_pointer[row].head = self.build_tree_from_list_flat(self.row_pointer[row].head)
                #print("build_tree_from_list_flat {} sec".format(time.time() - start_time))


        #Set terminal indptr
        indptr.append(len(data))

        return sps.csr_matrix((data, indices, indptr), shape=(self.num_rows, self.num_cols))



    cpdef rebalance_tree(self, long TopK = False):
        """
        The function builds a balanced binary tree from the current one, for all matrix rows
        
        :param TopK: either False or an integer number. Number of the highest elements to preserve
        """

        cdef long row

        #start_time = time.time()

        for row in range(self.num_rows):

            if self.row_pointer[row].head != NULL:

                # Flatten the data structure
                self.row_pointer[row].head = self.subtree_to_list_flat(self.row_pointer[row].head)
                #print("subtree_to_list_flat {} sec".format(time.time() - start_time))

                if TopK:
                    self.row_pointer[row].head = self.topK_selection_from_list(self.row_pointer[row].head, TopK)
                    #print("topK_selection_from_list {} sec".format(time.time() - start_time))

                # Rebuild the tree
                self.row_pointer[row].head = self.build_tree_from_list_flat(self.row_pointer[row].head)
                #print("build_tree_from_list_flat {} sec".format(time.time() - start_time))
















    cdef matrix_element_tree_s * subtree_to_list_flat(self, matrix_element_tree_s * root):
        """
        The function flatten the structure of the subtree whose root is passed as a parameter    
        The list is bidirectional and ordered with respect to the column
        The column ordering follows from the insertion policy
        
        :param root: tree root
        :return list, list: data and corresponding column. Empty list if root is None
        """

        if root == NULL:
            return NULL

        cdef matrix_element_tree_s *flat_list_head, *current_element

        # Flatten lower subtree
        flat_list_head = self.subtree_to_list_flat(root.lower)

        # If no lower elements exist, the head is the current element
        if flat_list_head == NULL:
            flat_list_head = root
            root.lower = NULL

        # Else move to the tail and add the subtree root
        else:
            current_element = flat_list_head
            while current_element.higher != NULL:
                current_element = current_element.higher

            # Attach the element with the bidirectional pointers
            current_element.higher = root
            root.lower = current_element

        # Flatten higher subtree and attach it to the tail of the flat list
        root.higher = self.subtree_to_list_flat(root.higher)

        # Attach the element with the bidirectional pointers
        if root.higher != NULL:
            root.higher.lower = root

        return flat_list_head



    cdef from_linked_list_to_python_list(self, matrix_element_tree_s * head):

        data = []
        column = []

        while head != NULL:

            if head.data != 0.0:
                data.append(head.data)
                column.append(head.column)

            head = head.higher

        return column, data



    cdef subtree_free_memory(self, matrix_element_tree_s* root):
        """
        The function frees all struct in the subtree whose root is passed as a parameter, root included 
        
        :param root: tree root
        """

        if root != NULL:
            # If the root exists, open recursion
            self.subtree_free_memory(root.higher)
            self.subtree_free_memory(root.lower)

            # Once the lower elements have been reached, start freeing from the bottom
            free(root)



    cdef list_free_memory(self, matrix_element_tree_s * head):
        """
        The function frees all struct in the list whose head is passed as a parameter, head included 
        
        :param head: list head
        """

        if head != NULL:
            # If the root exists, open recursion
            self.subtree_free_memory(head.higher)

            # Once the tail element have been reached, start freeing from them
            free(head)



    cdef matrix_element_tree_s* build_tree_from_list_flat(self, matrix_element_tree_s* flat_list_head):
        """
        The function builds a tree containing the passed data. This is the recursive function, the 
        data should be sorted by te caller
        To ensure the tree is balanced, data is sorted according to the column   
        
        :param row: row in which to create new tree
        :param column_vector: column coordinates 
        :param data_vector: cell data
        """

        if flat_list_head == NULL:
            return NULL


        cdef long list_length = 0
        cdef long middle_element_step = 0

        cdef matrix_element_tree_s *current_element, *middleElement, *tree_root

        current_element = flat_list_head
        middleElement = flat_list_head

        # Explore the flat list moving the middle elment every tho jumps
        while current_element != NULL:
            current_element = current_element.higher
            list_length += 1
            middle_element_step += 1

            if middle_element_step == 2:
                middleElement = middleElement.higher
                middle_element_step = 0

        tree_root = middleElement

        # To execute the recursion it is necessary to cut the flat list
        # The last of the lower elements will have to be a tail
        if middleElement.lower != NULL:
            middleElement.lower.higher = NULL

            tree_root.lower = self.build_tree_from_list_flat(flat_list_head)


        # The first of the higher elements will have to be a head
        if middleElement.higher != NULL:
            middleElement.higher.lower = NULL

            tree_root.higher = self.build_tree_from_list_flat(middleElement.higher)


        return tree_root




    cdef matrix_element_tree_s* topK_selection_from_list(self, matrix_element_tree_s* head, long TopK):
        """
        The function selects the topK highest elements in the given list 
        
        :param head: head of the list
        :param TopK: number of highest elements to preserve
        :return matrix_element_tree_s*: head of the new list
        """

        cdef head_pointer_tree_s *vector_pointer_to_list_elements
        cdef matrix_element_tree_s *current_element
        cdef long list_length, index, selected_count

        # Get list size
        current_element = head
        list_length = 0

        while current_element != NULL:
            list_length += 1
            current_element = current_element.higher


        # If list elements are not enough to perform a selection, return
        if list_length < TopK:
            return head

        # Allocate vector that will be used for sorting
        vector_pointer_to_list_elements = < head_pointer_tree_s *> malloc(list_length * sizeof(head_pointer_tree_s))

        # Fill vector wit pointers to list elements
        current_element = head
        for index in range(list_length):
            vector_pointer_to_list_elements[index].head = current_element
            current_element = current_element.higher


        # Sort array elements on their data field
        qsort(vector_pointer_to_list_elements, list_length, sizeof(head_pointer_tree_s), compare_struct_on_data)

        # Sort only the TopK according to their column field
        # Sort is from lower to higher, therefore the elements to be considered are from len-topK to len
        qsort(&vector_pointer_to_list_elements[list_length-TopK], TopK, sizeof(head_pointer_tree_s), compare_struct_on_column)


        # Rebuild list attaching the consecutive elements
        index = list_length-TopK

        # Detach last TopK element from previous ones
        vector_pointer_to_list_elements[index].head.lower = NULL

        while index<list_length-1:
            # Rearrange bidirectional pointers
            vector_pointer_to_list_elements[index+1].head.lower = vector_pointer_to_list_elements[index].head
            vector_pointer_to_list_elements[index].head.higher = vector_pointer_to_list_elements[index+1].head

            index += 1

        # Last element in vector will be the hew head
        vector_pointer_to_list_elements[list_length - 1].head.higher = NULL

        # Get hew list head
        current_element = vector_pointer_to_list_elements[list_length-TopK].head

        # If there are exactly enough elements to reach TopK, index == 0 will be the tail
        # Else, index will be the tail and the other elements will be removed
        index = list_length - TopK - 1
        if index > 0:

            index -= 1
            while index >= 0:
                free(vector_pointer_to_list_elements[index].head)
                index -= 1

        # Free array
        free(vector_pointer_to_list_elements)


        return current_element



				


##################################################################################################################
#####################
#####################            TEST FUNCTIONS
#####################
##################################################################################################################


    cpdef test_list_tree_conversion(self, long row):
        """
        The function tests the inner data structure conversion from tree to C linked list and back to tree
        
        :param row: row to use for testing
        """

        cdef matrix_element_tree_s *head, *tree_root
        cdef matrix_element_tree_s *current_element, *previous_element

        head = self.subtree_to_list_flat(self.row_pointer[row].head)
        current_element = head

        cdef numElements_higher = 0
        cdef numElements_lower = 0

        while current_element != NULL:
            numElements_higher += 1
            previous_element = current_element
            current_element = current_element.higher

        current_element = previous_element
        while current_element != NULL:
            numElements_lower += 1
            current_element = current_element.lower

        assert numElements_higher == numElements_lower, 'Bidirectional linked list not consistent.' \
                                                        ' From head to tail element count is {}, from tail to head is {}'.format(
                                                        numElements_higher, numElements_lower)

        print("Bidirectional list link - Passed")

        column_original, data_original = self.from_linked_list_to_python_list(head)

        assert numElements_higher == len(column_original), \
            'Data structure size inconsistent. LinkedList is {}, Python list is {}'.format(numElements_higher, len(column_original))

        for index in range(len(column_original)-1):
            assert column_original[index] < column_original[index+1],\
                'Columns not ordered correctly. Tree not flattened properly'

        print("Bidirectional list ordering - Passed")

        # Transform list into tree and back into list, as it is easy to test
        tree_root = self.build_tree_from_list_flat(head)
        head = self.subtree_to_list_flat(tree_root)

        cdef numElements_higher_after = 0
        cdef numElements_lower_after = 0

        current_element = head

        while current_element != NULL:
            numElements_higher_after += 1
            previous_element = current_element
            current_element = current_element.higher

        current_element = previous_element
        while current_element != NULL:
            numElements_lower_after += 1
            current_element = current_element.lower

        print("Bidirectional list from tree link - Passed")

        assert numElements_higher_after == numElements_lower_after, \
            'Bidirectional linked list after tree construction not consistent. ' \
            'From head to tail element count is {}, from tail to head is {}'.format(
            numElements_higher_after, numElements_lower_after)

        assert numElements_higher == numElements_higher_after, \
            'Data structure size inconsistent. Original length is {}, after tree conversion is {}'.format(
                numElements_higher, numElements_higher_after)

        column_after_tree, data_after_tree = self.from_linked_list_to_python_list(head)

        assert len(column_original) == len(column_after_tree), \
            'Data structure size inconsistent. Original length is {}, after tree conversion is {}'.format(
                len(column_original), len(column_after_tree))

        for index in range(len(column_original)):
            assert column_original[index] == column_after_tree[index],\
                'After tree construction columns are not ordered properly'
            assert data_original[index] == data_after_tree[index],\
                'After tree construction data content is changed'

        print("Bidirectional list from tree ordering - Passed")



    cpdef test_topK_from_list_selection(self, long row, long topK):
        """
        The function tests the topK selection from list
        
        :param row: row to use for testing
        """

        cdef matrix_element_tree_s *head

        head = self.subtree_to_list_flat(self.row_pointer[row].head)

        column_original, data_original = self.from_linked_list_to_python_list(head)

        head = self.topK_selection_from_list(head, topK)

        column_topK, data_topK = self.from_linked_list_to_python_list(head)

        assert len(column_topK) == len(data_topK),\
            "TopK data and column lists have different length. Columns length is {}, data is {}".format(len(column_topK), len(data_topK))
        assert len(column_topK) <= topK,\
            "TopK extracted list is longer than desired value. Desired is {}, while list is {}".format(topK, len(column_topK))

        print("TopK extracted length - Passed")

        # Sort with respect to the content to select topK
        idx_sorted = np.argsort(data_original)
        idx_sorted = np.flip(idx_sorted, axis=0)
        top_k_idx = idx_sorted[0:topK]

        column_topK_numpy = np.array(column_original)[top_k_idx]
        data_topK_numpy = np.array(data_original)[top_k_idx]

        # Sort with respect to the column to ensure it is ordered as the tree flattened list
        idx_sorted = np.argsort(column_topK_numpy)
        column_topK_numpy = column_topK_numpy[idx_sorted]
        data_topK_numpy = data_topK_numpy[idx_sorted]


        assert len(column_topK_numpy) <= len(column_topK),\
            "TopK extracted list and numpy one have different length. Extracted list lenght is {}, while numpy is {}".format(
                len(column_topK_numpy), len(column_topK))


        for index in range(len(column_topK)):

            assert column_topK[index] == column_topK_numpy[index], \
                "TopK extracted list and numpy one have different content at index {} as column value." \
                " Extracted list lenght is {}, while numpy is {}".format(index, column_topK[index], column_topK_numpy[index])

            assert data_topK[index] == data_topK_numpy[index], \
                "TopK extracted list and numpy one have different content at index {} as data value." \
                " Extracted list lenght is {}, while numpy is {}".format(index, data_topK[index], data_topK_numpy[index])

        print("TopK extracted content - Passed")







##################################################################################################################
#####################
#####################            OLD OR PYTHON-LIKE FUNCTIONS
#####################
##################################################################################################################
    #
    #
    #
    #
    # cdef subtree_to_list(self, matrix_element_tree_s* root):
    #     """
    #     The function returns the data contained in the subtree whose root is passed as a parameter
    #
    #     :param root: tree root
    #     :return list, list: data and corresponding column. Empty list if root is None
    #     """
    #
    #     data = []
    #     column = []
    #
    #     if root != NULL:
    #
    #         # Get the data from the root
    #         data = [root.data]
    #         column = [root.column]
    #
    #         # Explore higher column values
    #         subtree_data, subtree_column = self.subtree_to_list(root.higher)
    #
    #         # Merge the data
    #         data.extend(subtree_data)
    #         column.extend(subtree_column)
    #
    #         # Explore lower column values
    #         subtree_data, subtree_column = self.subtree_to_list(root.lower)
    #
    #         # Merge the data
    #         data.extend(subtree_data)
    #         column.extend(subtree_column)
    #
    #     return data, column
    #
    #
    #
		#
    # cdef matrix_element_tree_s* build_tree_from_vector_with_sorting(self, long[:] column_vector, double[:] data_vector):
    #     """
    #     The function builds a balanced binary tree containing the passed data.
    #     To ensure the tree is balanced, data is sorted according to the column
    #
    #     :param row: row in which to create new tree
    #     :param column_vector: column coordinates
    #     :param data_vector: cell data
    #     """
    #
    #     # Sort data column-wise to ensure tree is balanced
    #     idx_sorted = np.argsort(column_vector)
    #     column_vector = np.array(column_vector)[idx_sorted]
    #     data_vector = np.array(data_vector)[idx_sorted]
    #
    #     # cdef long * column_vector_pointer = < long *> malloc(len(data_vector) * sizeof(long))
    #     # cdef double *data_vector_pointer = < double *> malloc(len(data_vector) * sizeof(double))
    #     #
    #     # cdef long index
    #     #
    #     # for index in range(len(data_vector)):
    #     #     column_vector_pointer[index] = column_vector[index]
    #     #     data_vector_pointer[index] = data_vector[index]
    #
    #     # Call recursive function
    #     #self.build_tree_from_vector_pointer(row, 0, len(data_vector)-1,column_vector_pointer, data_vector_pointer)
    #     #self.build_tree_from_vector(row,column_vector, data_vector)
    #
    #     return self.build_tree_from_vector_subtree(column_vector, data_vector)
    #
    #
    #
    #
    #
    #
    #
    # cdef build_tree_from_vector(self, long row, long[:] column_vector, double[:] data_vector):
    #     """
    #     The function builds a tree containing the passed data. This is the recursive function, the
    #     data should be sorted by te caller
    #     To ensure the tree is balanced, data is sorted according to the column
    #
    #     :param row: row in which to create new tree
    #     :param column_vector: column coordinates
    #     :param data_vector: cell data
    #     """
    #
    #     cdef long middleElement
    #     cdef long length_data_vector = len(data_vector)
    #     cdef long[:] idx_sorted
    #
    #     # Base case, no data
    #     if length_data_vector == 0:
    #         return
    #
    #     # Base case, last cell
    #     if length_data_vector == 1:
    #         self.add_value(row, column_vector[0], data_vector[0])
    #         return
    #
    #
    #     middleElement = int(length_data_vector / 2)
    #
    #     # Add only middle element
    #     self.add_value(row, column_vector[middleElement], data_vector[middleElement])
    #
    #     # begin recursion for higher and lower subtrees
    #     self.build_tree_from_vector(row, column_vector[0:middleElement], data_vector[0:middleElement])
    #     self.build_tree_from_vector(row, column_vector[middleElement+1:length_data_vector], data_vector[middleElement+1:length_data_vector])
    #
    #
		#
		#
    #
    # cdef matrix_element_tree_s* build_tree_from_vector_subtree(self, long[:] column_vector, double[:] data_vector):
    #     """
    #     The function builds a tree containing the passed data. This is the recursive function, the
    #     data should be sorted by te caller
    #     To ensure the tree is balanced, data is sorted according to the column
    #
    #     :param row: row in which to create new tree
    #     :param column_vector: column coordinates
    #     :param data_vector: cell data
    #     """
    #
    #     cdef long middleElement
    #     cdef long length_data_vector = len(data_vector)
    #     cdef long[:] idx_sorted
    #     cdef matrix_element_tree_s* subtree_root
    #
    #
    #
    #     # Base case, no data
    #     if length_data_vector == 0:
    #         return NULL
    #
    #     # Base case, last cell
    #     if length_data_vector == 1:
    #         return pointer_new_matrix_element_tree_s(column_vector[0], data_vector[0], NULL, NULL)
    #
    #     middleElement = int(length_data_vector / 2)
    #
    #     # Add only middle element
    #     subtree_root = pointer_new_matrix_element_tree_s(column_vector[middleElement],  data_vector[middleElement], NULL, NULL)
    #
    #     # begin recursion for higher and lower subtrees
    #     subtree_root.lower = self.build_tree_from_vector_subtree(column_vector[0:middleElement], data_vector[0:middleElement])
    #     subtree_root.higher = self.build_tree_from_vector_subtree(column_vector[middleElement + 1:length_data_vector],
    #                                 data_vector[middleElement + 1:length_data_vector])
    #
    #     #print("Created Data {}, col {}".format(np.array(data_vector),np.array(column_vector) ))
    #
    #     return subtree_root
    #
    #
    #
    # cdef build_tree_from_vector_pointer(self, long row, long start_pos, long end_pos, long *column_vector, double *data_vector):
    #     """
    #     The function builds a tree containing the passed data. This is the recursive function, the
    #     data should be sorted by te caller
    #     To ensure the tree is balanced, data is sorted according to the column
    #
    #     :param row: row in which to create new tree
    #     :param column_vector: column coordinates
    #     :param data_vector: cell data
    #     """
    #
    #
    #     cdef long middleElement
    #     cdef long length_data_vector = end_pos - start_pos + 1
    #     cdef long[:] idx_sorted
    #
    #     #print("start_pos: {}, end_pos: {}".format(start_pos,end_pos))
    #
    #     # Base case, no data
    #     if length_data_vector <= 0:
    #         return
    #
    #     # Base case, last cell
    #     if length_data_vector == 1:
    #         self.add_value(row, column_vector[start_pos], data_vector[start_pos])
    #         return
    #
    #
    #     middleElement = start_pos + int(length_data_vector / 2)
    #
    #     # Add only middle element
    #     self.add_value(row, column_vector[middleElement], data_vector[middleElement])
    #
    #     # begin recursion for higher and lower subtrees
    #     self.build_tree_from_vector_pointer(row, start_pos, middleElement-1, column_vector, data_vector)
    #     self.build_tree_from_vector_pointer(row, middleElement+1, end_pos, column_vector, data_vector)
    #
