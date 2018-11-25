
import time

import numpy as np
cimport numpy as np
from cpython.array cimport array, clone

import scipy.sparse as sps


cdef class Cosine_Similarity:

    def __init__(self, URM, TopK = 100):
        """
        Dataset must be a matrix with items as columns
        :param dataset:
        :param TopK:
        """

        super(Cosine_Similarity, self).__init__()

        self.n_items = URM.shape[1]

        self.TopK = min(TopK, self.n_items)

        URM = URM.tocsr()
        self.user_to_item_row_ptr = URM.indptr
        self.user_to_item_cols = URM.indices
        self.user_to_item_data = np.array(URM.data, dtype=np.float64)

        URM = URM.tocsc()
        self.item_to_user_rows = URM.indices
        self.item_to_user_col_ptr = URM.indptr
        self.item_to_user_data = np.array(URM.data, dtype=np.float64)

        if self.TopK == 0:
            self.W_dense = np.zeros((self.n_items,self.n_items))



    def getUsersThatRatedItem(self, item_id):
        return self.item_to_user_rows[self.item_to_user_col_ptr[item_id]:self.item_to_user_col_ptr[item_id+1]]

    def getItemsRatedByUser(self, user_id):
        return self.user_to_item_cols[self.user_to_item_row_ptr[user_id]:self.user_to_item_row_ptr[user_id+1]]



    def computeItemSimilarities(self, item_id_input):
        """
        For every item the cosine similarity against other items depends on whether they have users in common. 
        The more common users the higher the similarity.
        
        The basic implementation is:
        - Select the first item
        - Loop through all other items
        -- Given the two items, get the users they have in common
        -- Update the similarity considering all common users
        
        That is VERY slow due to the common user part, in which a long data structure is looped multiple times.
        
        A better way is to use the data structure in a different way skipping the search part, getting directly
        the information we need.
        
        The implementation here used is:
        - Select the first item
        - Initialize a zero valued array for the similarities
        - Get the users who rated the first item
        - Loop through the users
        -- Given a user, get the items he rated (second item)
        -- Update the similarity of the items he rated
        
        
        """

        # Create template used to initialize an array with zeros
        result = np.zeros(self.n_items)


        users_that_rated_item = self.getUsersThatRatedItem(item_id_input)

        # Get users that rated the items
        for user_index in range(len(users_that_rated_item)):

            user_id = users_that_rated_item[user_index]
            rating_item_input = self.item_to_user_data[self.item_to_user_col_ptr[item_id_input]+user_index]

            # Get all items rated by that user
            items_rated_by_user = self.getItemsRatedByUser(user_id)

            for item_index in range(len(items_rated_by_user)):

                item_id_second = items_rated_by_user[item_index]

                # Do not compute the similarity on the diagonal
                if item_id_second != item_id_input:
                    # Increment similairty
                    rating_item_second = self.user_to_item_data[self.user_to_item_row_ptr[user_id]+item_index]

                    result[item_id_second] += rating_item_input*rating_item_second

        return result


    def compute_similarity(self):

        # Data structure to incrementally build sparse matrix
        # Preinitialize max possible length
        values = np.zeros((self.n_items*self.TopK))
        rows = np.zeros((self.n_items*self.TopK,), dtype=np.int32)
        cols = np.zeros((self.n_items*self.TopK,), dtype=np.int32)
        sparse_data_pointer = 0

        processedItems = 0


        start_time = time.time()

        # Compute all similarities for each item
        for itemIndex in range(self.n_items):

            processedItems += 1

            if processedItems % 10000==0 or processedItems==self.n_items:

                itemPerSec = processedItems/(time.time()-start_time)

                print("Similarity item {} ( {:2.0f} % ), {:.2f} item/sec, required time {:.2f} min".format(
                    processedItems, processedItems*1.0/self.n_items*100, itemPerSec, (self.n_items-processedItems) / itemPerSec / 60))

            this_item_weights = self.computeItemSimilarities(itemIndex)

            if self.TopK == 0:

                for innerItemIndex in range(self.n_items):
                    self.W_dense[innerItemIndex,itemIndex] = this_item_weights[innerItemIndex]

            else:

                # Sort indices and select TopK
                # Using numpy implies some overhead, unfortunately the plain C qsort function is even slower
                # top_k_idx = np.argsort(this_item_weights) [-self.TopK:]

                # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                # because we avoid sorting elements we already know we don't care about
                # - Partition the data to extract the set of TopK items, this set is unsorted
                # - Sort only the TopK items, discarding the rest
                # - Get the original item index

                this_item_weights_np = - np.array(this_item_weights)

                # Get the unordered set of topK items
                top_k_partition = np.argpartition(this_item_weights_np, self.TopK-1)[0:self.TopK]
                # Sort only the elements in the partition
                top_k_partition_sorting = np.argsort(this_item_weights_np[top_k_partition])
                # Get original index
                top_k_idx = top_k_partition[top_k_partition_sorting]



                # Incrementally build sparse matrix
                for innerItemIndex in range(len(top_k_idx)):

                    topKItemIndex = top_k_idx[innerItemIndex]

                    values[sparse_data_pointer] = this_item_weights[topKItemIndex]
                    rows[sparse_data_pointer] = topKItemIndex
                    cols[sparse_data_pointer] = itemIndex

                    sparse_data_pointer += 1


        if self.TopK == 0:

            return np.array(self.W_dense)

        else:

            values = np.array(values[0:sparse_data_pointer])
            rows = np.array(rows[0:sparse_data_pointer])
            cols = np.array(cols[0:sparse_data_pointer])

            W_sparse = sps.csr_matrix((values, (rows, cols)),
                                    shape=(self.n_items, self.n_items),
                                    dtype=np.float32)

            return W_sparse


