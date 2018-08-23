"""
Created on 16/09/2017

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

import numpy as np
cimport numpy as np

from Base.Recommender_utils import check_matrix
import time

from libc.float cimport DBL_MAX
from cpython.array cimport array, clone


cdef class Similarity_Matrix_Evaluator:


    cdef int[:] W_sparse_indices, W_sparse_indptr
    cdef double[:] W_sparse_data
    cdef double[:] W_sparse_support_vector


    cdef double[:] current_item_scores

    cdef double[:,:] W

    cdef int[:] URM_train_indices, URM_train_indptr
    cdef double[:] URM_train_data

    cdef int[:] URM_test_indices, URM_test_indptr
    cdef double[:] URM_test_data



    cdef int sparse_weights, filterTopPop, normalize
    cdef int[:] filterTopPop_ItemsID

    cdef int n_items




    def __init__(self, SimilarityMatrix, URM_test, URM_train, filterTopPop = False, filterTopPop_ItemsID=None, normalize=False):
        super(Similarity_Matrix_Evaluator, self).__init__()



        self.n_items = URM_train.shape[1]
        self.current_item_scores = np.zeros(self.n_items, dtype=np.float64)


        if isinstance(SimilarityMatrix, np.ndarray):
            self.W = np.array(SimilarityMatrix, dtype=np.float64)

            self.sparse_weights = False

        else:
            SimilarityMatrix = check_matrix(SimilarityMatrix, 'csc')
            self.W_sparse_indices = np.array(SimilarityMatrix.indices, dtype=np.int32)
            self.W_sparse_indptr = np.array(SimilarityMatrix.indptr, dtype=np.int32)
            self.W_sparse_data = np.array(SimilarityMatrix.data, dtype=np.float64)

            self.W_sparse_support_vector = np.zeros((self.n_items), dtype=np.float64)

            self.sparse_weights = True



        URM_train = check_matrix(URM_train, 'csr')
        URM_test = check_matrix(URM_test, 'csr')

        self.URM_test_indptr = np.array(URM_test.indptr, dtype=np.int32)
        self.URM_test_indices = np.array(URM_test.indices, dtype=np.int32)
        self.URM_test_data = np.array(URM_test.indices, dtype=np.float64)

        self.URM_train_indptr = np.array(URM_train.indptr, dtype=np.int32)
        self.URM_train_indices = np.array(URM_train.indices, dtype=np.int32)
        self.URM_train_data = np.array(URM_train.data, dtype=np.float64)


        self.filterTopPop = filterTopPop
        self.filterTopPop_ItemsID = filterTopPop_ItemsID

        self.normalize = normalize



    cdef int [:] get_user_seen_items(self, int user_id):
        return self.URM_train_indices[self.URM_train_indptr[user_id]:self.URM_train_indptr[user_id+1]]

    cdef double [:] get_user_seen_items_ratings(self, int user_id):
        return self.URM_train_data[self.URM_train_indptr[user_id]:self.URM_train_indptr[user_id+1]]



    cdef int [:] get_user_relevant_items(self, int user_id):
        return self.URM_test_indices[self.URM_test_indptr[user_id]:self.URM_test_indptr[user_id+1]]

    cdef double [:] get_user_relevant_items_ratings(self, int user_id):
        return self.URM_test_data[self.URM_test_indptr[user_id]:self.URM_test_indptr[user_id+1]]


    cdef int [:] get_item_weights(self, int item_id):
        return self.W_sparse_indices[self.W_sparse_indptr[item_id]:self.W_sparse_indptr[item_id+1]]

    cdef double [:] get_item_weights_value(self, int item_id):
        return self.W_sparse_data[self.W_sparse_indptr[item_id]:self.W_sparse_indptr[item_id+1]]



    cdef _filter_TopPop_on_scores(self):

        cdef int item_index, item_id

        for item_index in range(len(self.filterTopPop_ItemsID)):
            item_id = self.filterTopPop_ItemsID[item_index]

            self.current_item_scores[item_id] = -DBL_MAX





    cdef _filter_seen_on_scores(self, long user_id):

        cdef int item_index, item_id
        cdef int[:] seen = self.get_user_seen_items(user_id)

        for item_index in range(len(seen)):
            item_id = seen[item_index]
            self.current_item_scores[item_id] = -DBL_MAX





    def evaluateRecommendations(self, int[:] usersToEvaluate, int at=5, int exclude_seen=True, int filterTopPop = False):

        start_time = time.time()

        cdef double roc_auc_= 0.0, precision_= 0.0, recall_= 0.0, map_= 0.0, mrr_= 0.0, ndcg_ = 0.0

        cdef long test_user, index, n_eval = 0
        cdef int[:] relevant_items
        cdef long[:] recommended_items
        cdef int outer_index, inner_index
        cdef int[:] is_relevant = np.zeros((at), dtype=np.int32)


        for index in range(len(usersToEvaluate)):

            test_user = usersToEvaluate[index]

            relevant_items = self.get_user_relevant_items(test_user)

            n_eval += 1

            #print("initialized {}".format(time.time()-start_time))

            recommended_items = self.recommend(user_id=test_user, remove_seen_flag=exclude_seen,
                                               cutoff=at, remove_top_pop_flag=filterTopPop)

            #print("recommended_items {}".format(time.time()-start_time))

            for outer_index in range(at):
                inner_index = 0

                while inner_index < len(relevant_items) and recommended_items[outer_index] != relevant_items[inner_index]:

                    inner_index += 1

                if inner_index == len(relevant_items):
                    is_relevant[outer_index] = 0
                else:
                    is_relevant[outer_index] = 1


            #print("is_relevant {}".format(time.time()-start_time))


            #is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            # evaluate the recommendation list with ranking metrics ONLY
            roc_auc_ += roc_auc(is_relevant)
            precision_ += precision(is_relevant)
            recall_ += recall(is_relevant, relevant_items)
            map_ += map(is_relevant, relevant_items)
            mrr_ += rr(is_relevant)
            ndcg_ += ndcg(recommended_items, relevant_items, relevance=self.get_user_relevant_items_ratings(test_user), at=at)

            #print("ranking metrics {}".format(time.time()-start_time))

            #input()

            if(index % 1000 == 0 and index!=0):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Users per second: {:.0f}".format(
                                  n_eval,
                                  100.0* float(n_eval)/len(usersToEvaluate),
                                  time.time()-start_time,
                                  float(n_eval)/(time.time()-start_time)))




        if (n_eval > 0):
            roc_auc_ /= n_eval
            precision_ /= n_eval
            recall_ /= n_eval
            map_ /= n_eval
            mrr_ /= n_eval
            ndcg_ /= n_eval

        else:
            print("WARNING: No users had a sufficient number of relevant items")

        results_run = {}

        results_run["AUC"] = roc_auc_
        results_run["precision"] = precision_
        results_run["recall"] = recall_
        results_run["map"] = map_
        results_run["NDCG"] = ndcg_
        results_run["MRR"] = mrr_

        return (results_run)




    cpdef long[:] recommend(self, long user_id, n=5, exclude_seen=True, filterTopPop = False):


        cdef int item_index, item_id
        cdef int similar_item_index,  similar_item_id, seen_item_index, seen_item_id
        cdef int[:] item_weights_indices, user_profile
        cdef double[:] item_weights_value

        cdef np.ndarray[long, ndim=1] relevant_items_partition_sorting, relevant_items_partition, ranking
        cdef np.ndarray[np.float64_t, ndim=1] scores_array

        #start_time = time.time()

        user_profile = self.get_user_seen_items(user_id)

        #print("user_profile {}".format(time.time()-start_time))

        # compute the scores using the dot product
        if self.sparse_weights:

            # Get weights for each item and put them in a dense column
            for item_index in range(self.n_items):

                item_weights_indices = self.get_item_weights(item_index)
                item_weights_value = self.get_item_weights_value(item_index)

                # Put current weights in dense representation
                for similar_item_index in range(len(item_weights_indices)):
                    similar_item_id = item_weights_indices[similar_item_index]

                    self.W_sparse_support_vector[similar_item_id] = item_weights_value[similar_item_index]


                # Now compute the summation of all W corresponding to a seen item
                for seen_item_index in range(len(user_profile)):
                    seen_item_id = user_profile[seen_item_index]

                    self.current_item_scores[item_index] += self.W_sparse_support_vector[seen_item_id]


                # Clear data structure
                for similar_item_index in range(len(item_weights_indices)):
                    similar_item_id = item_weights_indices[similar_item_index]

                    self.W_sparse_support_vector[similar_item_id] = 0.0


            #print("self.W_sparse_support_vector {}".format(time.time()-start_time))


        else:

            for seen_item_index in range(len(user_profile)):
                seen_item_id = user_profile[seen_item_index]

                for item_index in range(self.n_items):
                    self.current_item_scores[item_index] += self.W[item_index, seen_item_id]


            #print("self.current_item_scores {}".format(time.time()-start_time))





        if self.normalize:
            raise ValueError("Normalize not implemented")

        if exclude_seen:
            scores = self._remove_seen_on_scores(user_id)

        if filterTopPop:
            scores = self._remove_TopPop_on_scores()


        #print("filterTopPop {}".format(time.time()-start_time))


        scores_array = np.array(self.current_item_scores)

        for item_index in range(self.n_items):
            self.current_item_scores[item_index] = 0.0


        #print("scores_array {}".format(time.time()-start_time))

        # rank items and mirror column to obtain a ranking in descending score
        #ranking = scores.argsort()
        #ranking = np.flip(ranking, axis=0)

        # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
        # - Partition the data to extract the set of relevant items
        # - Sort only the relevant items
        # - Get the original item index
        relevant_items_partition = (-scores_array).argpartition(n)[0:n]
        relevant_items_partition_sorting = np.argsort(-scores_array[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]

        #input()

        return ranking




from libc.math cimport log



cdef double min (double a, double b):

    if a > b:
        return b

    return a


cpdef double roc_auc(int[:] is_relevant):

    cdef int index, neg_ranks=0, pos_ranks=0
    cdef double auc_score = 0.0

    for index in range(len(is_relevant)):

        if is_relevant[index] == 1:
            pos_ranks+=1
            auc_score+=neg_ranks

        else:
            neg_ranks+=1

    if neg_ranks == 0:
        return 1.0

    elif pos_ranks > 0:
        auc_score /= (pos_ranks*neg_ranks*1.0)

    return auc_score



cpdef double precision(int[:] is_relevant):

    cdef double precision_score = 0.0
    cdef int index

    for index in range(len(is_relevant)):
        precision_score += is_relevant[index]

    return precision_score / len(is_relevant)



cpdef double recall(int[:] is_relevant, int[:] pos_items):

    cdef double recall_score = 0.0
    cdef int index

    for index in range(len(is_relevant)):
        recall_score += is_relevant[index]

    return recall_score / len(pos_items)


cpdef double rr(int[:] is_relevant):
    # reciprocal rank of the FIRST relevant item in the ranked list (0 if none)

    cdef int index

    for index in range(len(is_relevant)):
        if is_relevant[index] == 1:
            return 1.0 / (index+1)

    return 0.0


cpdef double map(int[:] is_relevant, int[:] pos_items):

    cdef int index, cumsum=0
    cdef double p_at_k, map_score = 0.0

    for index in range(len(is_relevant)):

        cumsum += is_relevant[index]
        p_at_k = cumsum * is_relevant[index] / (index + 1.0)
        map_score += p_at_k

    map_score /= min(len(is_relevant), len(pos_items))

    return map_score


cpdef double ndcg(long[:] ranked_list, int[:] pos_items, double[:] relevance, int at):

    cdef double ndcg_, rank_dcg, ideal_dcg

    it2rel = {it: r for it, r in zip(pos_items, relevance)}

    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in ranked_list[:at]], dtype=np.float64)

    ideal_dcg = dcg(np.sort(relevance)[::-1])
    rank_dcg = dcg(rank_scores)
    ndcg_ = rank_dcg / ideal_dcg

    return ndcg_


cdef double dcg(double[:] scores):

    cdef int index
    cdef double dcg_score = 0.0

    for index in range(len(scores)):
        scores[index] = scores[index]**2 - 1

        dcg_score += scores[index] / log(index + 2.0)

    return dcg_score
