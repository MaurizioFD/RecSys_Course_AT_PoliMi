

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 5/09/2017
@author: Maurizio Ferrari Dacrema
"""

import numpy as np


class BPR_Sampling(object):

    def __init__(self):
        super(BPR_Sampling, self).__init__()


    def sampleUser(self):
        """
        Sample a user that has viewed at least one and not all items
        :return: user_id
        """
        while (True):

            user_id = np.random.randint(0, self.n_users)
            numSeenItems = self.URM_train[user_id].nnz

            if (numSeenItems > 0 and numSeenItems < self.n_items):
                return user_id


    def sampleItemPair(self, user_id):
        """
        Returns for the given user a random seen item and a random not seen item
        :param user_id:
        :return: pos_item_id, neg_item_id
        """

        userSeenItems = self.URM_train[user_id].indices

        pos_item_id = userSeenItems[np.random.randint(0, len(userSeenItems))]

        while (True):

            neg_item_id = np.random.randint(0, self.n_items)

            if (neg_item_id not in userSeenItems):
                return pos_item_id, neg_item_id


    def sampleTriple(self):
        """
        Randomly samples a user and then samples randomly a seen and not seen item
        :return: user_id, pos_item_id, neg_item_id
        """

        user_id = self.sampleUser()
        pos_item_id, neg_item_id = self.sampleItemPair(user_id)

        return user_id, pos_item_id, neg_item_id


    def initializeFastSampling(self, positive_threshold=3):
        print("Initializing fast sampling")

        self.eligibleUsers = []
        self.userSeenItems = dict()

        # Select only positive interactions
        URM_train_positive = self.URM_train.multiply(self.URM_train>positive_threshold)

        for user_id in range(self.n_users):

            if (URM_train_positive[user_id].nnz > 0):
                self.eligibleUsers.append(user_id)
                self.userSeenItems[user_id] = URM_train_positive[user_id].indices

        self.eligibleUsers = np.array(self.eligibleUsers)


    def sampleBatch(self):
        user_id_list = np.random.choice(self.eligibleUsers, size=(self.batch_size))
        pos_item_id_list = [None]*self.batch_size
        neg_item_id_list = [None]*self.batch_size

        for sample_index in range(self.batch_size):
            user_id = user_id_list[sample_index]

            pos_item_id_list[sample_index] = np.random.choice(self.userSeenItems[user_id])

            negItemSelected = False

            # It's faster to just try again then to build a mapping of the non-seen items
            # for every user
            while (not negItemSelected):
                neg_item_id = np.random.randint(0, self.n_items)

                if (neg_item_id not in self.userSeenItems[user_id]):
                    negItemSelected = True
                    neg_item_id_list[sample_index] = neg_item_id

        return user_id_list, pos_item_id_list, neg_item_id_list