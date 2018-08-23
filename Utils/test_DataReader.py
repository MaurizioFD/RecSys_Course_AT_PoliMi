#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 02/03/18

@author: Maurizio Ferrari Dacrema
"""

# Check whether they work correctly

from data.TheMoviesDataset.TheMoviesDatasetReader import TheMoviesDatasetReader
from data.NetflixEnhanced.NetflixEnhancedReader import NetflixEnhancedReader
from data.BookCrossing.BookCrossingReader import BookCrossingReader
from data.XingChallenge2016.XingChallenge2016Reader import XingChallenge2016Reader
from data.XingChallenge2017.XingChallenge2017Reader import XingChallenge2017Reader
from data.AmazonReviewData.AmazonBooks.AmazonBooksReader import AmazonBooksReader
from data.AmazonReviewData.AmazonAutomotive.AmazonAutomotiveReader import AmazonAutomotiveReader
from data.AmazonReviewData.AmazonElectronics.AmazonElectronicsReader import AmazonElectronicsReader
from data.AmazonReviewData.AmazonInstantVideo.AmazonInstantVideo import AmazonInstantVideoReader
from data.AmazonReviewData.AmazonMusicalInstruments.AmazonMusicalInstruments import AmazonMusicalInstrumentsReader
from data.Movielens_1m.Movielens1MReader import Movielens1MReader
from data.Movielens_20m.Movielens20MReader import Movielens20MReader
from data.NetflixPrize.NetflixPrizeReader import NetflixPrizeReader
from data.ThirtyMusic.ThirtyMusicReader import ThirtyMusicReader


from data.DataSplitter import DataSplitter_Warm
import traceback


def read_split_for_data_reader(dataReader_class, force_new_split = False):

    dataReader = dataReader_class()
    dataSplitter = DataSplitter_Warm(dataReader_class, ICM_to_load=None, force_new_split=force_new_split)

    URM_train = dataSplitter.get_URM_train()
    URM_validation = dataSplitter.get_URM_validation()
    URM_test = dataSplitter.get_URM_test()




dataReader_list = [
    Movielens1MReader,
    Movielens20MReader,
    NetflixPrizeReader,
    TheMoviesDatasetReader,
    BookCrossingReader,
    NetflixEnhancedReader,
    XingChallenge2016Reader,
    XingChallenge2017Reader,
    AmazonAutomotiveReader,
    AmazonBooksReader,
    AmazonMusicalInstrumentsReader,
    AmazonInstantVideoReader,
    AmazonElectronicsReader,
    ThirtyMusicReader,

]

test_list = []


for dataReader_class in dataReader_list:

    try:

        read_split_for_data_reader(dataReader_class, force_new_split = False)

        print("Test for: {} - OK".format(dataReader_class))
        test_list.append((dataReader_class, "OK"))


    except Exception as exception:

        traceback.print_exc()

        print("Test for: {} - Trying to generate new split".format(dataReader_class))

        try:

            read_split_for_data_reader(dataReader_class, force_new_split = True)

            print("Test for: {} - OK".format(dataReader_class))
            test_list.append((dataReader_class, "OK"))


        except Exception as exception:

            traceback.print_exc()

            print("Test for: {} - FAIL".format(dataReader_class))
            test_list.append((dataReader_class, "FAIL"))






print("\n\n\n\nSUMMARY:")

for dataReader_class, outcome in test_list:

    print("Test for: {} - {}".format(dataReader_class, outcome))