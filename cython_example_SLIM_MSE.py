"""
Created on 09/11/2020

@author: Maurizio Ferrari Dacrema
"""

from Notebooks_utils.data_splitter import train_test_holdout
from Data_manager.Movielens.Movielens10MReader import Movielens10MReader

data_reader = Movielens10MReader()
data_loaded = data_reader.load_data()

URM_all = data_loaded.get_URM_all()

URM_train, URM_test = train_test_holdout(URM_all, train_perc = 0.8)


from Cython_examples.SLIM_MSE_fastest import train_multiple_epochs

train_multiple_epochs(URM_train, 1e-4, 10)