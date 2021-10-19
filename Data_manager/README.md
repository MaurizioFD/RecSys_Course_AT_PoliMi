# Recommender Systems Framework for Python 3.6

Developed by Maurizio Ferrari Dacrema, PhD candidate at Politecnico di Milano.

### Data split name convention

* _get_dataset_name_root() returns the root directory o the dataset folder tree, e.g., "Movielens_100k/"
* _get_dataset_name_data_subfolder() returns the subfolder inside the dataset folder tree containing the data to be loaded, e.g., "original/" or "5-cores" or "0.30-user-sample"
* _get_split_subfolder_name() returns the subfolder inside the dataset folder tree containing the specified split, e.g., "warm_5_fold/", "cold_item_5_fold"

The full path of the data is therefore: 
 * For the non-splitted data: _get_dataset_name_root() + _get_dataset_name_data_subfolder(), e.g., "Movielens_100k/original/"
 * For the splitted data: _get_dataset_name_root() + _get_split_subfolder_name() + _get_dataset_name_data_subfolder(), e.g., "Movielens_100k/cold_item_5_fold/original/"


### DataReaderPostprocessing
The postprocessing tool has to be applied between the instantiation of the dataReader and the splitter:
For example, if we want to apply the k-cores on the Movielens1M dataset we have to do the following:

```Python
from Data_manager_new.Movielens_1m.Movielens1MReader import Movielens1MReader

from Data_manager_new.DataSplitter_k_fold import DataSplitter_Warm_k_fold
from Data_manager_new.DataReaderPostprocessing_K_Cores import DataReaderPostprocessing_K_Cores

dataset = Movielens1MReader()

dataset = DataReaderPostprocessing_K_Cores(dataset, k_cores_value=25)

dataSplitter = DataSplitter_Warm_k_fold(dataset)

dataSplitter.load_data()

URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

```

The postprocessing tools can be chained together and the corresponding operations are performed in a FIFO ordering.
The closer to the dataset object gets applied earlier. In this example, first we apply a K-fold and then a user sampling, we also want our data to be implicit

```Python
from Data_manager_new.Movielens_1m.Movielens1MReader import Movielens1MReader

from Data_manager_new.DataSplitter_k_fold import DataSplitter_Warm_k_fold
from Data_manager_new.DataReaderPostprocessing_K_Cores import DataReaderPostprocessing_K_Cores
from Data_manager_new.DataReaderPostprocessing_User_sample import DataReaderPostprocessing_User_sample
from Data_manager_new.DataReaderPostprocessing_Implicit_URM import DataReaderPostprocessing_Implicit_URM


dataset = Movielens1MReader()

dataset = DataReaderPostprocessing_K_Cores(dataset, k_cores_value=25)
dataset = DataReaderPostprocessing_User_sample(dataset, user_quota=0.3)
dataset = DataReaderPostprocessing_Implicit_URM(dataset)



dataSplitter = DataSplitter_Warm_k_fold(dataset)

dataSplitter.load_data()

URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

```

The data of that dataset will be in the folder "Movielens_1M/0.3_user_sample/25_cores/..."