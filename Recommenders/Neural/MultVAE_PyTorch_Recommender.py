#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 24/06/2023

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.DataIO import DataIO
import scipy.sparse as sps
import numpy as np
from tqdm import tqdm
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
import torch, copy, math
from torch.autograd import Variable
import torch.nn.functional as f
from Recommenders.Neural.architecture_utils import generate_autoencoder_architecture
from Utils.PyTorch.utils import get_optimizer, clone_pytorch_model_to_numpy_dict

def from_sparse_to_tensor(A_tilde):
    A_tilde = sps.coo_matrix(A_tilde)
    A_tilde = torch.sparse_coo_tensor(np.vstack([A_tilde.row, A_tilde.col]), A_tilde.data, A_tilde.shape)
    A_tilde = A_tilde.coalesce()

    return A_tilde


torch.set_default_dtype(torch.float32)

class _Encoder(torch.nn.Module):

    def __init__(self, options, dropout_p=0.5, q_dims=None):
        super(_Encoder, self).__init__()
        self.options = options
        self.q_dims = q_dims

        self._network = torch.nn.Sequential()
        self._network.add_module("dropout_{}".format(0), torch.nn.Dropout(p=dropout_p))

        for i in range(len(q_dims)-1):
            in_features = q_dims[i]
            out_features = q_dims[i+1]
            if i == len(q_dims)-2:
                out_features*=2

            self._network.add_module("layer_{}".format(i), torch.nn.Linear(in_features = in_features, out_features = out_features, bias=True))
            if i != len(q_dims)-2:
                self._network.add_module("activation_{}".format(i), torch.nn.Tanh())


        for module_name, m in self.named_modules():
            if isinstance(m, torch.nn.Linear):
                # Based on original initialization
                # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling
                # torch.nn.init.xavier_uniform_(m.weight.data)
                n = (m.in_features + m.out_features)/2
                scale = 1.0
                limit = math.sqrt(3 * scale / n)
                torch.nn.init.uniform_(m.weight.data, a=-limit, b=limit)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.001)

    def forward(self, x):
        x = self._network(x)
        mu_q, logvar_q = torch.chunk(x, chunks=2, dim=1)
        return mu_q, logvar_q


class _Decoder(torch.nn.Module):
    def __init__(self, options, p_dims=None):
        super(_Decoder, self).__init__()
        self.options = options
        self.p_dims = p_dims

        self._network = torch.nn.Sequential()

        for i in range(len(p_dims)-1):
            in_features = p_dims[i]
            out_features = p_dims[i+1]
            self._network.add_module("layer_{}".format(i), torch.nn.Linear(in_features = in_features, out_features = out_features, bias=True))
            if i != len(p_dims)-2:
                self._network.add_module("activation_{}".format(i), torch.nn.Tanh())

        for module_name, m in self.named_modules():
            if isinstance(m, torch.nn.Linear):
                # Based on original initialization
                # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling
                # torch.nn.init.xavier_uniform_(m.weight.data)
                n = (m.in_features + m.out_features)/2
                scale = 1.0
                limit = math.sqrt(3 * scale / n)
                torch.nn.init.uniform_(m.weight.data, a=-limit, b=limit)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.001)
                    # m.bias.data = truncated_normal(m)
                    # truncated_normal_(m.bias, mean=0, std=0.001)

    def forward(self, x):
        x = self._network(x)
        return x

class _MultiVAEModel(torch.nn.Module):
    def __init__(self, dropout_p, q_dims, p_dims, device):
        super(_MultiVAEModel, self).__init__()
        self.device = device
        self.q_dims = q_dims
        self.p_dims = p_dims

        self.encoder = _Encoder(None, dropout_p=dropout_p, q_dims=self.q_dims)
        self.decoder = _Decoder(None, p_dims=self.p_dims)

    def forward(self, x):
        x = f.normalize(x, p=2, dim=1)

        mu_q, logvar_q = self.encoder.forward(x)
        std_q = torch.exp(0.5 * logvar_q)
        KL = torch.mean(torch.sum(0.5 * (-logvar_q + torch.exp(logvar_q) + mu_q ** 2 - 1), dim=1))
        epsilon = torch.randn_like(std_q, requires_grad=False)

        if self.training:
            sampled_z = mu_q + epsilon * std_q
        else:
            sampled_z = mu_q

        logits = self.decoder.forward(sampled_z)

        return logits, KL, mu_q, std_q, epsilon, sampled_z

    def get_l2_reg(self):
        l2_reg = Variable(torch.FloatTensor(1), requires_grad=True).to(self.device)
        for k, m in self.state_dict().items():
            if k.endswith('.weight'): # The original MultVAE implementation only regularizes on the weights and not the biases
                l2_reg = l2_reg + torch.norm(m, p=2) ** 2

        return l2_reg[0]



class MultVAERecommender_PyTorch(BaseRecommender, Incremental_Training_Early_Stopping):
    """ MultVAERecommender_PyTorch



    """

    RECOMMENDER_NAME = "MultVAERecommender_PyTorch"

    def __init__(self, URM_train, use_gpu=False, verbose=False):
        super(MultVAERecommender_PyTorch, self).__init__(URM_train, verbose=verbose)

        if use_gpu:
            assert torch.cuda.is_available(), "GPU is requested but not available"
            self.device = torch.device("cuda:0")
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu:0")

        self.warm_user_ids = np.arange(0, self.n_users)[np.ediff1d(sps.csr_matrix(self.URM_train).indptr) > 0]


    def _compute_item_score(self, user_id_array, items_to_compute = None):

        u = torch.LongTensor(user_id_array)

        # Transferring only the sparse structure to reduce the data transfer
        user_batch_tensor = self.URM_train[u]
        user_batch_tensor = torch.sparse_csr_tensor(user_batch_tensor.indptr,
                                                    user_batch_tensor.indices,
                                                    user_batch_tensor.data,
                                                    size=user_batch_tensor.shape, dtype=torch.float32,
                                                    device=self.device, requires_grad=False).to_dense()

        with torch.no_grad():
            self._model.eval()
            logits, _, _, _, _, _ = self._model.forward(user_batch_tensor)

        item_scores_to_compute = logits.cpu().detach().numpy()

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf
            item_scores[:, items_to_compute] = item_scores_to_compute[:, items_to_compute]
        else:
            item_scores = item_scores_to_compute

        return item_scores


    def _init_model(self, dropout, p_dims, device):
        """
        This function instantiates the model, it should only rely on attributes and not function parameters
        It should be used both in the fit function and in the load_model function
        :return:
        """

        torch.cuda.empty_cache()

        self._model = _MultiVAEModel(dropout_p=dropout,
                                     q_dims=p_dims[::-1],
                                     p_dims=p_dims,
                                     device=device).to(device)
        self._model.eval()

    def fit(self,
            epochs=10,
            learning_rate=1e-3,
            batch_size=500,
            dropout=0.5,
            # embedding_size=None,
            total_anneal_steps=200000,
            anneal_cap=0.2,
            p_dims=None,
            l2_reg=0.01,
            sgd_mode=None,
            **earlystopping_kwargs):

        assert p_dims[-1] == self.n_items, "p_dims inconsistent with data, first dimension expected to be the number of items"

        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.update_count = 0
        self.anneal_cap = anneal_cap
        self.total_anneal_steps = total_anneal_steps
        self.dropout = dropout
        self.p_dims = p_dims

        torch.cuda.empty_cache()

        self._init_model(self.dropout, self.p_dims, self.device)

        self._optimizer = get_optimizer(sgd_mode.lower(), self._model, learning_rate, 0.0)

        ###############################################################################
        ### This is a standard training with early stopping part

        # Initializing for epoch 0
        self._prepare_model_for_validation()
        self._update_best_model()

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:

        self._train_with_early_stopping(epochs,
                                        algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
        # prof.export_chrome_trace("trace.json")

        self._print("Training complete")
        self._model_state = self._model_state_best
        self._model.load_state_dict({key: torch.from_numpy(value).to(self.device) for key, value in self._model_state_best.items()}, strict=True)


    def _prepare_model_for_validation(self):
        with torch.no_grad():
            self._model.eval()
            self._model_state = clone_pytorch_model_to_numpy_dict(self._model)
            self._model.train()


    def _update_best_model(self):
        self._model_state_best = copy.deepcopy(self._model_state)

    def _run_epoch(self, num_epoch):

        num_batches_per_epoch = math.ceil(len(self.warm_user_ids) / self.batch_size)

        if self.verbose:
            batch_iterator = tqdm(range(0, num_batches_per_epoch))
        else:
            batch_iterator = range(0, num_batches_per_epoch)

        self._model.train()
        epoch_loss = 0

        for _ in batch_iterator:

            # Clear previously computed gradients
            self._optimizer.zero_grad()

            u = torch.LongTensor(np.random.choice(self.warm_user_ids, size=self.batch_size))

            # Transferring only the sparse structure to reduce the data transfer
            user_batch_tensor = self.URM_train[u]
            user_batch_tensor = torch.sparse_csr_tensor(user_batch_tensor.indptr,
                                                        user_batch_tensor.indices,
                                                        user_batch_tensor.data,
                                                        size=user_batch_tensor.shape, dtype=torch.float32, device=self.device, requires_grad=False).to_dense()

            logits, KL, mu_q, std_q, epsilon, sampled_z = self._model.forward(user_batch_tensor)

            log_softmax_var = f.log_softmax(logits, dim=1)
            neg_ll = - torch.mean(torch.sum(log_softmax_var * user_batch_tensor, dim=1))
            l2_reg = self._model.get_l2_reg()

            if self.total_anneal_steps > 0:
                anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
            else:
                anneal = self.anneal_cap

            loss = neg_ll + anneal * KL + l2_reg * self.l2_reg
            self.update_count += 1

            # Compute gradients given current loss
            loss.backward()
            epoch_loss += loss.item()

            # Apply gradient using the selected _optimizer
            self._optimizer.step()

        self._print("Loss {:.2E}".format(epoch_loss))
        self._model.eval()




    def save_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {
            # Hyperparameters
            'dropout': self.dropout,
            'p_dims': self.p_dims,

            # Model parameters
            "_model_state": self._model_state,
        }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        self._print("Saving complete")

    def load_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        # Reload the attributes dictionary
        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        for attrib_name in data_dict.keys():
            if not attrib_name.startswith("model_"):
                self.__setattr__(attrib_name, data_dict[attrib_name])


        self._init_model(self.dropout, self.p_dims, self.device)

        self._model.load_state_dict({key: torch.from_numpy(value).to(self.device) for key, value in data_dict["_model_state"].items()}, strict=True)

        self._print("Loading complete")







class MultVAERecommender_PyTorch_OptimizerMask(MultVAERecommender_PyTorch):

    def fit(self, epochs=10,
            batch_size=500,
            total_anneal_steps=200000,
            learning_rate=1e-3,
            l2_reg=0.01,
            dropout=0.5,
            anneal_cap=0.2,
            sgd_mode = "adam",
            encoding_size = 50,
            next_layer_size_multiplier = 2,
            max_parameters = np.inf,
            max_n_hidden_layers = 3,
            **earlystopping_kwargs):

        assert next_layer_size_multiplier > 1.0, "next_layer_size_multiplier must be > 1.0"
        assert encoding_size <= self.n_items, "encoding_size must be <= the number of items"

        p_dims = generate_autoencoder_architecture(encoding_size, self.n_items, next_layer_size_multiplier, max_parameters, max_n_hidden_layers)

        self._print("Architecture: {}".format(p_dims))

        super(MultVAERecommender_PyTorch_OptimizerMask, self).fit(epochs=epochs, batch_size=batch_size, dropout=dropout, learning_rate=learning_rate,
                total_anneal_steps=total_anneal_steps, anneal_cap=anneal_cap, p_dims=p_dims, l2_reg=l2_reg, sgd_mode=sgd_mode,
                **earlystopping_kwargs)

