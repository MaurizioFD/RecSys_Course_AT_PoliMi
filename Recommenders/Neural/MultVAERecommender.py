#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 31/10/18
@author: Maurizio Ferrari Dacrema, Cesare Bernardis
"""

try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow is not available")

import os,  shutil, zipfile

import numpy as np
from scipy import sparse
from Recommenders.Neural.architecture_utils import generate_autoencoder_architecture


from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.BaseTempFolder import BaseTempFolder
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.DataIO import DataIO



class _MultDAE_original(object):
    def __init__(self, p_dims, q_dims=None, lam=0.01, lr=1e-3, random_seed=None):
        self.p_dims = p_dims
        if q_dims is None:
            self.q_dims = p_dims[::-1]
        else:
            assert q_dims[0] == p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims
        self.dims = self.q_dims + self.p_dims[1:]

        self.lam = lam
        self.lr = lr
        self.random_seed = random_seed

        self.construct_placeholders()

    def construct_placeholders(self):
        self.input_ph = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=[None, self.dims[0]])
        self.keep_prob_ph = tf.compat.v1.placeholder_with_default(1.0, shape=None)

    def build_graph(self):

        self.construct_weights()

        saver, logits = self.forward_pass()
        log_softmax_var = tf.nn.log_softmax(logits)

        # per-user average negative log-likelihood
        neg_ll = -tf.reduce_mean(input_tensor=tf.reduce_sum(
            input_tensor=log_softmax_var * self.input_ph, axis=1))
        # apply regularization to weights
        reg = tf.keras.regularizers.l2(self.lam)
        loss = neg_ll + sum(reg(w) for w in self.weights)

        train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(loss)

        # add summary statistics
        tf.compat.v1.summary.scalar('negative_multi_ll', neg_ll)
        tf.compat.v1.summary.scalar('loss', loss)
        merged = tf.compat.v1.summary.merge_all()
        return saver, logits, loss, train_op, merged

    def forward_pass(self):
        # construct forward graph
        h = tf.nn.l2_normalize(self.input_ph, 1)
        h = tf.nn.dropout(h, 1 - (self.keep_prob_ph))

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = tf.matmul(h, w) + b

            if i != len(self.weights) - 1:
                h = tf.nn.tanh(h)
        return tf.compat.v1.train.Saver(), h

    def construct_weights(self):

        self.weights = []
        self.biases = []

        # define weights
        for i, (d_in, d_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            weight_key = "weight_{}to{}".format(i, i+1)
            bias_key = "bias_{}".format(i+1)

            self.weights.append(tf.compat.v1.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                    scale=1.0, mode="fan_avg", distribution="uniform", seed=self.random_seed)))

            self.biases.append(tf.compat.v1.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.compat.v1.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))

            # add summary stats
            # tf.compat.v1.summary.histogram(weight_key, self.weights[-1])
            # tf.compat.v1.summary.histogram(bias_key, self.biases[-1])




class _MultVAE_original(_MultDAE_original):

    def construct_placeholders(self):
        super(_MultVAE_original, self).construct_placeholders()

        # placeholders with default values when scoring
        self.is_training_ph = tf.compat.v1.placeholder_with_default(0., shape=None)
        self.anneal_ph = tf.compat.v1.placeholder_with_default(1., shape=None)

    def build_graph(self):
        self._construct_weights()

        saver, logits, KL = self.forward_pass()
        log_softmax_var = tf.nn.log_softmax(logits)

        neg_ll = -tf.reduce_mean(input_tensor=tf.reduce_sum(
            input_tensor=log_softmax_var * self.input_ph,
            axis=-1))
        # apply regularization to weights
        reg = tf.keras.regularizers.l2(self.lam)
        neg_ELBO = neg_ll + self.anneal_ph * KL + sum(reg(w) for w in self.weights_q + self.weights_p)

        train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(neg_ELBO)

        # add summary statistics
        tf.compat.v1.summary.scalar('negative_multi_ll', neg_ll)
        tf.compat.v1.summary.scalar('KL', KL)
        tf.compat.v1.summary.scalar('neg_ELBO_train', neg_ELBO)
        merged = tf.compat.v1.summary.merge_all()

        return saver, logits, neg_ELBO, train_op, merged

    def q_graph(self):
        mu_q, std_q, KL = None, None, None

        h = tf.nn.l2_normalize(self.input_ph, 1)
        h = tf.nn.dropout(h, rate = 1 - self.keep_prob_ph)

        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            h = tf.matmul(h, w) + b

            if i != len(self.weights_q) - 1:
                h = tf.nn.tanh(h)
            else:
                mu_q = h[:, :self.q_dims[-1]]
                logvar_q = h[:, self.q_dims[-1]:]

                std_q = tf.exp(0.5 * logvar_q)
                KL = tf.reduce_mean(input_tensor=tf.reduce_sum(
                        input_tensor=0.5 * (-logvar_q + tf.exp(logvar_q) + mu_q**2 - 1), axis=1))
        return mu_q, std_q, KL

    def p_graph(self, z):
        h = z

        for i, (w, b) in enumerate(zip(self.weights_p, self.biases_p)):
            h = tf.matmul(h, w) + b

            if i != len(self.weights_p) - 1:
                h = tf.nn.tanh(h)
        return h

    def forward_pass(self):
        # q-network
        mu_q, std_q, KL = self.q_graph()
        epsilon = tf.random.normal(tf.shape(input=std_q))

        sampled_z = mu_q + self.is_training_ph *\
            epsilon * std_q

        # p-network
        logits = self.p_graph(sampled_z)

        return tf.compat.v1.train.Saver(), logits, KL

    def _construct_weights(self):
        self.weights_q, self.biases_q = [], []

        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                # we need two sets of parameters for mean and variance,
                # respectively
                d_out *= 2
            weight_key = "weight_q_{}to{}".format(i, i+1)
            bias_key = "bias_q_{}".format(i+1)

            self.weights_q.append(tf.compat.v1.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                    scale=1.0, mode="fan_avg", distribution="uniform", seed=self.random_seed)))

            self.biases_q.append(tf.compat.v1.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.compat.v1.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))

            # add summary stats
            # tf.compat.v1.summary.histogram(weight_key, self.weights_q[-1])
            # tf.compat.v1.summary.histogram(bias_key, self.biases_q[-1])

        self.weights_p, self.biases_p = [], []

        for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
            weight_key = "weight_p_{}to{}".format(i, i+1)
            bias_key = "bias_p_{}".format(i+1)
            self.weights_p.append(tf.compat.v1.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                    scale=1.0, mode="fan_avg", distribution="uniform", seed=self.random_seed)))

            self.biases_p.append(tf.compat.v1.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.compat.v1.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))

            # add summary stats
            # tf.compat.v1.summary.histogram(weight_key, self.weights_p[-1])
            # tf.compat.v1.summary.histogram(bias_key, self.biases_p[-1])






class MultVAERecommender(BaseRecommender, Incremental_Training_Early_Stopping, BaseTempFolder):

    RECOMMENDER_NAME = "MultVAERecommender"


    def __init__(self, URM_train, force_gpu = False):
        super(MultVAERecommender, self).__init__(URM_train)

        if force_gpu:
            assert len(tf.config.list_physical_devices('GPU'))>=1


    def _compute_item_score(self, user_id_array, items_to_compute = None):

        URM_train_user_slice = self.URM_train[user_id_array]

        if sparse.isspmatrix(URM_train_user_slice):
            URM_train_user_slice = URM_train_user_slice.toarray()

        URM_train_user_slice = URM_train_user_slice.astype('float32')

        item_scores_to_compute = self.sess.run(self.logits_var, feed_dict={self.vae.input_ph: URM_train_user_slice})

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf
            item_scores[:, items_to_compute] = item_scores_to_compute[:, items_to_compute]
        else:
            item_scores = item_scores_to_compute

        return item_scores



    def fit(self,
            epochs=100,
            learning_rate=1e-3,
            batch_size=500,
            dropout=0.5,
            total_anneal_steps=200000,
            anneal_cap=0.2,
            p_dims=None,
            l2_reg=0.01,
            temp_file_folder=None,
            **earlystopping_kwargs):

        self.temp_file_folder = self._get_unique_temp_folder(input_temp_file_folder=temp_file_folder)

        self.batch_size = batch_size
        self.total_anneal_steps = total_anneal_steps
        self.anneal_cap = anneal_cap
        self.batches_per_epoch = int(np.ceil(float(self.n_users) / batch_size))
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.update_count = 0.0

        if p_dims is None:
            self.p_dims = [200, 600]
        else:
            self.p_dims = p_dims

        if self.p_dims[-1] != self.n_items:
            self.p_dims.append(self.n_items)

        self._get_clean_session()

        q_dims = self.p_dims[::-1]

        self.vae = _MultVAE_original(self.p_dims, q_dims=q_dims, lr=self.learning_rate, lam=self.l2_reg, random_seed=98765)
        self.saver, self.logits_var, self.loss_var, self.train_op_var, self.merged_var = self.vae.build_graph()

        self.sess.run(tf.compat.v1.global_variables_initializer())

        self._update_best_model()

        try:

            self._train_with_early_stopping(epochs,
                                            algorithm_name = self.RECOMMENDER_NAME,
                                            **earlystopping_kwargs)

            self.load_model(self.temp_file_folder, file_name="_best_model", create_zip = False)

        except (Exception, tf.errors.InvalidArgumentError) as e:
            raise e

        finally:
            self._clean_temp_folder(temp_file_folder=self.temp_file_folder)


    def _get_clean_session(self):
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.disable_eager_execution()

        try:
            if self.sess is not None:
                self.sess.close()
        except AttributeError:
            pass

        self.sess = tf.compat.v1.Session()


    def _prepare_model_for_validation(self):
        pass


    def _update_best_model(self):
        self.save_model(self.temp_file_folder, file_name="_best_model", create_zip = False)


    def _run_epoch(self, num_epoch):

        user_index_list_train = list(range(self.n_users))

        np.random.shuffle(user_index_list_train)

        # train for one epoch
        for bnum, st_idx in enumerate(range(0, self.n_users, self.batch_size)):

            end_idx = min(st_idx + self.batch_size, self.n_users)
            X = self.URM_train[user_index_list_train[st_idx:end_idx]]

            if sparse.isspmatrix(X):
                X = X.toarray()
            X = X.astype('float32')

            if self.total_anneal_steps > 0:
                anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
            else:
                anneal = self.anneal_cap

            feed_dict = {self.vae.input_ph: X,
                         self.vae.keep_prob_ph: self.dropout,
                         self.vae.anneal_ph: anneal,
                         self.vae.is_training_ph: 1}
            self.sess.run(self.train_op_var, feed_dict=feed_dict)

            self.update_count += 1








    def save_model(self, folder_path, file_name = None, create_zip = True):

        #https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        # Save session within a temp folder called as the desired file_name
        if not os.path.isdir(folder_path + file_name + "/.session"):
            os.makedirs(folder_path + file_name + "/.session")

        saver = tf.compat.v1.train.Saver()
        saver.save(self.sess, folder_path + file_name + "/.session/session")

        data_dict_to_save = {
            "batch_size": self.batch_size,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "l2_reg": self.l2_reg,
            "total_anneal_steps": self.total_anneal_steps,
            "anneal_cap": self.anneal_cap,
            "update_count": self.update_count,
            "p_dims": self.p_dims,
            "batches_per_epoch": self.batches_per_epoch,
            # "log_dir": self.log_dir,
            # "chkpt_dir": self.chkpt_dir,
        }

        dataIO = DataIO(folder_path=folder_path + file_name + "/")
        dataIO.save_data(file_name="fit_attributes", data_dict_to_save = data_dict_to_save)

        # Create a zip folder containing fit_attributes and saved session
        if create_zip:
            # Unfortunately I cannot avoid compression so it is too slow for earlystopping
            shutil.make_archive(
              folder_path + file_name,          # name of the file to create
              'zip',                            # archive format - or tar, bztar, gztar
              root_dir = folder_path + file_name + "/",     # root for archive
              base_dir = None)                  # start archiving from the root_dir

            shutil.rmtree(folder_path + file_name + "/", ignore_errors=True)

        self._print("Saving complete")





    def load_model(self, folder_path, file_name = None, create_zip = True):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        if create_zip:
            shutil.unpack_archive(folder_path + file_name + ".zip",
                                  folder_path + file_name + "/",
                                  "zip")

        dataIO = DataIO(folder_path=folder_path + file_name + "/")
        data_dict = dataIO.load_data(file_name="fit_attributes")

        for attrib_name in data_dict.keys():
             self.__setattr__(attrib_name, data_dict[attrib_name])

        self._get_clean_session()

        q_dims = self.p_dims[::-1]

        self.vae = _MultVAE_original(self.p_dims, q_dims=q_dims, lr=self.learning_rate, lam=self.l2_reg, random_seed=98765)
        self.saver, self.logits_var, self.loss_var, self.train_op_var, self.merged_var = self.vae.build_graph()

        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.saver.restore(self.sess, folder_path + file_name + "/.session/session")

        shutil.rmtree(folder_path + file_name + "/", ignore_errors=True)

        self._print("Loading complete")


class MultVAERecommender_OptimizerMask(MultVAERecommender):

    def fit(self, epochs=100, batch_size=500, total_anneal_steps=200000, learning_rate=1e-3, l2_reg=0.01,
            dropout=0.5, anneal_cap=0.2, encoding_size = 50, next_layer_size_multiplier = 2, max_parameters = np.inf, max_n_hidden_layers = 3,
            temp_file_folder=None, **earlystopping_kwargs):

        assert next_layer_size_multiplier > 1.0, "next_layer_size_multiplier must be > 1.0"
        assert encoding_size <= self.n_items, "encoding_size must be <= the number of items"

        p_dims = generate_autoencoder_architecture(encoding_size, self.n_items, next_layer_size_multiplier, max_parameters, max_n_hidden_layers)

        self._print("Architecture: {}".format(p_dims))

        super(MultVAERecommender_OptimizerMask, self).fit(epochs=epochs, batch_size=batch_size, dropout=dropout, learning_rate=learning_rate,
                total_anneal_steps=total_anneal_steps, anneal_cap=anneal_cap, p_dims=p_dims, l2_reg=l2_reg,
                temp_file_folder=temp_file_folder, **earlystopping_kwargs)
