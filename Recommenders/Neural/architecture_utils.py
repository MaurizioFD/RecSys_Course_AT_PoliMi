#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/09/2021

@author: Maurizio Ferrari Dacrema
"""

def generate_autoencoder_architecture(encoding_size, last_layer_size, next_layer_size_multiplier, max_layer_size, max_n_hidden_layers):
    """
    Generates architecture in the form: [encoding_size, encoding_size*next_layer_size_multiplier, ... , last_layer_size] with sizes strictly increasing

    Termination condition, stop generating the architecture if:
    -
    - New layer size is too close to the n_items considering the layer size progression that is provided as hyperparameter
    - New layer size is larger than the maximum threshold (for the sake of RAM)
    - Maximum architecture depth is reached

    At this point the last layer will be replaced with the output one having size last_layer_size.
    The architecture is always guaranteed to contain at least the encoding_size and the output_layer even if the embedding is very large

    :param encoding_size:
    :param last_layer_size:
    :param next_layer_size_multiplier:
    :param max_layer_size:
    :param max_n_hidden_layers:
    :return:
    """

    assert encoding_size < last_layer_size, "The encoding_size must be strictly lower than the last_layer_size"

    layers = [encoding_size]

    while last_layer_size/layers[-1] > next_layer_size_multiplier*0.9 and layers[-1] < max_layer_size and len(layers) <= max_n_hidden_layers:
        next_layer_size = int(layers[-1]*next_layer_size_multiplier)
        assert next_layer_size != layers[-1], "Two consecutive layers have the same number of neurons. " \
                                              "Infinite loop in constructing the architecture. Layers are: {}".format(layers)

        layers.append(next_layer_size)

    # Ensure that the encoding layer is *never* removed
    if len(layers) > 1:
        layers[-1] = last_layer_size
    else:
        layers.append(last_layer_size)

    return layers
