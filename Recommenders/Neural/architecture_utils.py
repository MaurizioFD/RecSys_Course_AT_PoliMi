#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/09/2021

@author: Maurizio Ferrari Dacrema
"""

def generate_autoencoder_architecture(encoding_size, output_size, next_layer_size_multiplier,
                                      max_decoder_parameters, max_n_hidden_layers):
    """
    Generates architecture in the form: [encoding_size, encoding_size*next_layer_size_multiplier, ... , output_size] with sizes strictly increasing

    Termination condition, stop generating the architecture if:

    - New layer size is too close to the n_items considering the layer size progression that is provided as hyperparameter
    - Total parameter size is larger than the maximum threshold (for the sake of RAM)
    - Maximum architecture depth is reached

    At this point the last layer will be replaced with the output one having size output_size.
    The architecture is always guaranteed to contain at least the encoding_size and the output_layer even if the embedding is very large

    Number of parameters for a dense layer: num_params = (input_size + output_size) * output_size

    :param encoding_size:
    :param output_size:
    :param next_layer_size_multiplier:
    :param max_decoder_parameters:
    :param max_n_hidden_layers:
    :return:
    """

    assert encoding_size < output_size, "The encoding_size must be strictly lower than the output_size"

    layers = [encoding_size]
    n_total_parameters = 0
    n_last_layer_parameters = (layers[-1] + 1)*output_size

    terminate = False

    while not terminate:

        next_layer_size = int(layers[-1]*next_layer_size_multiplier)

        assert next_layer_size != layers[-1], "Two consecutive layers have the same number of neurons. " \
                                              "Infinite loop in constructing the architecture. Layers are: {}".format(layers)

        # Calculate parameters for the input of the new layer and update those needed for the last
        n_total_parameters += (layers[-1] + 1)*next_layer_size
        n_last_layer_parameters = (next_layer_size + 1) * output_size

        # Consistency check w.r.t. the desired limits:
        # - Does the model exceed the maximum number of layers
        if len(layers) >= max_n_hidden_layers:
            terminate = True

        # - Is the new layer too large compared to the output_size
        elif next_layer_size * next_layer_size_multiplier > output_size:
            terminate = True

        # - Does the model exceed the maximum number of parameters
        elif n_total_parameters + n_last_layer_parameters > max_decoder_parameters:
            # print("Terminating architecture as total of parameters adding an additional layer {:.2E} would exceed the allowed maximum {:.2E}".format(n_total_parameters + n_last_layer_parameters, max_decoder_parameters))
            terminate = True

        else:
            layers.append(next_layer_size)


    # Add the output layer
    layers.append(output_size)

    return layers





def generate_autoencoder_architecture_previous(encoding_size, last_layer_size, next_layer_size_multiplier, max_layer_size, max_n_hidden_layers):
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
