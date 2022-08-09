# FILE CONTAINING THE NEURAL NETWORK CONFIGURATIONS FOR THE QUERY (USER) FEATURES

import tensorflow as tf
from .utils import *
from .layers import *
from .loss_graphs import *
from .initializer_graphs import *


class QTSoftmax:

    def __init__(self, n_features, li_units, li_activations):
        """
        The Query Tower Softmax model will take in the number of user features and a list of all hidden layer units + output.

        We must also supply a list of activation functions (this is purposefully left vague as it is meant to be customizable).

        :param n_features: python int: number of features in input
        :param li_units: python list: list containing number of units in each layer
        :param li_activations: python list: list containing the activation functions in each layer
        """

        self.activations = li_activations

        self.weights = []

        li_first_comp = [n_features] + li_units[:-1]

        # build the weights using the initializer graphs and the layer template
        for x, y in zip(li_first_comp, li_units):
            self.weights.append(tfLayerWeight(x, y).return_value())


    def predict(self, X):
        """
        Feedforward mechanism. For inferencing.

        :param X: tensorflow tensor: the data to feed in
        :return:
        """
        # instantiate lambda function to apply activation to weights
        lam_func = lambda x, y: x(y)

        x_in = X
        for weights, activation_function in zip(self.weights, self.activations):
            res = tf.matmul(x_in, lam_func(activation_function, weights))
            x_in = res
        logit = x_in

        # return both softmax probabilities and logit
        return tf.nn.softmax(logit, axis=1), logit

    def fit(self):
        pass
