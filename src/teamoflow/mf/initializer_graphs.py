# FILE CONTAINS ABSTRACTION LAYER FOR WEIGHT INITIALIZERS

import tensorflow as tf
from abc import *


class Initializer(ABC):
    """
    This is an abstract base class for weight initializers to be used in the matrix factorization model
    """

    @abstractmethod
    def initialize_weights(self, n_features, n_components):
        """
        :param n_features: python int: the number of features used in the input into the embedding
        :param n_components: python int: the dimension of the embedding space (the number of latent features)
        :return: tensorflow tensor: the weights to be used in the embedding
        """
        pass


class NormalInitializer(Initializer):
    """
    A subclass of Initializer, representing the object to initialize weights that are randomly normally distributed
    """

    def initialize_weights(self, n_features, n_components):
        """
        NOTE: Look in docstring for Initializer().initialize_weights()
        :param n_features:
        :param n_components:
        :return:
        """
        weight = tf.Variable(tf.math.l2_normalize(tf.random.normal(shape=(n_features, n_components), dtype=tf.float32)), trainable=True)
        return weight


class UniformInitializer(Initializer):
    """
    A subclass of Initializer, representing the object to initialize weights that are uniformly distributed
    """

    def initialize_weights(self, n_features, n_components):
        """
        NOTE: Look in docstring for Initializer().initialize_weights()

        :param n_features:
        :param n_components:
        :return:
        """
        weight = tf.Variable(tf.math.l2_normalize(tf.random.uniform(shape=(n_features, n_components), dtype=tf.float32)), trainable=True)
        return weight
