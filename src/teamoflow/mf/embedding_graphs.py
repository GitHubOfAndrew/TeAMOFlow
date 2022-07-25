# THIS FILE CONTAINS ALL THE DIFFERENT TYPES OF EMBEDDINGS THAT CAN BE UTILIZED IN OUR MODEL

import tensorflow as tf
from abc import *


class Embeddings(ABC):
    """
    An abstract base class representing embedding graph objects that will be utilized to compute user and item embeddings
    """
    @abstractmethod
    def get_repr(self, features, weights, aux_dim=None, relu_weight=None, relu_bias=None, linear_bias=None):
        """
        :param features: a tensorflow tensor: the features that will be embedded into the matrix factorization model
        :param weights: a tensorflow tensor: the trainable weights used in the embeddings
        :param aux_dim: a python int: the auxiliary dimension used for relu embedding (only for ReLU Embedding)
        :param relu_weight: a tensorflow tensor: the weight used for the relu layer (only for ReLU Embedding)
        :param relu_bias: a tensorflow tensor: the bias used for the relu layer (only for ReLU Embedding)
        :param linear_bias: a tensorflwo tensor: the bias term for the linear embedding (only for BiasedLinearEmbedding)
        :return: a tensorflow: the embeddings (or representations) to be utilized in the predictions
        """
        pass


class LinearEmbedding(Embeddings):
    """
    A non-biased Linear Embedding, the simplest possible dimension-reduction technique possible
    """

    def get_repr(self, features, weights, aux_dim=None, relu_weight=None, relu_bias=None, linear_bias=None):
        """
        NOTE: Look at docstring for Embeddings.get_repr()
        :param features:
        :param weights:
        :param aux_dim:
        :return:
        """
        return tf.matmul(features, weights), [weights]


class BiasedLinearEmbedding(Embeddings):
    """
    A biased linear embedding. The bias is trainable.
    """
    def get_repr(self, features, weights, aux_dim=None, relu_weight=None, relu_bias=None, linear_bias=None):
        """
        :param features:
        :param weights:
        :param aux_dim:
        :return:
        """
        _, n_components = weights.shape

        if linear_bias is None:
            linear_bias = tf.Variable(tf.zeros(shape=(1, n_components), dtype=tf.float32), trainable=True)

        # linear bias will be broadcasted to all rows of the features x weights matrix, serving as a bias per user/item
        return tf.matmul(features, weights) + linear_bias, [weights, linear_bias]


class ReLUEmbedding(Embeddings):
    """
    A rectified linear unit embedding, specifically, it is a 1-layer neural network with a ReLU activation. Consists of 3 trainable variables.
    """

    def get_repr(self, features, weights, aux_dim=None, relu_weight=None, relu_bias=None, linear_bias=None):
        """
        :param features:
        :param weights:
        :param aux_dim:
        :return:
        """
        _, n_features = features.shape

        if aux_dim is None:
            _, n_components = weights.shape
            aux_dim = 5 * n_components

        # generate the ReLU weights randomly
        if relu_weight is None:
            relu_weight = tf.Variable(tf.random.normal(shape=(n_features, aux_dim), dtype=tf.float32), trainable=True)
        if relu_bias is None:
            relu_bias = tf.Variable(tf.zeros(shape=(1, aux_dim), dtype=tf.float32), trainable=True)

        relu_output = tf.nn.relu(tf.matmul(features, relu_weight) + relu_bias)

        return tf.matmul(relu_output, weights), [weights, relu_weight, relu_bias]

