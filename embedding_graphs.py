# THIS FILE CONTAINS ALL THE DIFFERENT TYPES OF EMBEDDINGS THAT CAN BE UTILIZED IN OUR MODEL

import tensorflow as tf
from abc import *


class Embeddings(ABC):
    """
    An abstract base class representing embedding graph objects that will be utilized to compute user and item embeddings
    """
    @abstractmethod
    def get_repr(self, features, weights, aux_dim=None):
        """
        :param features: a tensorflow tensor: the features that will be embedded into the matrix factorization model
        :param weights: a tensorflow tensor: the trainable weights used in the embeddings
        :param aux_dim: a python int: the auxiliary dimension used for relu embedding (only for ReLU Embedding)
        :return: a tensorflow: the embeddings (or representations) to be utilized in the predictions
        """
        pass


class LinearEmbedding(Embeddings):
    """
    A non-biased Linear Embedding, the simplest possible dimension-reduction technique possible
    """

    def get_repr(self, features, weights, aux_dim=None):
        """
        NOTE: Look at docstring for Embeddings.get_repr()
        :param features:
        :param weights:
        :param aux_dim:
        :return:
        """
        return tf.matmul(features, weights)


class BiasedLinearEmbedding(Embeddings):
    """
    A biased linear embedding. The bias is trainable.
    """
    def get_repr(self, features, weights, aux_dim=None):
        """
        :param features:
        :param weights:
        :param aux_dim:
        :return:
        """
        _, n_components = weights.shape

        linear_bias = tf.Variable(tf.zeros(shape=(1, n_components), dtype=tf.float32), trainable=True)

        # linear bias will be broadcasted to all rows of the features x weights matrix, serving as a bias per user/item
        return tf.matmul(features, weights) + linear_bias


class ReLUEmbedding(Embeddings):
    """
    A rectified linear unit embedding. Consists of 3 trainable variables.
    """

    def get_repr(self, features, weights, aux_dim=None):
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
        relu_weight = tf.Variable(tf.random.normal(shape=(n_features, aux_dim), dtype=tf.float32), trainable=True)
        relu_bias = tf.Variable(tf.zeros(shape=(1, aux_dim), dtype=tf.float32), trainable=True)

        relu_output = tf.nn.relu(tf.matmul(features, relu_weight) + relu_bias)

        return tf.matmul(relu_output, weights)

