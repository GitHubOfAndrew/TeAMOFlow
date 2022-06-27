# THIS FILE CONTAINS ALL THE DIFFERENT TYPES OF EMBEDDINGS THAT CAN BE UTILIZED IN OUR MODEL

import tensorflow as tf
from abc import *


class Embeddings(ABC):
    """
    An abstract base class representing embedding graph objects that will be utilized to compute user and item embeddings
    """
    @abstractmethod
    def get_repr(self, features, weights):
        """
        :param features: a tensorflow tensor: the features that will be embedded into the matrix factorization model
        :param weights: a tensorflow tensor: the trainable weights used in the embeddings
        :return: a tensorflow: the embeddings (or representations) to be utilized in the predictions
        """
        pass


class LinearEmbedding(Embeddings):
    """
    A non-biased Linear Embedding, the simplest possible dimension-reduction technique possible
    """

    def get_repr(self, features, weights):
        """
        NOTE: Look at docstring for Embeddings.get_repr()
        :param features:
        :param weights:
        :return:
        """
        return tf.matmul(features, weights)
