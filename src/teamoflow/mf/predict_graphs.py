# THIS CONTAINS THE NECESSARY ABSTRACTIONS FOR THE SCORING SYSTEM IN TENSORFLOW

import tensorflow as tf
from abc import *

class PredictionGraph(ABC):
    """
    An abstract base class representing the prediction graphs. These will be objects that compute user-item relationships.
    """
    @abstractmethod
    def get_prediction(self, user_embedding, item_embedding):
        """
        This will give a prediction score based on the user and item embeddings.

        NOTE: This method should be tf-tracing compatible as it must be included in the nodes of the computational graph during gradient computation, therefore, only TensorFlow methods should be utilized in creating any custom prediction graphs.

        :param user_embedding: a tensorflow tensor: the user embedding, of shape [n_users, n_components]
        :param item_embedding: a tensorflow tensor: the item embedding, of shape [n_items, n_components]
        :return: a tensorflow tensor: the prediction, of shape [n_users, n_items] (if otherwise, we will indicate in the respective docstring)
        """
        pass


class DotProductPrediction(PredictionGraph):
    """
    A subclass of the PredictionGraph, abstraction for the dot product prediction for user and item embeddings.
    """

    def get_prediction(self, user_embedding, item_embedding):
        """
        :param user_embedding:
        :param item_embedding:
        :return:
        """
        return tf.matmul(user_embedding, tf.transpose(item_embedding))
