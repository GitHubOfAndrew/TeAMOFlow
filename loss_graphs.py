# THIS CONTAINS THE NECESSARY ABSTRACTIONS FOR LOSS FUNCTIONS IN TENSORFLOW

import tensorflow as tf
import numpy as np
# from utils import random_sampler
from abc import *


class LossGraph(ABC):
    """
    An abstract base class representing the various loss functions that can be utilized in our matrix factorization model
    """

    @abstractmethod
    def get_loss(self, tf_interactions, user_embedding, item_embedding):
        """
        :param tf_interactions: a tf.sparse.SparseTensor object, (i.e. a sparse tensor): represents the interaction table in sparse tensor form, of dense shape [n_users, n_items]
        :param user_embedding: a tensorflow tensor: the user embedding (representation), of shape [n_users, n_user_features]
        :param item_embedding: a tensorflow tensor: the item embedding (representation), of shape [n_items, n_item_features]
        :return: a tensorflow tensor: vector containing the loss functions, the vector's dimensionality is equivalent to the number of nonzero entries in tf_interactions
        """
        pass


class MSELoss(LossGraph):
    """
    A strict Mean Squared Error Loss, it computes the loss by looking at the observed interactions (nonzeros in the interaction table)
    """

    def get_loss(self, tf_interactions, user_embedding, item_embedding):

        # this is an array of indices corresponding to nonzeros in tf_interactions
        nonzero_ind = tf_interactions.indices

        # this is an array of predictions corresponding to the nonzeros in tf_interactions
        tf_predictions = tf.gather_nd(params=tf.matmul(user_embedding, tf.transpose(item_embedding)), indices=nonzero_ind)

        return tf.square(tf_interactions.values - tf_predictions)
