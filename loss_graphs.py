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
    def get_loss(self, tf_interactions, tf_sample_predictions, tf_prediction_serial, predictions, n_items, n_samples):
        """
        :param tf_interactions: a tf.sparse.SparseTensor object, (i.e. a sparse tensor): represents the interaction table in sparse tensor form, of dense shape [n_users, n_items]
        :param predictions: a tf tensor: the predictions computed from the user and item embeddings
        :return: a tensorflow tensor: vector containing the loss functions, the vector's dimensionality is equivalent to the number of nonzero entries in tf_interactions
        """
        pass


class MSELoss(LossGraph):
    """
    A strict Mean Squared Error Loss, it computes the loss by looking at the observed interactions (nonzeros in the interaction table)
    """

    def get_loss(self, tf_interactions, predictions, tf_sample_predictions=None, tf_prediction_serial=None, n_items=None, n_samples=None):

        # this is an array of indices corresponding to nonzeros in tf_interactions
        nonzero_ind = tf_interactions.indices

        # this is an array of predictions corresponding to the nonzeros in tf_interactions
        tf_predictions = tf.gather_nd(params=predictions, indices=nonzero_ind)

        return tf.square(tf_interactions.values - tf_predictions)


class WMRBLoss(LossGraph):
    """
    The Weighted-Margin Rank Batch Loss (WMRB). This is a direct implementation of https://arxiv.org/pdf/1711.04015.pdf for only positive interactions
    """

    def get_loss(self, tf_interactions, tf_sample_predictions, tf_prediction_serial, n_items, n_sampled_items, predictions=None):
        """
        :param tf_interactions:
        :param tf_sample_predictions:
        :param tf_prediction_serial:
        :param n_items:
        :param n_sampled_items:
        :return:
        """
        pass
