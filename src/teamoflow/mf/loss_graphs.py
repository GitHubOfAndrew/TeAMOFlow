# THIS CONTAINS THE NECESSARY ABSTRACTIONS FOR LOSS FUNCTIONS IN TENSORFLOW

import tensorflow as tf
import tensorflow_probability as tp
from abc import *


class LossGraph(ABC):
    """
    An abstract base class representing the various loss functions that can be utilized in our matrix factorization model.
    """

    @abstractmethod
    def get_loss(self, tf_interactions, tf_sample_predictions, tf_prediction_serial, predictions, n_items, n_samples):
        """
        NOTE: These loss graphs are intended to be fully vectorized, tf-tracing compatible, and should be GPU-compatible.

        For custom implementations, please DO NOT use any non-tf functions to perform the computations as this will cause errors with tracing during the training loop. Any function that is in numpy can be replicated with a tensorflow method, please look at tensorflow documentation if this is a concern: https://www.tensorflow.org/api_docs/python/tf

        :param tf_interactions: a tensorflow sparse tensor: the sparse tensor containing the observed interactions, dense shape should be [n_users, n_items]
        :param tf_sample_predictions: a tensorflow tensor: the predictions of the model corresponding to the sampled items per user, it is of shape [n_users, n_samples]
        :param tf_prediction_serial: a tensorflow tensor: the predictions corresponding to the observed interactions, it is of shape [nonzero_interactions, 1]
        :param predictions: a tensorflow tensor: the full set of predictions
        :param n_items: a python int: the number of items
        :param n_samples: a python int: the number of sampled items
        :return:
        """
        pass


class MSELoss(LossGraph):
    """
    A strict Mean Squared Error Loss, it computes the loss by looking at the observed interactions (nonzeros in the interaction table)
    """

    def get_loss(self, tf_interactions, predictions, tf_sample_predictions=None, tf_prediction_serial=None, n_items=None, n_samples=None):
        """
        :param tf_interactions:
        :param predictions:
        :param tf_sample_predictions:
        :param tf_prediction_serial:
        :param n_items:
        :param n_samples:
        :return:
        """
        # this is an array of indices corresponding to nonzeros in tf_interactions
        nonzero_ind = tf_interactions.indices

        # this is an array of predictions corresponding to the nonzeros in tf_interactions
        tf_predictions = tf.gather_nd(params=predictions, indices=nonzero_ind)

        return tf.square(tf_interactions.values - tf_predictions)


class WMRBLoss(LossGraph):
    """
    The Weighted-Margin Rank Batch Loss (WMRB). This is a direct implementation of https://arxiv.org/pdf/1711.04015.pdf for only positive interactions.

    This implementation is a direct result of James Kirk's work in TensorRec (https://github.com/jfkirk/tensorrec/blob/master/tensorrec/loss_graphs.py). My main contribution is in making the workflow/terms in this function clearer and in making the arguments clearer.
    """

    def get_loss(self, tf_interactions, tf_sample_predictions, tf_prediction_serial, n_items, n_samples, predictions=None):
        """
        :param tf_interactions:
        :param tf_sample_predictions:
        :param tf_prediction_serial:
        :param n_items:
        :param n_samples:
        :param predictions:
        :return:
        """
        # this wmrb only takes into account positive interactions, ignores all negative interactions

        positive_interaction_mask = tf.greater(tf_interactions.values, 0.0)

        positive_interaction_indices = tf.boolean_mask(tf_interactions.indices, positive_interaction_mask)

        positive_predictions = tf.boolean_mask(tf_prediction_serial, positive_interaction_mask)

        mapped_predictions_sample_per_interaction = tf.gather(params=tf_sample_predictions,
                                                              indices=tf.transpose(positive_interaction_indices)[0])

        summation_term = tf.maximum(
            1.0 - tf.expand_dims(positive_predictions, axis=1) + mapped_predictions_sample_per_interaction, 0.0)

        sampled_margin_rank = (n_items / n_samples) * tf.reduce_sum(summation_term, axis=1)

        return tf.math.log(1.0 + sampled_margin_rank)


class KLDivergenceLoss(LossGraph):
    """
    The Kullback-Leibler Divergence Loss. This loss handles interaction tables consisting of both positive and negative interactions.

    It models the distribution of positive and negative interactions as normal distributions, and computes the complement of the CDF of their intersection.

    For more resources on this implementation, please refer to: https://www.tensorflow.org/api_docs/python/tf/keras/losses/KLDivergence
    """

    def get_loss(self, tf_prediction_serial, tf_interactions, tf_sample_predictions=None, predictions=None, n_items=None, n_samples=None):
        """
        :param tf_prediction_serial:
        :param tf_interactions:
        :param tf_sample_predictions:
        :param predictions:
        :param n_items:
        :param n_samples:
        :return:
        """

        tf_pos_mask, tf_neg_mask = tf.greater(tf_interactions.values, 0.0), tf.less_equal(tf_interactions.values, 0.0)

        tf_pos_pred, tf_neg_pred = tf.boolean_mask(tf_prediction_serial, tf_pos_mask), tf.boolean_mask(tf_prediction_serial,
                                                                                                       tf_neg_mask)

        tf_pos_mean, tf_pos_var = tf.nn.moments(tf_pos_pred, axes=[0])

        tf_neg_mean, tf_neg_var = tf.nn.moments(tf_neg_pred, axes=[0])

        overlap_dist = tp.distributions.Normal(loc=(tf_neg_mean - tf_pos_mean), scale=tf.sqrt(tf_pos_var + tf_neg_var))

        return 1.0 - overlap_dist.cdf(0.0)

