# FILE CONTAINING THE LOSS FUNCTIONS

import tensorflow as tf
from abc import *

class LossGraph(ABC):
    """
    Abstract base class representing loss graphs to be used.
    """

    @abstractmethod
    def get_loss(self, y, y_hat):
        """
        The abstract method to compute the loss. All loss computations should always take in the ground truth and prediction.

        :param y: tensorflow tensor: the ground truth labels
        :param y_hat: tensorflow tensor: the predicted labels
        :return: tensorflow tensor: whatever the return type of the loss is
        """
        pass


class CrossEntropy(LossGraph):
    """
    Class representing the cross entropy loss. This class admits the general case of cross entropy loss for multi-class classification.

    NOTE: You must use one-hot encoded labels here.
    """

    def get_loss(self, y, y_hat):
        """
        Computes the cross entropy loss per row (user) for the entire batch.

        :param y:
        :param y_hat:
        :return:
        """
        return -tf.reduce_sum(y * tf.math.log(y_hat + 1e-4), axis=1)
