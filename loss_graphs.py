# THIS CONTAINS THE NECESSARY ABSTRACTIONS FOR LOSS FUNCTIONS IN TENSORFLOW

import tensorflow as tf
import numpy as np
from utils import random_sampler
from abc import *

class LossGraph(ABC):
    """
    This is an abstract base class to uniformize the serving of loss functions in our model framework. All loss functions should take in the interaction table, user, item embeddings, and extra arguments unique to each loss function.
    """

    @abstractmethod
    def invoke_loss_graph(self, A, U, V, lambda_1, lambda_2, n_samples):
        """
        As a method of an abstract base class, this does nothing. We will write subclasses of this ABC to share this method.\n

        Arguments:
        - A: tensorflow tensor, the interaction table
        - U: tensorflow tensor, the user embedding matrix
        - V: tensorflow tensor, the item embedding matrix
        - *args: extra arguments corresponding to the specific loss graphs
        """
        pass


class MSE(LossGraph):
    """
    Class meant to serve abstraction layer for the Mean Square Error Loss for implementation in tensorflow's computational graph.

    NOTE: This is the default loss graph that will be utilized by all MatrixFactorization objects. We must specify other loss functions if we want to use it.
    """

    def invoke_loss_graph(self, A, U, V, lambda_1=0.01, lambda_2=0.001, n_samples=None):
        """
        Extra Arguments:\n
        - *args: unpacked iterable, for MSE, we input the hyperparameters lambda_1, lambda_2
        """

        return tf.reduce_mean(tf.where(A != 0, tf.multiply(lambda_1, tf.pow(A - tf.matmul(U, tf.transpose(V)), 2)),
                        tf.multiply(lambda_2, tf.pow(-tf.matmul(U, tf.transpose(V)), 2))))


class WMRB_1(LossGraph):
    """
    Class meant to serve as an abstraction layer for the Weighted-Margin Rank Batch Loss (WMRB) as described in this paper: https://arxiv.org/pdf/1711.04015.pdf.\n
    There are slight tweaks we can do to this loss function to achieve better results.
    """

    def invoke_loss_graph(self, A, U, V, lambda_1=None, lambda_2=None, n_samples=3):
        """
        Extra Arguments:\n
        - *args: unpacked iterable, for WRMB_1, we input n_samples

        Purpose:\n
        - To compute the WMRB loss (link to paper first introducing it: https://arxiv.org/pdf/1711.04015.pdf)
        - returns a tensorflow tensor (a floating point value)
        """

        n_items, _ = V.shape
        n_users, _ = U.shape

        # sampled (without replacement) indices
        sampled_ind = random_sampler(n_items, n_users, n_samples)

        # generate predictions
        predictions = tf.matmul(U, tf.transpose(V))

        # implement WMRB loss described in paper in class docstring
        summation_terms = tf.maximum(tf.where(A == 0, 1.0 - A + predictions, 0.0), 0.0)

        margin_rank = ((n_users * n_items) / n_samples) * tf.reduce_sum(tf.gather_nd(params=summation_terms, indices=sampled_ind))

        return tf.math.log(1.0 + margin_rank)

