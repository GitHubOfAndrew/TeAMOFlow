# THIS CONTAINS THE NECESSARY ABSTRACTIONS FOR LOSS FUNCTIONS IN TENSORFLOW

import tensorflow as tf
import numpy as np
from utils import random_sampler

class MSE:
    """
    Class meant to serve abstraction layer for the Mean Square Error Loss for implementation in tensorflow's computational graph.

    NOTE: This is the default loss graph that will be utilized by all MatrixFactorization objects. We must specify other loss functions if we want to use it.
    """
    def __init__(self, A, U, V, lambda_1=0.01, lambda_2=0.001):
        """
        Arguments:\n
        - A: a tensorflow tensor, the original interaction table
        - U: a tensorflow tensor, the user embeddings matrix
        - V: a tensorflow tensor, the item embeddings matrix
        - lambda_1: a python float, the hyperparameter weighting the observed interactions
        - lambda_2: a python float, the hyperparameter weighting the unobserved interactions

        Purpose:\n
        """
        self.A = A
        self.U = U
        self.V = V
        self.lambda_1 = tf.constant(float(lambda_1))
        self.lambda_2 = tf.constant(float(lambda_2))

    def MSE_loss(self):

        return tf.where(self.A != 0, tf.multiply(self.lambda_1, tf.pow(self.A - tf.matmul(self.U, tf.transpose(self.V)), 2)),
                        tf.multiply(self.lambda_2, tf.pow(-tf.matmul(self.U, tf.transpose(self.V)), 2)))


class WMRB_1:
    """
    Class meant to serve as an abstraction layer for the Weighted-Margin Rank Batch Loss (WMRB) as described in this paper: https://arxiv.org/pdf/1711.04015.pdf.\n
    There are slight tweaks we can do to this loss function to achieve better results.
    """
    def __init__(self, A, U, V, n_samples):
        self.A = A
        self.U = U
        self.V = V
        self.n_items = V.shape[0]
        self.n_users = U.shape[0]
        self.n_samples = n_samples

    def weighted_margin_rank_batch_loss(self):
        """
        Arguments:\n
        - A: tensorflow tensor, the original interaction table
        - predictions: tensorflow tensor, the predicted interaction table from our .predict method of the MatrixFactorization class

        Purpose:\n
        - To compute the WMRB loss (link to paper first introducing it: https://arxiv.org/pdf/1711.04015.pdf)
        - returns a tensorflow tensor (a floating point value)
        """

        # sampled (without replacement) indices
        sampled_ind = random_sampler(self.n_items, self.n_users, self.n_samples)

        # generate predictions
        predictions = tf.matmul(self.U, tf.transpose(self.V))

        # implement WMRB loss described in paper in class docstring
        summation_terms = tf.maximum(tf.where(self.A == 0, 1.0 - self.A + predictions), 0.0)

        margin_rank = ((self.n_users * self.n_items) / (self.n_samples)) * tf.reduce_sum(tf.gather_nd(params=summation_terms, indices=sampled_ind))

        return tf.math.log(1.0 + margin_rank)

