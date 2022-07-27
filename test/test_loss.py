# THIS FILE CONTAINS UNIT TESTS FOR THE loss_graphs MODULE

import tensorflow as tf

from src.teamoflow.mf.loss_graphs import *
from src.teamoflow.mf.utils import generate_random_interaction
from src.teamoflow.mf.matrix_factorization import MatrixFactorization

import unittest
from unittest import TestCase


# load the data outside of the class
n_users, n_items = 50, 100
n_samples = n_items // 2
sparse_interaction, dense_interaction = generate_random_interaction(n_users=n_users, n_items=n_items, density=0.05)
user_features, item_features = tf.eye(n_users), tf.eye(n_items)

# use this interaction table of positive AND negative interactions for testing kl divergence loss
sparse_mixed_interaction, dense_mixed_interaction = generate_random_interaction(n_users=n_users, n_items=n_items,
                                                                                min_val=-5.0, max_val=5.0, density = 0.01)


class TestLossGraph(TestCase):
    """
    Class to perform unit tests of loss graph functionality in teamoflow.mf library.
    """

    def test_mseloss(self):
        """
        This method will fit a MatrixFactorization model with the mse loss.

        :return:
        """
        try:
            mf_model = MatrixFactorization(3)

            mf_model.fit(epochs=25, user_features=user_features, item_features=item_features,
                         tf_interactions=sparse_interaction)

        except:
            print('MSE Loss Test Failed.')

    def test_wmrbloss(self):
        """
        This method will fit a Matrix Factorization model with the wmrb loss.

        :return:
        """
        try:
            mf_model = MatrixFactorization(3, loss_graph=WMRBLoss(), n_users=n_users, n_items=n_items, generate_sample=True)

            mf_model.fit(25, user_features=user_features, item_features=item_features,
                         tf_interactions=sparse_interaction, lr=0.1)

        except:
            print('WMRB Loss Test Failed.')

    def test_kl_divergence(self):
        """
        This method will fit the matrix factorization model with the KL Divergence loss.

        :return:
        """
        try:
            mf_model = MatrixFactorization(3, loss_graph=KLDivergenceLoss())

            mf_model.fit(25, user_features=user_features, item_features=item_features,
                         tf_interactions=sparse_mixed_interaction, lr=0.1)

        except:
            print('KL Divergence Loss Test Failed.')


if __name__ == 'main':
    unittest.main()
