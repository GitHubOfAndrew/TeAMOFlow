# THIS FILE CONTAINS UNIT TESTS FOR THE loss_graphs MODULE

import tensorflow as tf

from src.teamoflow.mf.loss_graphs import *
from src.teamoflow.mf.utils import generate_random_interaction
from src.teamoflow.mf.matrix_factorization import MatrixFactorization

from unittest import TestCase


class TestLossGraph(TestCase):
    """
    Class to perform unit tests of loss graph functionality in teamoflow.mf library.
    """

    @classmethod
    def generate_information(cls):
        cls.n_components = 3
        cls.epochs = 25
        cls.user_features, cls.item_features = tf.eye(100), tf.eye(100)
        cls.sparse_interaction, cls.dense_interaction = generate_random_interaction(n_users=100, n_items=100,
                                                                                    density=0.05)

    def test_mseloss(self):
        """
        This method will fit a MatrixFactorization model with the mse loss.

        :return:
        """
        mf_model = MatrixFactorization(self.n_components)

        mf_model.fit(epochs=self.epochs, user_features=self.user_features, item_features=self.item_features,
                     tf_interactions=self.sparse_interaction)

    def test_wmrbloss(self):
        """
        This method will fit a Matrix Factorization model with the wmrb loss.

        :return:
        """
        mf_model = MatrixFactorization(self.n_components, loss_graph=WMRBLoss())

        mf_model.fit(epochs=self.epochs, user_features=self.user_features, item_features=self.item_features,
                     tf_interactions=self.sparse_interaction, lr=0.1)


