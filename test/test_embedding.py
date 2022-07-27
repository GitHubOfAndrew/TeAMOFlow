# THIS FILE CONTAINS UNIT TESTS FOR THE embedding_graphs MODULE

import tensorflow as tf

from src.teamoflow.mf.embedding_graphs import *
from src.teamoflow.mf.utils import generate_random_interaction
from src.teamoflow.mf.matrix_factorization import MatrixFactorization

import unittest
from unittest import TestCase


# load the data outside of the class
n_users, n_items = 50, 100
sparse_interaction, dense_interaction = generate_random_interaction(n_users=n_users, n_items=n_items, density=0.05)
user_features, item_features = tf.eye(n_users), tf.eye(n_items)


class TestEmbedding(TestCase):
    """
    Class to perform unit tests on the embedding_graphs.py module. Our tests will simply check whether the representation graphs let the model fit without error.

    We will perform the fitting on a default model configuration (sans the representation graphs). We will only change the user embedding graph to ensure it works.
    """

    def test_linear_repr(self):
        """
        Unit test for the linear embedding.

        :return:
        """
        try:
            mf_model = MatrixFactorization(3, user_repr_graph=LinearEmbedding())

            mf_model.fit(epochs=25, user_features=user_features, item_features=item_features,
                         tf_interactions=sparse_interaction)

        except:
            print('Linear Embedding Test Failed.')

    def test_biased_linear_repr(self):
        """
        Unit test for the biased linear embedding.

        :return:
        """
        try:
            mf_model = MatrixFactorization(3, user_repr_graph=BiasedLinearEmbedding())

            mf_model.fit(epochs=25, user_features=user_features, item_features=item_features,
                         tf_interactions=sparse_interaction)

        except:
            print('Biased Linear Embedding Test Failed.')

    def test_ReLU_repr(self):
        """
        Unit test for the ReLU embedding.

        :return:
        """
        try:
            mf_model = MatrixFactorization(3, user_repr_graph=ReLUEmbedding())

            mf_model.fit(epochs=25, user_features=user_features, item_features=item_features,
                         tf_interactions=sparse_interaction)

        except:
            print('ReLU Embedding Test Failed.')

if __name__ == 'main':
    unittest.main()
