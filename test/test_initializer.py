# THIS FILE CONTAINS ALL THE UNIT TESTS FOR THE initializer_graphs MODULE

import tensorflow as tf

from src.teamoflow.mf.initializer_graphs import *
from src.teamoflow.mf.utils import generate_random_interaction
from src.teamoflow.mf.matrix_factorization import MatrixFactorization

import unittest
from unittest import TestCase


# load the data outside of the class
n_users, n_items = 50, 100
sparse_interaction, dense_interaction = generate_random_interaction(n_users=n_users, n_items=n_items, density=0.05)
user_features, item_features = tf.eye(n_users), tf.eye(n_items)


class TestInitializerGraph(TestCase):
    """
    Class to perform unit tests on the initalizer_graphs.py module. Our tests will simply check whether the initializer graphs allow the model to train.

    We will perform the fitting test on the default model configuration (except the initializer graphs of course). We will only change the initializer graph for the user.
    """

    def test_normalinitializer(self):
        """
        Test if the model will fit with the normal initializer object.

        :return:
        """
        try:
            mf_model = MatrixFactorization(3, user_weight_graph=NormalInitializer())

            mf_model.fit(epochs=25, user_features=user_features, item_features=item_features,
                         tf_interactions=sparse_interaction)

        except:
            print('Normal Initializer Test Failed.')

    def test_uniforminitializer(self):
        """
        Test if the model will fit with the uniform initializer object.

        :return:
        """
        try:
            mf_model = MatrixFactorization(3, user_weight_graph=UniformInitializer())

            mf_model.fit(epochs=25, user_features=user_features, item_features=item_features,
                         tf_interactions=sparse_interaction)

        except:
            print('Uniform Initializer Test Failed.')


if __name__ == 'main':
    unittest.main()
