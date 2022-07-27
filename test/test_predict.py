# THIS FILE CONTAINS ALL THE UNIT TESTS FOR THE predict_graph MODULE

import tensorflow as tf

from src.teamoflow.mf.predict_graphs import *

from src.teamoflow.mf.utils import generate_random_interaction
from src.teamoflow.mf.matrix_factorization import MatrixFactorization

import unittest
from unittest import TestCase


# load the data outside of the class
n_users, n_items = 50, 100
sparse_interaction, dense_interaction = generate_random_interaction(n_users=n_users, n_items=n_items, density=0.05)
user_features, item_features = tf.eye(n_users), tf.eye(n_items)


class TestPrediction(TestCase):
    """
    Class to perform unit tests of predict_graphs.py module. Our tests will simply check whether the prediction graphs let the model fit without error.

    We will perform the fitting on a default model configuration (sans prediction graph). As of now, we only have one prediction graph (dot product) and we haven't made accommodations for it yet. We will do this later. We will simply fit the default model as it uses the dot product prediction graph.
    """

    def test_dot_product_prediction(self):
        mf_model = MatrixFactorization(3)

        mf_model.fit(epochs=25, user_features=user_features, item_features=item_features,
                     tf_interactions=sparse_interaction)

        # call .predict
        prediction = mf_model.predict()

        if prediction is not None:
            pass
        else:
            print('Graph Test Failed.')


if __name__ == 'main':
    unittest.main()
