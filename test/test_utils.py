# THIS FILE CONTAINS THE UNIT TESTS FOR THE utils MODULE

import tensorflow as tf

from src.teamoflow.mf.utils import *

import unittest
from unittest import TestCase


class TestUtils(TestCase):
    """
    Class to perform unit tests on the input_utils.py and utils.py modules.

    The tests will be more traditional, checking a deterministic output to ensure that the functions behave as intended.
    """

    def test_random_sampler(self):
        """
        Test the random_sampler() function. Just check that it produces an output.

        :return:
        """
        n_users, n_items = 50, 100
        n_samples = n_items // 2
        random_sampled_arr = random_sampler(n_items, n_users, n_samples)

        if isinstance(random_sampled_arr, tf.Tensor):
            print('Random Sampler Data Type Test Passed.')
            
            if random_sampled_arr.shape == (n_users, n_samples):
                print('Random Sampler Shape Test Passed.')
            else:
                print('Random Sampler Shape Test Failed. Check that the dimensions are consistent with your input.')
        else:
            print('Random Sampler Data Type Test Failed. Please check if you refactored the function in any manner.')

    def test_gather_matrix_indices(self):
        """
        Test the gather_matrix_indices() function. Check that it is consistent with an example output.

        :return:
        """

        # give toy example from docstring for gather_matrix_indices() function

        input_arr = tf.constant([[1, 4, 2],
                                 [5, 7, 8],
                                 [6, 2, 1]], dtype=tf.float32)

        index_arr = tf.constant([[0, 2, 0],
                                 [2, 2, 2],
                                 [2, 1, 0]], dtype=tf.int64)

        result = tf.constant([[1, 2, 1],
                              [8, 8, 8],
                              [1, 2, 6]], dtype=tf.float32)

        test_arr = gather_matrix_indices(input_arr, index_arr)

        assert tf.reduce_sum(tf.where(result == test_arr, 1.0, 0.0)) == tf.size(result).numpy()

        print('Gather Matrix Indices Test Passed.')

    def test_generate_random_interactions(self):
        """
        Test the generate_random_interaction() function to ensure that it spits out the correct data type and that the dense and sparse interaction tables are consistent with one another.

        :return:
        """
        n_users, n_items = 50, 100
        sparse_int, dense_int = generate_random_interaction(n_users, n_items, density=0.05)

        # throw assertion error if the sparse tensor is not a sparse tensor object
        assert isinstance(sparse_int, tf.sparse.SparseTensor)

        # throw assertion error if the dense tensor is not a tf.Tensor object
        assert isinstance(dense_int, tf.Tensor)

        # check if the dense shape is consistent
        assert tuple(sparse_int.dense_shape.numpy()) == dense_int.shape


if __name__ == 'main':
    unittest.main()
