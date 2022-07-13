# THIS CONTAINS THE UTILITY FUNCTIONS NECESSARY FOR SOME OF OUR IMPLEMENTATIONS

import numpy as np
from scipy import sparse
import tensorflow as tf


def random_sampler(n_items, n_users, n_samples, replace=False):

    """
    Generates the sampled column indices (items) per row (user). This should be a matrix of shape [n_users, n_samples]

    :param n_items: python int: number of items in interactions
    :param n_users: python int: number of users in interactions
    :param n_samples: python int: number of item samples
    :param replace: python boolean: whether to sample items by replacement or not
    :return: tensorflow tensor: contains n_samples sampled items per user
    """

    items_per_user = [np.random.choice(a=n_items, size=n_samples, replace=replace) for _ in range(n_users)]

    return tf.constant(np.array(items_per_user), dtype=tf.int64)


def generate_random_interaction(n_users, n_items, min_val=0.0, max_val=5.0, density=0.50):
    """
    Generates a random matrix of shape (n_users, n_items) in the range of [min_val, max_val] with density being the proportion of nonzero entries in the tensor.

    :param n_users: python int: number of users (queries)
    :param n_items: python int: number of items (keys)
    :param min_val: python int: minimum value of entries
    :param max_val: python int: maximum value of entries
    :param density: python float: the proportion of nonzero entries in the entire matrix
    :return: tf.sparse.SparseTensor, tf.tensor: representing the interaction table tensor, the sparse tensor is meant to be used for training
    """

    p = sparse.random(n_users, n_items, density=density)

    p = (max_val - min_val) * p + min_val * p.ceil()

    random_arr = np.round(p.toarray())

    scipy_random_arr = sparse.csr_matrix(random_arr)

    # convert to tensorflow tensor

    A = tf.constant(random_arr, dtype=tf.float32)

    # get nonzero elements of sparse matrix

    nonzero_vals = tf.constant(scipy_random_arr.data, dtype=tf.float32)

    row, col = scipy_random_arr.nonzero()

    nonzero_ind = np.array(list(map(np.array, zip(row, col))))

    tf_interactions = tf.sparse.SparseTensor(indices=nonzero_ind, values=nonzero_vals, dense_shape=(n_users, n_items))

    return tf_interactions, A


def gather_matrix_indices(input_arr, index_arr):
    """
    This method takes in column indices per row, indicated in index_arr, and gathers the respective entries in the row and column of index_arr, of input_arr.

    For example:

    input_arr = tf.constant([[1,4,2],
                             [5,7,8],
                             [6,2,1]], dtype=tf.float32)

    index_arr = tf.constant([[0,2,0],
                             [2,2,2]
                             [2,1,0], dtype=tf.int64]

    Now call this method:

    result = gather_matrix_indices(input_arr, index_arr)

    And it results in:

    result = tf.constant([[1, 2, 1],
                          [8, 8, 8],
                          [1, 2, 6]], dtype=tf.float32)



    :param input_arr: a tensorflow tensor, the tensor whose entries we are gathering
    :param index_arr: a tensorflow tensor, the tensor of row-wise indices, must have same number of rows as input_arr
    :return: a tensorflow tensor, the tensor containing the elements of input_arr that correspond to indices in index_arr

    NOTE: No function exists in tensorflow to do this in one function call.
    """
    row_shape, col_shape = index_arr.shape

    # expand column indices
    test_col_ind = index_arr[:, :, tf.newaxis]

    # create row indices
    test_row_ind = tf.repeat(tf.range(row_shape, dtype=tf.int64)[:, tf.newaxis, tf.newaxis], col_shape, axis=1)

    # put row and column indices together
    test_ind = tf.concat([test_row_ind, test_col_ind], axis=-1)

    return tf.gather_nd(indices=test_ind, params=input_arr)

