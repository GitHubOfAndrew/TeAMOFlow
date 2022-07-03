# THIS CONTAINS THE UTILITY FUNCTIONS NECESSARY FOR SOME OF OUR IMPLEMENTATIONS

import numpy as np
from scipy import sparse
import tensorflow as tf

# def random_sampler(n_items, n_users, n_samples, replace=False):
#
#     """
#     Arguments:\n
#     - n_items: python int, the number of items
#     - n_users: python int, the number of users
#     - n_samples: python int, the number of user samples
#     - replace: python boolean, whether we sample with replacement or without\n
#     NOTE: we should almost always sample without replacement as we will get duplicate indices
#
#    Purpose:\n
#    - Randomly sample indices from our existing interaction table
#    - Returns a numpy array of sampled indices
#     """
#
#     items_per_user = [np.random.choice(a=n_items, size=n_samples, replace=replace) for _ in range(n_users)]
#
#     sample_indices = []
#
#     for user, user_items in enumerate(items_per_user):
#         for item in user_items:
#             sample_indices.append((user, item))
#
#     return np.array(sample_indices)


def generate_random_interaction(n_users, n_items, max_entry=5.0, density=0.50):
    random_arr = np.round(max_entry * sparse.random(n_users, n_items, density=density).toarray())

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
    :param input_arr: a tensorflow tensor, the tensor whose entries we are gathering
    :param index_arr: a tensorflow tensor, the tensor of row-wise indices, must have same number of rows as input_arr
    :return: a tensorflow tensor, the tensor containing the elements of input_arr that correspond to indices in index_arr

    NOTE: No function exists in tensorflow to do this.
    """
    row, _ = input_arr.shape

    li = []

    for i in range(row):
        li.append(tf.expand_dims(tf.gather(params=input_arr[i], indices=index_arr[i]), axis=0))

    return tf.concat(li, axis=0)
