# CONTAINS HELPER FUNCTIONS TO GET THE PROPER DATA TYPE INPUT

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import sparse


def convert_np_to_tf_sparse(np_arr):
    """
    :param np_arr: a numpy array
    :return: a tf.sparse.SparseTensor instance (i.e. a sparse tensor)

    Purpose: This is a utility function to help convert between a numpy array and a sparse tensor
    """

    row_dim, col_dim = np_arr.shape

    scipy_arr = sparse.csr_matrix(np_arr)

    nonzero_vals = tf.constant(scipy_arr.data, dtype=tf.float32)

    row, col = scipy_arr.nonzero()

    nonzero_ind = np.array(list(map(np.array, zip(row, col))))

    tf_sparse = tf.sparse.SparseTensor(indices=nonzero_ind, values=nonzero_vals, dense_shape=(row_dim, col_dim))

    return tf_sparse


def convert_tf_to_tf_sparse(tf_arr):
    """
    :param tf_arr: a tensorflow tensor
    :return: sparse tensor
    """
    return convert_np_to_tf_sparse(tf_arr.numpy())


def convert_list_to_tf_sparse(li_arr):
    """
    :param li_arr: a python list
    :return: sparse tensor
    """
    return convert_np_to_tf_sparse(np.array(li_arr))


def convert_df_to_tf_sparse(df_arr):
    """
    :param df_arr: pandas dataframe
    :return: sparse tensor
    """
    return convert_np_to_tf_sparse(np.array(df_arr))


def convert_sp_sparse_to_tf_sparse(sp_arr):
    """
    :param sp_arr: a scipy sparse matrix (csr_matrix)
    :return: sparse tensor
    """

    np_arr = sp_arr.toarray()

    row_dim, col_dim = np_arr.shape

    nonzero_vals = tf.constant(sp_arr.data, dtype=tf.float32)

    row, col = sp_arr.nonzero()

    nonzero_ind = np.array(list(map(np.array, zip(row, col))))

    tf_sparse = tf.sparse.SparseTensor(indices=nonzero_ind, values=nonzero_vals, dense_shape=(row_dim, col_dim))

    return tf_sparse


def convert_to_tf_sparse(arr):
    """
    :param arr: any instance of the following array-like objects: numpy array, list, pandas dataframe, tensorflow tensor, scipy csr_matrix
    :return: tensorflow sparse tensor
    """

    if isinstance(arr, list):
        return convert_list_to_tf_sparse(arr)

    if isinstance(arr, np.ndarray):
        return convert_np_to_tf_sparse(arr)

    if isinstance(arr, pd.DataFrame):
        return convert_df_to_tf_sparse(arr)

    if isinstance(arr, tf.Tensor):
        return convert_tf_to_tf_sparse(arr)

    if isinstance(arr, sparse.csr_matrix):
        return convert_sp_sparse_to_tf_sparse(arr)


def convert_to_tensor_constant(A):
    """
    :param A: instance of an array-like object: numpy arrays, python lists, pandas dataframes/series, tf tensor
    :return: tensorflow tensor
    """

    # NOTE TO OTHER DEVELOPERS: THIS IS MEANT TO BE FLEXIBLE, IF YOU FIND ANY OTHER ARRAY-LIKE DATATYPE,
    # FEEL FREE TO PUT IT IN HERE

    # if input is a tf.Tensor, then don't change it
    if isinstance(A, tf.Tensor):
        pass

    # convert to constant tensor if the datatype is a list or numpy-like array
    if isinstance(A, list) or isinstance(A, np.ndarray) or isinstance(A, pd.core.series.Series):
        return tf.constant(A, dtype=tf.float32)

    # convert to tensor using convert_to_tensor api if the datatype is a pandas dataframe
    if isinstance(A, pd.DataFrame):
        return tf.convert_to_tensor(A, dtype=tf.float32)

def convert_to_tensor_trainable(self, arr):
    """
    Arguments:\n
    - arr: A list, pandas dataframe, or a numpy-like array

    Purpose:\n
    - To convert an array or any iterable into a tf TRAINABLE tensor
    """
    const_arr = self.convert_to_tensor_constant(arr)
    return tf.Variable(const_arr, trainable=True)
