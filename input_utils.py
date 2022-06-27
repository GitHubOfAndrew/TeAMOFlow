# CONTAINS HELPER FUNCTIONS TO GET THE PROPER DATA TYPE INPUT

import numpy as np
import pandas as pd
import tensorflow as tf


# class TensorData:
#     """
#     A class for an object that checks and converts datatypes of the input tensors into our model.
#     """
#
#     def __init__(self):
#         pass
#
#     def convert_to_tensor_constant(self, A):
#         """
#         Arguments:\n
#         - A: A list, pandas dataframe, or a numpy-like array
#
#         Purpose:\n
#         - To convert an iterable (array-like) into a tf CONSTANT tensor
#         """
#
#         # NOTE TO OTHER DEVELOPERS: THIS IS MEANT TO BE FLEXIBLE, IF YOU FIND ANY OTHER ARRAY-LIKE DATATYPE,
#         # FEEL FREE TO PUT IT IN HERE
#
#         # if input is a tf.Tensor, then don't change it
#         if isinstance(A, tf.Tensor):
#             pass
#
#         # convert to constant tensor if the datatype is a list or numpy-like array
#         if isinstance(A, list) or isinstance(A, np.ndarray) or isinstance(A, pd.core.series.Series):
#             return tf.constant(A, dtype=tf.float32)
#
#         # convert to tensor using convert_to_tensor api if the datatype is a pandas dataframe
#         if isinstance(A, pd.DataFrame):
#             return tf.convert_to_tensor(A, dtype=tf.float32)
#
#     def convert_to_tensor_trainable(self, arr):
#         """
#         Arguments:\n
#         - arr: A list, pandas dataframe, or a numpy-like array
#
#         Purpose:\n
#         - To convert an array or any iterable into a tf TRAINABLE tensor
#         """
#         const_arr = self.convert_to_tensor_constant(arr)
#         return tf.Variable(const_arr, trainable=True)
#
# # # test for TensorData class
# #
# # tens_data = TensorData()
# #
# # print(tens_data.convert_to_tensor_trainable(A))