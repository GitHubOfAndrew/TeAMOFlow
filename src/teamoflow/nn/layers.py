# FILE CONTAINING THE LAYERS

import tensorflow as tf
from .initializer_graphs import *

class tfLayerWeight:
    """
    The class for the layers, a fundamental building block of the neural network.
    """

    def __init__(self, x_num, y_num, initializer=NormalInitializer):
        """
        :param x_num: python int: input dimension
        :param y_num: python int: output dimension
        :param initializer: an instance of Initializer(), it is the graph containing the initialized weights
        """
        self.weight = initializer(x_num, y_num)

    def return_value(self):
        """
        Gets the initialized weights, representing the layer pre-activation.

        :return:
        """
        return self.weight.initialize_weights()