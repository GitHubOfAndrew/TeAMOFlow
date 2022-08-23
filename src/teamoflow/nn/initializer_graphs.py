# FILE CONTAINING ALL INITIALIZER GRAPHS

import tensorflow as tf
from abc import *

class Initializer(ABC):
    """
    Initializer class to initialize the trainable variables for the neural network.
    """

    @abstractmethod
    def initialize_weights(self):
        """
        Abstract method that must be fulfilled by all subclasses of this class. This will initialize weights and make sure they are trainable.

        :return: a tensorflow variable tensor: trainable and to be traced in the gradient computation and optimization
        """
        pass


class NormalInitializer(Initializer):
    """
    The normal initializer.
    """
    def __init__(self, x_num, y_num):
        """
        :param x_num: python int: input dimension
        :param y_num: python int: output dimension
        """
        self.x_num = x_num
        self.y_num = y_num

    def initialize_weights(self):
        """
        Normally distributed, normalized entries of shape [x_num, y_num]. Normal distribution centered at 0.0, std of 1.0

        :return:
        """
        return tf.Variable(tf.random.normal(shape=(self.x_num, self.y_num), dtype=tf.float32), trainable=True)


class UniformInitializer(Initializer):
    """
    The uniform initializer.
    """
    def __init__(self, x_num, y_num):
        """
        :param x_num: python int: input dimension
        :param y_num: python int: output dimension
        """
        self.x_num = x_num
        self.y_num = y_num

    def initialize_weights(self):
        """
        Uniformly distributed, normalized entries of shape [x_num, y_num].

        :return:
        """
        return tf.Variable(tf.random.uniform(shape=(self.x_num, self.y_num), dtype=tf.float32),
                           trainable=True)

