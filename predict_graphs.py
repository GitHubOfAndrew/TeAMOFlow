# THIS CONTAINS THE NECESSARY ABSTRACTIONS FOR THE SCORING SYSTEM IN TENSORFLOW

import tensorflow as tf
from abc import *

class PredictionGraph(ABC):
    """
    This is an abstract base class to uniformize the serving of the prediction layer in our framework. There are many different ways to generate an output from a matrix factorization model.
    """

    @abstractmethod
    def predict(self, U, V):
        """
        Arguments:\n
        - U: a tensorflow tensor, the user embeddings
        - V: a tensorflow tensor, the item embeddings

        Purpose:\n
        - To generate a prediction
        """
        pass