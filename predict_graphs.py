# THIS CONTAINS THE NECESSARY ABSTRACTIONS FOR THE SCORING SYSTEM IN TENSORFLOW

import tensorflow as tf
from abc import *

class PredictionGraphs(ABC):
    """
    An abstract base class representing the prediction graphs. These will be objects that compute user-item relationships.
    """
    pass