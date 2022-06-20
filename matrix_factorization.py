# THIS CONTAINS THE NECESSARY ABSTRACTION FOR THE MATRIX FACTORIZATION MODEL
# NOTE:
# - WE DO NOT HAVE TO CREATE A COMPUTATIONAL GRAPH SEPARATELY, AS THE EAGER EXECUTION
# AND AUTOGRAPH WILL AUTOMATICALLY INCORPORATE ANY TENSORFLOW METHODS INTO THE GRAPH
# - THIS RUNS ONLY ON TENSORFLOW 2.5+, USE AT YOUR OWN RISK

import tensorflow as tf
import numpy as np
import timeit as t
from loss_graphs import MSE, WMRB_1


class MatrixFactorization:
    """Class for a Matrix Factorization Model. A hybrid (content-based and collaborative filtering) recommender model
    that takes user embeddings and item embeddings to predict unobserved interactions in a given interaction table"""

    def __init__(self, num_features, shape, U=None, V=None):

        """
        Arguments:\n
        - num_features: a python int, number of latent features for the user and item embeddings\n
        - shape: a python tuple of the form (n,m) where (n,m) is the shape of the interaction table we are approximating\n
        - U: a tensorflow tensor, the initialized user embeddings\n
        - V: a tensorflow tensor, the initialized item embeddings\n
        --- NOTE: If user and item embeddings are not supplied (i.e. U=None and V=None), then they will be initialized using random uniform values
        --- otherwise, we can input our own choice of initializations for each embedding that we see fit

        Purpose:
        Initializes the following instance attributes:\n
        - user embeddings: U ---> of shape (n, num_features)\n
        - item embeddings: V ---> of shape (m, num_features)
        """

        n, m = shape
        user_shape = (n, num_features)
        item_shape = (m, num_features)

        # initialize user and item embeddings if they are not specified as inputs, otherwise, set it to input
        if U is None:
            self.U = tf.Variable(tf.random.uniform(shape=user_shape), trainable=True, dtype=tf.float32)
        else:
            self.U = U

        if V is None:
            self.V = tf.Variable(tf.random.uniform(shape=item_shape), trainable=True, dtype=tf.float32)
        else:
            self.V = V

        # save inputs for later use when saving model
        self.num_features, self.shape = num_features, shape

    # @tf.function
    # def loss(self, A, U, V, lambda_1, lambda_2):
    #     """
    #     Arguments:\n
    #     - A: an array with 2 axes, an interaction table between users and items\n
    #     - U: user embeddings\n
    #     - V: item embeddings\n
    #     - lambda_1: weighting hyperparameters that controls contribution of observed interactions\n
    #     - lambda_2: weighting hyperparameters that controls contribution of unobserved interactions
    #
    #     Purpose:\n
    #     Outputs the loss for both observed and unobserved interactions. In particular, it creates a loss graph for computation
    #     in tensorflow's autograd
    #
    #     2022-06-03 - The only loss available is the MSE loss, this will be changed as necessary as we discover more losses to use
    #     """
    #
    #     # mean squared error loss with regularization
    #     return tf.where(A != 0, tf.multiply(lambda_1, tf.pow(A - tf.matmul(U, tf.transpose(V)), 2)),
    #                     tf.multiply(lambda_2, tf.pow(-tf.matmul(U, tf.transpose(V)), 2)))
    #
    @tf.function
    def loss(self, A, lambda_1=None, lambda_2=None, n_samples=None, loss_graph='mse'):
        """
        Arguments:\n
        - loss_graph: a TeAMOFlow loss_graph object, this is a computational graph representing our loss function; please look in loss_graphs.py to see more info
        - *args: a python iterable, contains custom inputs for certain loss functions
        --- MSE: *args = lambda_1, lambda_2 (I recommend 0.01 and 0.001, respectively)
        --- WMRB_1: *args = n_samples

        Purpose:\n
        - A loss function that will be incorporated into our computational graph in our training loop
        - Returns an iterable containing the loss graph and all of its inputs
        """

        if loss_graph == 'mse':
            return MSE().invoke_loss_graph(A, self.U, self.V, lambda_1=lambda_1, lambda_2=lambda_2, n_samples=n_samples)

        if loss_graph == 'wmrb_1':
            return WMRB_1().invoke_loss_graph(A, self.U, self.V, lambda_1=lambda_1, lambda_2=lambda_2, n_samples=n_samples)

    def fit(self, A, epochs, optimizer, lambda_1=None, lambda_2=None, n_samples=None, loss_graph='mse', verbose=1):
        """
        Arguments:\n
        - A: an array with 2 axes, an interaction table between users and items\n
        - epochs: a python int, the number of times the model will train\n
        - optimizer: a tf.keras.optimizers object, the optimization algorithm used in the minimization of the loss\n
        - *args: a python iterable, custom inputs for corresponding loss graphs; look at docstring for .loss method
        - verbose: a python int,\n
        --- 0: no information about training process printed\n
        --- 1: epoch number, loss, cumulative training runtime printed\n
        --- 2: epoch number printed

        Purpose:\n
        Runs a training loop for a specified loss function using an optimization algorithm of choice.
        """

        # get history of losses and training parameters
        train_history, li_loss = {}, []

        # # recast hyperparameters as floats so we get no errors on ints
        # lambda_1, lambda_2 = float(lambda_1), float(lambda_2)
        #
        # # recast hyperparameters as tf.constant objects so it can be incorporated into graph
        # lambda_1, lambda_2 = tf.constant(lambda_1), tf.constant(lambda_2)

        total_time = 0
        for epoch in range(epochs):

            # start timer
            start = t.default_timer()

            with tf.GradientTape(persistent=True) as tape:

                # initialize loss to take gradients
                loss_fn = self.loss(A, lambda_1, lambda_2, n_samples, loss_graph=loss_graph)
                # loss_fn = self.loss(A, self.U, self.V, lambda_1, lambda_2)

            # backpropagate to compute gradient
            dloss_dU, dloss_dV = tape.gradient(loss_fn, self.U), tape.gradient(loss_fn, self.V)

            # get arrays of variables and gradients
            li_grads = [dloss_dU, dloss_dV]
            li_vars = [self.U, self.V]

            # now do the optimization
            optimizer.apply_gradients(zip(li_grads, li_vars))

            # compute the total mean loss and end timer
            epoch_loss = loss_fn.numpy()
            end = t.default_timer()

            # add a break in the training loop for when the loss is too low, we want to combat overfitting
            if epoch_loss <= 1e-12:
                break

            # put in key for train history, get loss from every epoch
            li_loss.append(epoch_loss)
            train_history['Loss'] = li_loss

            # verbose outputs
            if verbose == 0:
                pass

            if verbose == 1:
                total_time += (end - start)

                # print the updates for every epoch
                if (epoch + 1) % 50 == 0:
                    print(
                        f'Epoch {epoch + 1}/{epochs} | Loss {epoch_loss:.6f} | Total Runtime {total_time:.4f} sec.')

            if verbose == 2:
                total_time += (end - start)

                # print the updates for every epoch
                if (epoch + 1) % 50 == 0:
                    print(f'Epoch {epoch + 1}/{epochs} | Total Runtime {total_time:.4f} sec.')

        return train_history

    def predict(self, A=None):
        """
        Arguments:\n
        - A: a tensorflow tensor, the original interaction table; by default it is None
        --- If not None: We will clip any values that go beyond the minimum or maximum entries in the original interaction table
        --- If None: We will directly output the model's predicted interactions based on its embeddings

        Purpose:\n
        - Infer the prediction for the interaction table
        """

        predicted_interaction = tf.matmul(self.U, tf.transpose(self.V))

        if A is not None:

            # perform the rounding and clipping if an interaction table is supplied
            min_val, max_val = np.min(A.numpy()), np.max(A.numpy())
            pred_int = tf.round(predicted_interaction)
            pred_int = tf.where(pred_int < min_val, min_val, pred_int)
            pred_int = tf.where(pred_int > max_val, max_val, pred_int)
        else:

            # just output the direct result of the embeddings if no interaction table is supplied
            # NOTE: THIS IS THE RECOMMENDED METHOD OF OUTPUT, WE ARE GOING TO JUDGE THE MODEL'S PERFORMANCE USING PRECISION AND RECALL @ K
            pred_int = predicted_interaction

        return pred_int

    def score(self, thresh, k, A):
        """
        Arguments:\n
        - thresh: python int, threshold of score that we want to set for relevancy of item
        - k: python int,  the top k scores that we will take from our predictions
        - A: tensorflow tensor, the original interaction table with observed interactions

        Purpose:\n
        - score the model's performance according to the recall @ k and by indicating relevance, returns as (precision, recall, f1 score)
        NOTE: for now, relevance will be judged by the model's ability to exceed a certain threshold

        Methodology:\n
        - We will score by recall @ k which we interpret as follows:
        --- fix k to specify the top k items to look at, according to a supplied threshhold, thresh, we pick out two items:\n
        ----- relevant items: all interactions (nonzero entries) that are >= thresh\n
        ----- irrelevant items: all interactions (nonzero entries) that are < thresh

        --- then we score the predicted recommendations according to the following:\n
        ----- predicted rating >= thresh ---> recommend\n\n
        ----- predicted rating < thresh ---> don't recommend

        --- Precision @ k: the proportion of recommendations that are in top-k that are relevant (i.e. # of relevant recommended items at k / # of recommended items)\n
        --- Recall @ k: the proportion of relevant items found in top k recs (i.e. # of relevant recommended items at k / # of relevant items)
        """

        # step 1: compute relevant recommended items @ k

        ### get predicted interactions
        predicted_interactions = self.predict()

        # step 2: compute all necessary parameters for recall and precision

        # collect all recommendations (predictions) made by the model
        recommendations = tf.where(A == 0, predicted_interactions, 0.0)

        # get number of recommended (predicted) items
        num_rec_items = tf.math.count_nonzero(recommendations).numpy()

        # collect all relevant (meets threshold) recommendations (predictions)
        rel_recs = tf.where(recommendations >= thresh, recommendations, 0.0)

        # get the number of top-k recommendations (predictions) that are relevant (meet the threshold)
        rel_recs_flat = tf.reshape(rel_recs, shape=[-1])

        num_rel_recs_top_k = tf.size(tf.math.top_k(rel_recs_flat, k=k).values).numpy()

        # get the number of relevant items in all interactions
        num_rel_items = tf.math.count_nonzero(tf.where(A >= thresh, 1.0, 0.0)).numpy()

        # step 3: compute precision at k, recall at k
        precision_k, recall_k = num_rel_recs_top_k / num_rec_items, num_rel_recs_top_k / num_rel_items

        ### compute the f1 score at k as well
        f1_at_k = (2.0 * precision_k * recall_k) / (precision_k + recall_k)

        return precision_k, recall_k, f1_at_k

    def save_model(self):
        """
        Arguments:\n
        - Nothing

        Purpose:\n
        - Save the configuration of our model needed to generate predictions\n
        - Have the configuration ready to throw back into our model if necessary\n
        - Make the trained model easy to serve
        """

        # dictionary compatible with model
        model_configuration = {'num_features': self.num_features, 'shape': self.shape,
                               'U': self.U, 'V': self.V}

        return model_configuration

    @classmethod
    def from_saved(cls, config):

        """
        Arguments:\n
        - cls: class involved in class method\n
        - config: a python dictionary containing the trained embeddings and input parameters used for a previous training run

        Purpose:\n
        - initialize a new model (instance of this class) with the new configuration

        Note:\n
        This is a class method so be sure to call directly with the base class
        """

        # unpack the configuration dictionary and put it into our class to initialize our new model
        return cls(**config)



