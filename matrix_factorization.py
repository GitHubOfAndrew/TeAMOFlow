# THIS CONTAINS THE NECESSARY ABSTRACTION FOR THE MATRIX FACTORIZATION MODEL
# NOTE:
# - WE DO NOT HAVE TO CREATE A COMPUTATIONAL GRAPH SEPARATELY, AS THE EAGER EXECUTION
# AND AUTOGRAPH WILL AUTOMATICALLY INCORPORATE ANY TENSORFLOW METHODS INTO THE GRAPH
# - THIS RUNS ONLY ON TENSORFLOW 2.5+, USE AT YOUR OWN RISK

# base libraries
import tensorflow as tf
import numpy as np
import timeit as t

# our libraries
from initializer_graphs import NormalInitializer
from embedding_graphs import LinearEmbedding
from loss_graphs import MSELoss


class MatrixFactorization:
    """
    A class representing the standard matrix factorization model. It admits custom loss functions, embeddings, weight initializers.
    """

    def __init__(self, n_components, user_repr_graph=LinearEmbedding(), item_repr_graph=LinearEmbedding(), loss_graph=MSELoss(), user_weight_graph=NormalInitializer(), item_weight_graph=NormalInitializer()):
        """
        :param n_components: a python int: represents the number of latent features our item and user embeddings share
        :param user_repr_graph: an instance of Embeddings(): the graph that will embed the user features into a space of dimension n_components
        :param item_repr_graph: an instance of Embeddings(): the graph that will embed the item features into a space of dimension n_components
        :param loss_graph: an instance of LossGraph(): the graphs that will be utilized to compute the loss
        :param user_weight_graph: an instance of Initializer(): the graphs that will initialize the weights to be used in user embedding
        :param item_weight_graph: an instance of Initializer(): the graphs that will initialize the weights to be used in item embedding
        """
        self.n_components = n_components
        self.user_repr_graph = user_repr_graph
        self.item_repr_graph = item_repr_graph
        self.loss_graph = loss_graph
        self.user_weight_graph = user_weight_graph
        self.item_weight_graph = item_weight_graph

    def fit(self, epochs, user_features, item_features, tf_interactions, lr=1e-2):
        """
        :param epochs: python int: the number of iterations to perform optimization
        :param user_features: tensorflow tensor: the user features that are available to help perform the predictions
        :param item_features: tensorflow tensor: the item features that are available to help perform predictions
        :param tf_interactions: a sparse tensor: the interaction table
        :param lr: python float: the learning rate used for successive iterations in the optimization algorithm
        :return: nothing (mutates the initialized weights, and we initialized the embeddings)
        """
        # extract feature dimensions
        n_users, n_user_features = user_features.shape
        n_items, n_item_features = item_features.shape

        # initialize weights
        U = self.user_weight_graph.initialize_weights(n_user_features, self.n_components)
        V = self.item_weight_graph.initialize_weights(n_item_features, self.n_components)

        # run training loop
        cumulative_time = 0

        for epoch in range(epochs):
            start = t.default_timer()
            with tf.GradientTape(persistent=True) as tape:
                # build embedding graphs for tracing in gradient
                user_embedding = self.user_repr_graph.get_repr(user_features, U)
                item_embedding = self.item_repr_graph.get_repr(item_features, V)

                # compute loss
                loss_fn = self.loss_graph.get_loss(tf_interactions, user_embedding, item_embedding)

            dloss_dU = tape.gradient(loss_fn, U)
            dloss_dV = tape.gradient(loss_fn, V)

            li_grads = [dloss_dU, dloss_dV]
            li_vars = [U, V]

            tf.keras.optimizers.Adam(learning_rate=lr).apply_gradients(zip(li_grads, li_vars))
            end = t.default_timer()

            loss_one_epoch = tf.reduce_mean(loss_fn)
            cumulative_time += (end - start)

            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1} Complete | Loss {loss_one_epoch} | Runtime {cumulative_time:.5} s')

        # IMPORTANT: compute the embeddings with the most recently updated weights
        self.user_embedding = self.user_repr_graph.get_repr(user_features, U)
        self.item_embedding = self.item_repr_graph.get_repr(item_features, V)

    def predict(self, A=None):
        """
        :param A: a tensorflow tensor: the full interaction table
        :return: tensorflow tensor(s): the predictions for all entries (observed and unobserved), if A is supplied as an argument, then the predictions corresponding to the unobserved entries is also returned
        """
        # we only have dot product prediciton for now
        all_predictions = tf.matmul(self.user_embedding, tf.transpose(self.item_embedding))

        if A is not None:
            tf_predictions = tf.gather_nd(params=all_predictions, indices=tf.where(A==0))
            return all_predictions, tf_predictions
        else:
            return all_predictions

    def predict_ranks(self, A):
        """
        :param A: a tensorflow tensor: the interaction table
        :return: returns the indices of all the top predicted items in descending order of prediction value
        """
        # rank predictions corresponding to uninteracted indices
        predictions, tf_predictions = self.predict(A)

        predicted_item_size = tf.shape(tf_predictions)[0]

        # get indices corresponding to the highest ranked predictions in descending order
        tf_indices_ranks = tf.math.top_k(tf_predictions, k=predicted_item_size)[1]

        return tf_indices_ranks

    def recall_at_k(self, A, k=10):
        """
        :param A: tensorflow tensor: the interaction table
        :param k: a python int: the number of top items we want to see
        :return: python float: the recall @ k which is = # of relevant recommended items in top k / # of relevant items

        NOTE: In our case, 'relevant' is defined as a positive interaction, relevance is highly subjective however, please modify this as fit
        """
        _, tf_predictions = self.predict(A)

        positive_predictions = tf.where(tf_predictions > 0, tf_predictions, 0.0)

        rel_rec_items_k = tf.math.count_nonzero(tf.math.top_k(positive_predictions, k=k).values)

        rel_items = tf.math.count_nonzero(A)

        return (rel_rec_items_k / rel_items).numpy()

    def precision_at_k(self, A, k=10):
        """
        :param A: tensorflow tensor: the interaction table
        :param k: python int: number of top items we want to see
        :return: python float: the precision @ k which is = # of relevant recommended items in top k / # of recommendations

        NOTE: In our case, 'relevant' is defined as a positive interaction
        """
        _, tf_predictions = self.predict(A)

        positive_predictions = tf.where(tf_predictions > 0, tf_predictions, 0.0)

        rel_rec_items_k = tf.math.count_nonzero(tf.math.top_k(positive_predictions, k=k).values)

        rec_items = tf.size(tf_predictions).numpy()

        return (rel_rec_items_k / rec_items).numpy()

    def f1_at_k(self, A, k=10, beta=1.0):

        prec, rec = self.precision_at_k(A, k=k), self.recall_at_k(A, k=k)

        return ((1 + beta**2) * prec * rec) / (beta**2 * (prec + rec))

