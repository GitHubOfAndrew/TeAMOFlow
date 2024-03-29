# THIS CONTAINS THE NECESSARY ABSTRACTION FOR THE MATRIX FACTORIZATION MODEL
# NOTE:
# - WE DO NOT HAVE TO CREATE A COMPUTATIONAL GRAPH SEPARATELY, AS THE EAGER EXECUTION
# AND AUTOGRAPH WILL AUTOMATICALLY INCORPORATE ANY TENSORFLOW METHODS INTO THE GRAPH
# - THIS RUNS ONLY ON TENSORFLOW 2.5+, USE AT YOUR OWN RISK
# This is based on a library of James Kirk's, called TensorRec (https://github.com/jfkirk/tensorrec/). Consider this a spiritual successor to TensorRec written in tensorflow 2.x.


# base libraries
import tensorflow as tf
import numpy as np
import timeit as t

# our libraries: other objects for this model
from .initializer_graphs import NormalInitializer
from .embedding_graphs import *
from .loss_graphs import *

# our libraries: utility/helper functions
from .utils import gather_matrix_indices, random_sampler


class MatrixFactorization:
    """
    A class representing the standard matrix factorization model. It admits custom loss functions, embeddings, weight initializers.
    """

    def __init__(self, n_components, user_repr_graph=LinearEmbedding(), item_repr_graph=LinearEmbedding(), loss_graph=MSELoss(), user_weight_graph=NormalInitializer(), item_weight_graph=NormalInitializer(), n_users=None, n_items=None, n_samples=None, generate_sample=False):
        """
        By default, the model will be initialized as a regression-based model using the mean-squared error.

        In general, our models will be of two types:\n
        Regression-based RATING models (i.e. MSE Loss)\n
        Sample-based RANKING models (i.e. WMRB Loss)

        - Regression-based rating models will train to predict the observed interactions, and in the process, obtains user/item embeddings that will guess the correct unobserved interactions. In this way, the model will rate every item per user, and we will take the highest rated items as our recommendations. This configuration is recommended for smaller interaction datasets (~10^5 interactions) as the results are easily interpretable and training runtime is instant.

        - Sample-based ranking models will train to prioritize the highest rated items over lower rated items (i.e. rank the items). This type of model prioritizes the order of items that the user would like more over others, training user/item embeddings to reflect this in the process. This model is recommended for larger datasets (~10^5 + interactions) as it learns to rank items with some idea of "priority" rather than hoping for an accurate rating.

        :param n_components: a python int: represents the number of latent features our item and user embeddings share
        :param user_repr_graph: an instance of Embeddings(): the graph that will embed the user features into a space of dimension n_components
        :param item_repr_graph: an instance of Embeddings(): the graph that will embed the item features into a space of dimension n_components
        :param loss_graph: an instance of LossGraph(): the graphs that will be utilized to compute the loss
        :param user_weight_graph: an instance of Initializer(): the graphs that will initialize the weights to be used in user embedding
        :param item_weight_graph: an instance of Initializer(): the graphs that will initialize the weights to be used in item embedding
        :param n_users: python int: the number of users, NOTE: this is used to generate samples for the WMRB loss
        :param n_items: python int: the number of items, NOTE: this is used to generate samples for the WMRB loss
        :param n_samples: python in: the number of samples, NOTE: this is used to generate samples for the WMRB loss; n_samples <= n_items (this is a must or it will throw an error)
        :param generate_sample: python boolean, NOTE: this will indicate whether to generate a random sample or not
        """

        # initialize graphs for training and evaluation
        self.n_components = n_components
        self.user_repr_graph = user_repr_graph
        self.item_repr_graph = item_repr_graph
        self.loss_graph = loss_graph
        self.user_weight_graph = user_weight_graph
        self.item_weight_graph = item_weight_graph

        # check if the model is a sampled-based model
        self.n_users = n_users
        self.n_items = n_items
        self.n_samples = n_samples
        self.random_ind = None
        self.generate_sample = generate_sample

        # if n_items is given, but n_samples is not specified, take half of the total number of items
        if n_samples is None and n_items is not None:
            self.n_samples = n_items // 2

        # do random sampling without replacement if generate_sample flag is true
        if generate_sample == True:
            self.random_ind = random_sampler(n_items, n_users, self.n_samples)

        # check if embedding graphs are instances of ReLU Embedding as we need to specify the relu dimensions
        if isinstance(self.user_repr_graph, ReLUEmbedding):
            self.user_aux_dim = 5 * self.n_components
        if isinstance(self.item_repr_graph, ReLUEmbedding):
            self.item_aux_dim = 5 * self.n_components

        # initialize all parameters necessary for ReLU Embedding
        self.user_relu_bias = None
        self.user_relu_weight = None

        self.item_relu_bias = None
        self.item_relu_weight = None

        # initialize all parameters necessary for BiasedLinearEmbedding
        self.user_linear_bias = None
        self.item_linear_bias = None

        # collect all trainable variables as an attribute
        self.user_trainable = None
        self.item_trainable = None

    def fit(self, epochs, user_features, item_features, tf_interactions, lr=1e-2):
        """
        NOTE: There are two types of loss functions to consider with our model:
        1) regression-based models
        2) sample-based models

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
        # check what instance the embedding graph is
        if not isinstance(self.user_repr_graph, ReLUEmbedding):
            U = self.user_weight_graph.initialize_weights(n_user_features, self.n_components)
        else:
            U = self.user_weight_graph.initialize_weights(self.user_aux_dim, self.n_components)

        if not isinstance(self.item_repr_graph, ReLUEmbedding):
            V = self.item_weight_graph.initialize_weights(n_item_features, self.n_components)
        else:
            V = self.item_weight_graph.initialize_weights(self.item_aux_dim, self.n_components)

        # run training loop
        cumulative_time = 0

        for epoch in range(epochs):
            start = t.default_timer()
            with tf.GradientTape(persistent=True) as tape:
                # build embedding graphs for tracing in gradient
                user_embedding, self.user_trainable = self.user_repr_graph.get_repr(features=user_features, weights=U, relu_weight=self.user_relu_weight, relu_bias=self.user_relu_bias, linear_bias=self.user_linear_bias)
                item_embedding, self.item_trainable = self.item_repr_graph.get_repr(features=item_features, weights=V, relu_weight=self.item_relu_weight, relu_bias=self.item_relu_bias, linear_bias=self.item_linear_bias)

                # save only certain weights depending on embedding graph
                if isinstance(self.user_repr_graph, BiasedLinearEmbedding):
                    _, self.user_linear_bias = self.user_trainable

                if isinstance(self.user_repr_graph, ReLUEmbedding):
                    _, self.user_relu_weight, self.user_relu_bias = self.user_trainable

                if isinstance(self.item_repr_graph, BiasedLinearEmbedding):
                    _, self.item_linear_bias = self.item_trainable

                if isinstance(self.item_repr_graph, ReLUEmbedding):
                    _, self.item_relu_weight, self.item_relu_bias = self.item_trainable

                # generate predictions to trace the embeddings in autograph
                predictions = tf.matmul(user_embedding, tf.transpose(item_embedding))

                # check which loss the model corresponds to
                if isinstance(self.loss_graph, WMRBLoss):
                    tf_sample_predictions = gather_matrix_indices(predictions, self.random_ind)
                    tf_prediction_serial = tf.gather_nd(params=predictions, indices=tf_interactions.indices)
                    predictions = None
                if isinstance(self.loss_graph, MSELoss):
                    tf_sample_predictions = None
                    tf_prediction_serial = None
                if isinstance(self.loss_graph, KLDivergenceLoss):
                    tf_prediction_serial = tf.gather_nd(params=predictions, indices=tf_interactions.indices)
                    tf_sample_predictions = None
                    predictions = None

                # compute loss (if sample_based == True, use WMRB loss)
                loss_fn = self.loss_graph.get_loss(tf_interactions=tf_interactions, tf_sample_predictions=tf_sample_predictions,
                                                   tf_prediction_serial=tf_prediction_serial, predictions=predictions,
                                                   n_items=self.n_items, n_samples=self.n_samples)

            # perform optimization on ALL trainable weights
            dloss_dusers = tape.gradient(loss_fn, self.user_trainable)
            dloss_ditems = tape.gradient(loss_fn, self.item_trainable)

            li_grads = dloss_dusers + dloss_ditems
            li_vars = self.user_trainable + self.item_trainable

            tf.keras.optimizers.Adam(learning_rate=lr).apply_gradients(zip(li_grads, li_vars))
            end = t.default_timer()

            loss_one_epoch = tf.reduce_mean(loss_fn)
            cumulative_time += (end - start)

            if (epoch + 1) % 25 == 0:
                print(f'Epoch {epoch + 1} Complete | Loss {loss_one_epoch} | Runtime {cumulative_time:.5} s')

        # IMPORTANT: compute the embeddings with the most recently updated weights, return the most updated weights
        self.user_embedding, self.user_trainable = self.user_repr_graph.get_repr(features=user_features, weights=U, relu_weight=self.user_relu_weight, relu_bias=self.user_relu_bias, linear_bias=self.user_linear_bias)
        self.item_embedding, self.item_trainable = self.item_repr_graph.get_repr(features=item_features, weights=V, relu_weight=self.item_relu_weight, relu_bias=self.item_relu_bias, linear_bias=self.item_linear_bias)

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

    def recall_at_k(self, A, k=10, preserve_rows=False):
        """
        The recall at k here is computed according to this article: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Recall

        Furthermore, this recall_at_k is consistent with the definition of recall at k in the LightFM documentation as well: https://making.lyst.com/lightfm/docs/_modules/lightfm/evaluation.html#precision_at_k

        Specifically, the recall at k refers to: for a user, the number of relevant recommended items in top k predictions / number of relevant items in top k

        This function computes the recall @ k for each user. To find the total recall at k, just take the mean.

        :param A: tensorflow tensor: the interaction table
        :param k: a python int: the number of top items we want to see
        :param preserve_rows: a python boolean: if false, then take out the users who have 0 interactions from the computation, otherwise, include all users
        :return: tensorflow tensor: the recall @ k per user

        NOTE: In our case, 'relevant' is defined as a positive interaction, relevance is highly subjective however, please modify this as fit
        """
        # generate predictions and get positive predictions
        predictions = self.predict()
        positive_predictions = tf.where(predictions > 0.0, predictions, 0.0)

        # get the known positives
        known_positives = tf.where(A > 0.0, A, 0.0)

        num_users, _ = positive_predictions.shape

        # find the indices (items) corresponding to top k items per user
        top_k_items_user = tf.cast(tf.math.top_k(positive_predictions, k=k).indices, dtype=tf.int64)

        # gather the entries of the interaction (A) according to the top k items per user (top_k_items_user)
        res_top_k = gather_matrix_indices(A, top_k_items_user)

        # count the number of known positive rated items per user
        relevant = tf.math.count_nonzero(known_positives, axis=1, dtype=tf.float32)

        # count the number of known positive items in top k predictions, per user
        hits = tf.math.count_nonzero(res_top_k, axis=1, dtype=tf.float32)

        # if preserve_rows is true, then we will set the recall of users with no interactions to 0
        # otherwise, we take out any users with no interactions
        if not preserve_rows:
            zero_interaction_mask = tf.math.not_equal(relevant, 0.0)
            masked_hits = tf.boolean_mask(hits, zero_interaction_mask)
            masked_relevant = tf.boolean_mask(relevant, zero_interaction_mask)
            return masked_hits / masked_relevant
        else:
            # return number of known positive items in top k / number of positive items, per user
            recall = hits / relevant

            # check for null recall values and set it to 0
            nan_mask = tf.math.is_nan(recall)
            return tf.where(nan_mask == False, recall, 0.0)

    def precision_at_k(self, A, k=10, preserve_rows=False):
        """
        Computes precision at k, this is the number of predictions in the top k that are relevant. This is different from the recall at k, which is the number of known positives in the top k predictions.

        Specifically, precision at k = relevant recommended items in top k / k

        :param A: tensorflow tensor: the interaction table
        :param k: python int: number of top items we want to see
        :param preserve_rows: a python boolean: if false, then take out the users who have 0 interactions from the computation, otherwise, include all users
        :return: python float: the precision @ k which is = # of relevant recommended items in top k / # of recommendations

        NOTE: In our case, 'relevant' is defined as a positive interaction
        """
        # functionality identical to recall_at_k...
        predictions = self.predict()
        positive_predictions = tf.where(predictions > 0.0, predictions, 0.0)

        num_users, _ = positive_predictions.shape

        top_k_items_user = tf.cast(tf.math.top_k(positive_predictions, k=k).indices, dtype=tf.int64)

        res_top_k = gather_matrix_indices(A, top_k_items_user)

        hits = tf.math.count_nonzero(res_top_k, axis=1, dtype=tf.float32)

        # if preserve_rows is false, we take out any users with no interactions, otherwise, include all users
        if not preserve_rows:
            relevant = tf.math.count_nonzero(tf.where(A > 0.0, A, 0.0), axis=1, dtype=tf.float32)
            zero_int_mask = tf.math.not_equal(relevant, 0.0)
            masked_hits = tf.boolean_mask(hits, zero_int_mask)
            return masked_hits / k
        # ... except that we do number of known positives in top k / k, per user
        else:
            return hits / k

    def f1_at_k(self, A, k=10, beta=1.0):
        """
        :param A: a tensorflow tensor: the interaction table
        :param k: a python int: the number of top predictions to use in judging
        :param beta: a python float: a weighting parameter that influences the contribution of precision vs. recall
        :return: a python float: the f1 score @ k, a harmonic mean of precision @ k and recall @ k metrics
        """

        precision, recall = self.precision_at_k(A, k=k), self.recall_at_k(A, k=k)

        prec, rec = tf.reduce_mean(precision), tf.reduce_mean(recall)

        return ((1 + beta**2) * prec * rec) / (beta**2 * (prec + rec))

    def dcg_at_k(self, dense_interactions, k=10):
        """
        Compute the Discounted Cumulative Gain (DCG) for top k predictions.

        The DCG score considers the order of the items retrieved in a query (i.e. recommended items for a user).

        A relevant (in the interaction table) item that is retrieved earlier contributes more to the DCG score. This metric judges the effectiveness of the recommender/retrieval system in not only retrieving relevant recommendations, but doing so in the most relevant ORDER as well.

        :param dense_interactions: tensorflow tensor: a dense (ordinary) tensor containing the interactions
        :param k: python int: the amount of items retrieved
        :return: tensorflow tensor: contains the dcg score per user for the top k retrieved items
        """
        predictions = self.predict()
        user_num, item_num = predictions.shape

        # get the indices of the top k indices
        ranks_user = tf.cast(tf.math.top_k(predictions, k=item_num).indices, dtype=tf.int64)

        # NOTE: We use the modified definition of dcg score to compute this, this version gives more weight to relevant items
        numerator = tf.pow(2.0, gather_matrix_indices(dense_interactions, ranks_user)) - 1.0

        # create tensor with order number of each interaction in the numerator
        denominator_arg = tf.transpose(
            tf.repeat(tf.range(1, item_num + 1, dtype=tf.float32)[:, tf.newaxis], user_num, axis=1))

        # convert log of the order to base 2
        denominator = tf.math.log1p(denominator_arg) / tf.math.log(2.0)

        summation_term = numerator / denominator

        # return the dcg score of the top k predictions
        return tf.reduce_sum(summation_term[:, :k], axis=1)

    def idcg_at_k(self, dense_interactions, k=10):
        """
        Computes the Ideal Discounted Cumulative Gain (IDCG) for top k predictions.

        The IDCG score is the DCG score for a theoretically perfect recommendation/retrieval (i.e. if the top k retrieved items for a user corresponded to the relevant items in the original interaction).

        :param dense_interactions: a tensorflow tensor: a dense (ordinary) tensor containing the interactions
        :param k: python int: the amount of items retrieved
        :return: tensorflow tensor: contains the idcg score per user for top k items retrieved
        """
        predictions = self.predict()
        user_num, item_num = predictions.shape

        # get the indices of the top k indices
        ranks_user = tf.cast(tf.math.top_k(predictions, k=item_num).indices, dtype=tf.int64)

        # NOTE: We use the modified definition of dcg score to compute this, this version gives more weight to relevant items
        numerator = tf.pow(2.0, gather_matrix_indices(dense_interactions, ranks_user)) - 1.0

        # sort the numerator terms to get the ideal dcg
        ideal_int = tf.math.top_k(numerator, k=item_num).values

        # create tensor with order number of each interaction in the numerator
        denominator_arg = tf.transpose(
            tf.repeat(tf.range(1, item_num + 1, dtype=tf.float32)[:, tf.newaxis], user_num, axis=1))

        # convert log of the order to base 2
        denominator = tf.math.log1p(denominator_arg) / tf.math.log(2.0)

        ideal_summation_term = ideal_int / denominator

        return tf.reduce_sum(ideal_summation_term[:, :k], axis=1)

    def ndcg_at_k(self, A, k=10, preserve_rows=False):
        """
        Computes the NDCG score (Normalized Discounted Cumulative Gain).

        Computes the success of the model's retrieval by judging if it retrieved relevant items with most priority.

        :param A: tensorflow tensor: a dense interaction table
        :param k: python int: how many top items we will retrieve
        :param preserve_rows: python boolean: flag indicating whether to preserve the users with no interactions
        :return:
        """
        # compute dcg
        dcg = self.dcg_at_k(A, k)

        # compute idcg
        idcg = self.idcg_at_k(A, k)

        # compute ndcg
        ndcg = dcg / idcg

        if not preserve_rows:
            # we throw away all rows with no interactions in the computation
            nonzero_ints = tf.math.count_nonzero(A, axis=1)
            mask = nonzero_ints > 0
            return tf.boolean_mask(ndcg, mask)
        else:
            # ensure that all the metrics for rows with no interactions are 0 so that we don't get overflow error
            return tf.where(~tf.math.is_nan(ndcg), ndcg, 0.0)


    def retrieve_user_recs(self, user=None, k=None):
        """
        Method to retrieve the item recommendations for a certain user. We will be retrieving item ranks, not prediction values.

        :param user: python int: a row index representing the user
        :param k: python int: specifies how many top items to bring
        :return: a numpy array containing the indices to the top k rated items
        """
        all_predictions = self.predict()
        num_users, num_items = all_predictions.shape

        # if user is not specified, but rank is specified, return top k item rankings for all users
        if user is None and k is not None:
            return tf.math.top_k(all_predictions, k=k).indices.numpy()
        # if user is specified, but k is specified, return all rankings for specified user
        if user is not None and k is None:
            return tf.math.top_k(all_predictions[user], k=num_items).indices.numpy()
        # if user is specified and k is specified, return top k rankings for specific user
        if user is not None and k is not None:
            return tf.math.top_k(all_predictions[user], k=k).indices.numpy()
        # if user is not specified and k is not specified, return all item rankings for all users
        if user is None and k is None:
            return tf.math.top_k(all_predictions, k=num_items).indices.numpy()

    def save_model(self):
        """
        Method to save the model. When serving a model, we prioritize two things:

        - Having the exact training setup (All the necessary graphs from a previous training run)
        - Having the results of the training (in this case, it's our learned embeddings)

        These two are sufficient to re-initialize models and train models again.
        
        :return: python dictionaries: a configuration file containing all the necessary attributes about this model, and a file with the trained embeddings, so that we can easily make predictions post-training
        """

        # we will have a dictionary of graph configurations so that we can reproduce training
        dict_config = {'Latent Dimension': self.n_components, 'User Embedding': self.user_repr_graph,
                       'Item Embedding': self.item_repr_graph, 'Loss': self.loss_graph,
                       'User Initialization': self.user_weight_graph, 'Item Initialization': self.item_weight_graph,
                       'Number of Users': self.n_users, 'Number of Items': self.n_items,
                       'Number of Samples': self.n_samples, 'Generate Sample': self.generate_sample}

        # will have another dictionary, containing results of the training run, so that we may make predictions and do other things easily
        dict_results = {'User Embedding': self.user_embedding, 'Item Embedding': self.item_embedding, 'User Variables': self.user_trainable, 'Item Variables': self.item_trainable}

        return dict_config, dict_results


    @classmethod
    def from_saved(cls, config):
        """
        The purpose of this is to initialize a model again using the dict_config from our save_model() method. Furthermore, we could just create a dictionary with all of the necessary components to initialize the model and input it into here as well.

        :param config: python dictionary: holds the arguments to initialize the model as desired
        :return: a newly initialized, untrained, MatrixFactorization() object
        """

        # unpack the configuration dictionary and put it into our class to initialize our new model
        return cls(**config)
