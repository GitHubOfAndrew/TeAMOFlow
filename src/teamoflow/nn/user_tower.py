# FILE CONTAINING THE NEURAL NETWORK CONFIGURATIONS FOR THE QUERY (USER) FEATURES

import tensorflow as tf
import timeit as t
from .utils import *
from .layers import *
from .loss_graphs import *
from .initializer_graphs import *


class QTSoftmax:

    def __init__(self, n_features, li_units, li_activations, loss_graph=CrossEntropy()):
        """
        The Query Tower Softmax model will take in the number of user features and a list of all hidden layer units + output.

        We must also supply a list of activation functions (this is purposefully left vague as it is meant to be customizable).

        :param n_features: python int: number of features in input
        :param li_units: python list: list containing number of units in each layer
        :param li_activations: python list: list containing the activation functions in each layer
        """

        self.loss_graph = loss_graph

        self.activations = li_activations

        self.weights = []

        li_first_comp = [n_features] + li_units[:-1]

        # build the weights using the initializer graphs and the layer template
        for x, y in zip(li_first_comp, li_units):
            self.weights.append(tfLayerWeight(x, y).return_value())


    def predict(self, X):
        """
        Feedforward mechanism. For inferencing.

        :param X: tensorflow tensor: the data to feed in
        :return:
        """
        # instantiate lambda function to apply activation to weights
        lam_func = lambda x, y: x(y)

        # apply all the layers successively
        x_in = X
        for weights, activation_function in zip(self.weights, self.activations):
            res = tf.matmul(x_in, lam_func(activation_function, weights))
            x_in = res
        logit = x_in

        # return both softmax probabilities and logit
        return tf.nn.softmax(logit, axis=1), logit

    def fit(self, x_train, y_train, epochs=150, lr=1e-3):
        """
        Fit the model.

        :param x_train: tensorflow tensor: the training data tensor
        :param y_train: tensorflow tensor: the training label tensor (these are the interactions)
        :param epochs: python int: the number of times the model will see the training data during fitting
        :param lr: python float: the learning rate during optimization
        :return:
        """
        # WE WILL DO BATCH GRADIENT DESCENT
        dict_losses = {}
        li_losses = []
        # CONVERT TRAIN LABELS TO ONE-HOT ENCODED
        y_train_ohe = tf.where(y_train != 0.0, 1.0, 0.0)

        cumulative_time = 0.0
        for epoch in range(epochs):
            start = t.default_timer()

            with tf.GradientTape() as tape:
                y_hat, _ = self.predict(x_train)
                loss = self.loss_graph.get_loss(y_train_ohe, y_hat)

            # COMPUTE GRADIENTS
            dl_dw = tape.gradient(loss, self.weights)
            # OPTIMIZATION ROUTINE
            tf.keras.optimizers.Adam(learning_rate=lr).apply_gradients(zip(dl_dw, self.weights))

            li_losses.append(tf.reduce_mean(loss).numpy())
            end = t.default_timer()

            cumulative_time += (end - start)

            # GIVE OUTPUT DURING FITTING
            if (epoch + 1) % 25 == 0:
                print(f'Epoch {epoch + 1} | Loss {tf.reduce_mean(loss).numpy()} | Runtime {cumulative_time} sec.')

        dict_losses['Training Loss'] = li_losses
        return dict_losses

    def recall_at_k(self, user_features, A, k=10, preserve_rows=False):
        """
        The proportion of ground truths that align with top k predictions.

        :param user_features: tensorflow tensor: this is the input for inferencing
        :param A: tensorflow tensor: interaction table (also our "y" values)
        :param k: python int: how many predictions we will look at
        :param preserve_rows: python boolean: flag that indicates if we should drop users with no interactions or not
        :return:
        """

        # THE LOGITS FROM OUR MODEL ARE OUR "REAL" PREDICTIONS
        _, predictions = self.predict(user_features)

        # GET KNOWN POSITIVES
        interaction = tf.where(A > 0.0, A, 0.0)

        top_k_indices = tf.math.top_k(predictions, k=k).indices

        relevant_retrieved = gather_matrix_indices(interaction, tf.cast(top_k_indices, dtype=tf.int64))

        hits = tf.math.count_nonzero(relevant_retrieved, axis=1, dtype=tf.float32)

        relevant_per_user = tf.math.count_nonzero(interaction, axis=1, dtype=tf.float32)

        if not preserve_rows:
            zero_interaction_mask = tf.math.not_equal(relevant_per_user, 0.0)
            masked_hits = tf.boolean_mask(hits, zero_interaction_mask)
            masked_relevant = tf.boolean_mask(relevant_per_user, zero_interaction_mask)
            return masked_hits / masked_relevant
        else:
            recall = hits / relevant_per_user
            nan_mask = tf.math.is_nan(recall)
            return tf.where(nan_mask == False, recall, 0.0)

    def precision_at_k(self, user_features, A, k=10, preserve_rows=False):
        """
        The proportion of predictions that align with the ground truth, for top k predictions.

        :param user_features: tensorflow tensor: this is the input for inferencing
        :param A: tensorflow tensor: interaction table (also our "y" values)
        :param k: python int: how many predictions we will look at
        :param preserve_rows: python boolean: flag that indicates if we should drop users with no interactions or not
        :return:
        """
        # THE LOGITS FROM OUR MODEL ARE OUR "REAL" PREDICTIONS
        _, predictions = self.predict(user_features)

        # GET KNOWN POSITIVES
        interaction = tf.where(A > 0.0, A, 0.0)

        top_k_indices = tf.math.top_k(predictions, k=k).indices

        relevant_retrieved = gather_matrix_indices(interaction, tf.cast(top_k_indices, dtype=tf.int64))

        hits = tf.math.count_nonzero(relevant_retrieved, axis=1, dtype=tf.float32)

        if not preserve_rows:
            relevant = tf.math.count_nonzero(tf.where(A > 0.0, A, 0.0), axis=1, dtype=tf.float32)
            zero_int_mask = tf.math.not_equal(relevant, 0.0)
            masked_hits = tf.boolean_mask(hits, zero_int_mask)
            return masked_hits / k
        else:
            return hits / k

    def f1_at_k(self, user_features, A, k=10, beta=1.0):
        """
        A harmonic mean of both precision at k and recall at k. Beta indicates how much more we are going to weight the precision over recall.
        :param user_features: a tensorflow tensor: the input features
        :param A: a tensorflow tensor: the interaction table
        :param k: a python int: the number of top predictions to use in judging
        :param beta: a python float: a weighting parameter that influences the contribution of precision vs. recall
        :return: a python float: the f1 score @ k, a harmonic mean of precision @ k and recall @ k metrics
        """

        precision, recall = self.precision_at_k(user_features, A, k=k), self.recall_at_k(user_features, A, k=k)

        prec, rec = tf.reduce_mean(precision), tf.reduce_mean(recall)

        return ((1 + beta**2) * prec * rec) / (beta**2 * (prec + rec))

    def dcg_at_k(self, user_features, dense_interactions, k=10):
        """
        Compute the Discounted Cumulative Gain (DCG) for top k predictions.

        The DCG score considers the order of the items retrieved in a query (i.e. recommended items for a user).

        A relevant (in the interaction table) item that is retrieved earlier contributes more to the DCG score. This metric judges the effectiveness of the recommender/retrieval system in not only retrieving relevant recommendations, but doing so in the most relevant ORDER as well.

        :param user_features: tensorflow tensor: the input features
        :param dense_interactions: tensorflow tensor: a dense (ordinary) tensor containing the interactions
        :param k: python int: the amount of items retrieved
        :return: tensorflow tensor: contains the dcg score per user for the top k retrieved items
        """
        _, predictions = self.predict(user_features)
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

    def idcg_at_k(self, user_features, dense_interactions, k=10):
        """
        Computes the Ideal Discounted Cumulative Gain (IDCG) for top k predictions.

        The IDCG score is the DCG score for a theoretically perfect recommendation/retrieval (i.e. if the top k retrieved items for a user corresponded to the relevant items in the original interaction).

        :param user_features: a tensorflow tensor: the input features
        :param dense_interactions: a tensorflow tensor: a dense (ordinary) tensor containing the interactions
        :param k: python int: the amount of items retrieved
        :return: tensorflow tensor: contains the idcg score per user for top k items retrieved
        """
        # WE WILL RETURN LOGITS AS OUR PREDICTIONS
        _, predictions = self.predict(user_features)
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

    def ndcg_at_k(self, user_features, A, k=10, preserve_rows=False):
        """
        Computes the NDCG Score (Normalized Discounted Cumulative Gain).

        Computes the success of the model's retrieval by judging if it retrieved relevant items with most priority.

        :param user_features: tensorflow tensor: the input features
        :param A: tensorflow tensor: a dense interaction table
        :param k: python int: how many top items we will retrieve
        :param preserve_rows: python boolean: flag indicating whether to preserve the users with no interactions
        :return:
        """
        # compute dcg
        dcg = self.dcg_at_k(user_features, A, k)

        # compute idcg
        idcg = self.idcg_at_k(user_features, A, k)

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


    def retrieve_user_recs(self, user_feature, user=None, k=None):
        """
        Method to retrieve the item recommendations for a certain user. We will be retrieving item ranks, not prediction values.

        :param user_feature: a tensorflow tensor: the specific tensor representing the features of all users
        :param user: python int: a row index representing the user
        :param k: python int: specifies how many top items to bring
        :return: a numpy array containing the indices to the top k rated items
        """
        all_predictions = self.predict(user_feature)
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