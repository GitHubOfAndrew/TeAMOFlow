# THIS CONTAINS THE NECESSARY ABSTRACTION FOR THE MATRIX FACTORIZATION MODEL
# NOTE:
# - WE DO NOT HAVE TO CREATE A COMPUTATIONAL GRAPH SEPARATELY, AS THE EAGER EXECUTION
# AND AUTOGRAPH WILL AUTOMATICALLY INCORPORATE ANY TENSORFLOW METHODS INTO THE GRAPH
# - THIS RUNS ONLY ON TENSORFLOW 2.5+, USE AT YOUR OWN RISK

import tensorflow as tf
import timeit as t


class MatrixFactorization:
    """Class for a Matrix Factorization Model. A hybrid (content-based and collaborative filtering) recommender model
    that takes user embeddings and item embeddings to predict unobserved interactions in a given interaction table"""

    def __init__(self, num_features, shape):

        """
        Arguments:\n
        - num_features: a python int, number of latent features for the user and item embeddings\n
        - shape: a python tuple of the form (n,m) where (n,m) is the shape of the interaction table we are approximating

        Purpose:
        Initializes the following instance attributes:\n
        - user embeddings: U ---> of shape (n, num_features)\n
        - item embeddings: V ---> of shape (m, num_features)
        """
        n, m = shape
        user_shape = (n, num_features)
        item_shape = (m, num_features)

        self.U = tf.Variable(tf.random.uniform(shape=user_shape), trainable=True, dtype=tf.float32)
        self.V = tf.Variable(tf.random.uniform(shape=item_shape), trainable=True, dtype=tf.float32)

    @tf.function
    def loss(self, A, U, V, lambda_1, lambda_2):
        """
        Arguments:\n
        - A: an array with 2 axes, an interaction table between users and items\n
        - U: user embeddings
        - V: item embeddings
        - lambda_1: weighting hyperparameters that controls contribution of observed interactions
        - lambda_2: weighting hyperparameters that controls contribution of unobserved interactions

        Purpose:\n
        Outputs the loss for both observed and unobserved interactions. In particular, it creates a loss graph for computation
        in tensorflow's autograd

        2022-06-03 - The only loss available is the MSE loss, this will be changed as necessary as we discover more losses to use
        """

        # mean squared error loss with regularization
        return tf.where(A != 0, tf.multiply(lambda_1, tf.pow(A - tf.matmul(U, tf.transpose(V)), 2)),
                        tf.multiply(lambda_2, tf.pow(-tf.matmul(U, tf.transpose(V)), 2)))

    def fit(self, A, epochs, optimizer, lambda_1=0.01, lambda_2=0.001, verbose=1):
        """
        Arguments:\n
        - A: an array with 2 axes, an interaction table between users and items\n
        - epochs: a python int, the number of times the model will train\n
        - optimizer: a tf.keras.optimizers object, the optimization algorithm used in the minimization of the loss\n
        - lambda_1: a python float, the hyperparamater that weights the contribution of observed entries in the training\n
        - lambda_2: a python float, the hyperparameter that weights the contribution of unobserved entries in the training\n
        - verbose: a python int,\n
        --- 0: no information about training process printed\n
        --- 1: epoch number, loss, cumulative training runtime printed\n
        --- 2: epoch number printed

        Purpose:\n
        Runs a training loop for a specified loss function using an optimization algorithm of choice.
        """

        # get history of losses and training parameters
        train_history, li_loss = {}, []

        # recast hyperparameters as tf.constant objects so it can be incorporated into graph
        lambda_1, lambda_2 = tf.constant(lambda_1), tf.constant(lambda_2)

        total_time = 0
        for epoch in range(epochs):

            # start timer
            start = t.default_timer()

            with tf.GradientTape(persistent=True) as tape:

                # initialize loss to take gradients
                loss_fn = self.loss(A, self.U, self.V, lambda_1, lambda_2)

            # backpropagate to compute gradient
            dloss_dU, dloss_dV = tape.gradient(loss_fn, self.U), tape.gradient(loss_fn, self.V)

            # get arrays of variables and gradients
            li_grads = [dloss_dU, dloss_dV]
            li_vars = [self.U, self.V]

            # now do the optimization
            optimizer.apply_gradients(zip(li_grads, li_vars))

            # compute the total mean loss and end timer
            epoch_loss = tf.reduce_mean(loss_fn).numpy()
            end = t.default_timer()

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
                        f'Epoch {epoch + 1}/{epochs} | Loss {epoch_loss} | Total Runtime {total_time} sec.')

            if verbose == 2:
                total_time += (end - start)

                # print the updates for every epoch
                if (epoch + 1) % 50 == 0:
                    print(f'Epoch {epoch + 1}/{epochs} | Total Runtime {total_time} sec.')

        return train_history
