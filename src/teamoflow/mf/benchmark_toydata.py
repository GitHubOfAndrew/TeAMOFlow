# SCRIPT TO TEST MATRIX FACTORIZATION MODEL in matrix_factorization.py

import tensorflow as tf
from .matrix_factorization import MatrixFactorization
from .loss_graphs import WMRBLoss

from .utils import generate_random_interaction


# call the instance

if __name__ == "__main__":
    loss = 'wmrb'

    # give dimensions and load toy data
    n_users, n_items, n_components = 100, 100, 3
    tf_interaction, A = generate_random_interaction(n_users, n_items, density=0.01)

    if loss == 'mse':
        # generate user, item features: use indicator features
        user_features = tf.eye(n_users)
        item_features = tf.eye(n_items)

        # initialize model

        model1 = MatrixFactorization(n_components)

        # check attributes
        print(model1.user_repr_graph)
        print(model1.item_repr_graph)
        print(model1.loss_graph)

        # run training loop
        epochs = 450
        model1.fit(epochs, user_features, item_features, tf_interaction)

        # run prediction
        _, tf_predictions = model1.predict(A)

        # get scores

        recall_at_10_1 = model1.recall_at_k(A)
        precision_at_10_1 = model1.precision_at_k(A)
        f1_at_10_1 = model1.f1_at_k(A)
        print(f'Recall at 10: {tf.reduce_mean(recall_at_10_1)}')
        print(f'Precision at 10: {tf.reduce_mean(precision_at_10_1)}')
        print(f'F1 at 10: {f1_at_10_1}')

    if loss == 'wmrb':
        n_sampled_items = n_items // 2

        # generate user, item features: use indicator features
        user_features = tf.eye(n_users)
        item_features = tf.eye(n_items)

        # initialize model
        model2 = MatrixFactorization(n_components=n_components, n_users=n_users, n_items=n_items,
                                     n_samples=n_sampled_items, generate_sample=True, loss_graph=WMRBLoss())

        epochs = 150

        model2.fit(epochs, user_features, item_features, tf_interaction, is_sample_based=True)

        recall_at_10_2 = model2.recall_at_k(A, preserve_rows=True)
        precision_at_10_2 = model2.precision_at_k(A, preserve_rows=True)
        f1_at_10_2 = model2.f1_at_k(A)

        print(f'Recall @ 10 w/ WMRB: {tf.reduce_mean(recall_at_10_2).numpy()}')
        print(f'Precision @ 10 w/ WMRB: {tf.reduce_mean(precision_at_10_2).numpy()}')
        print(f'f1 @ 10 w/ WMRB: {f1_at_10_2}')

