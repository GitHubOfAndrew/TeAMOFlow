# SCRIPT TO TEST MATRIX FACTORIZATION MODEL in matrix_factorization.py

import tensorflow as tf

from src.teamoflow.mf.matrix_factorization import MatrixFactorization
from src.teamoflow.mf.loss_graphs import WMRBLoss, KLDivergenceLoss
from src.teamoflow.mf.embedding_graphs import *

from src.teamoflow.mf.utils import generate_random_interaction


# call the instance

if __name__ == "__main__":
    loss = 'wmrb'

    # give dimensions and load toy data
    n_users, n_items, n_components = 500, 1000, 5
    tf_interaction, A = generate_random_interaction(n_users, n_items, min_val=0.0, density=0.01)

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

        # initialize models
        model2 = MatrixFactorization(n_components=n_components, n_users=n_users, n_items=n_items,
                                     n_samples=n_sampled_items, generate_sample=True, loss_graph=WMRBLoss())

        model3 = MatrixFactorization(n_components=n_components, n_users=n_users, n_items=n_items,
                                     n_samples=n_sampled_items, generate_sample=True, loss_graph=WMRBLoss(),
                                     user_repr_graph=ReLUEmbedding(), item_repr_graph=ReLUEmbedding())

        epochs = 100

        model2.fit(epochs, user_features, item_features, tf_interaction, lr=0.1)
        model3.fit(epochs, user_features, item_features, tf_interaction, lr=0.1)

        recall_at_10_2 = model2.recall_at_k(A, preserve_rows=True)
        precision_at_10_2 = model2.precision_at_k(A, preserve_rows=True)
        f1_at_10_2 = model2.f1_at_k(A)
        ndcg_at_10_2 = model2.ndcg_at_k(A)

        recall_at_10_3 = model3.recall_at_k(A, preserve_rows=True)
        precision_at_10_3 = model3.precision_at_k(A, preserve_rows=True)
        f1_at_10_3 = model3.f1_at_k(A)
        ndcg_at_10_3 = model3.ndcg_at_k(A)

        print(f'Recall @ 10 w/ WMRB: {tf.reduce_mean(recall_at_10_2).numpy()}')
        print(f'Precision @ 10 w/ WMRB: {tf.reduce_mean(precision_at_10_2).numpy()}')
        print(f'f1 @ 10 w/ WMRB: {f1_at_10_2}')
        print(f'NDCG @ 10 w/ WMRB: {tf.reduce_mean(ndcg_at_10_2).numpy()}')
        print('\n')
        print(f'Recall @ 10 w/ WMRB, ReLU Embedding: {tf.reduce_mean(recall_at_10_3).numpy()}')
        print(f'Precision @ 10 w/ WMRB, ReLU Embedding: {tf.reduce_mean(precision_at_10_3).numpy()}')
        print(f'f1 @ 10 w/ WMRB, ReLU Embedding: {f1_at_10_3}')
        print(f'NDCG @ 10 w/ WMRB, ReLU Embedding: {tf.reduce_mean(ndcg_at_10_3).numpy()}')
        print('\n')
        print('Check User ReLU Weights:')
        print(model3.user_relu_bias)
        print(model3.user_trainable[2])
        print('\n')
        print('Check Item ReLU Weights:')
        print(model3.item_relu_bias)
        print(model3.item_trainable[2])


    if loss == 'KL':
        # WE USE KL DIVERGENCE FOR INTERACTIONS WITH BOTH NEGATIVE AND POSITIVE INTERACTIONS

        # generate user, item features: use indicator features
        user_features = tf.eye(n_users)
        item_features = tf.eye(n_items)

        # initialize model

        model3 = MatrixFactorization(n_components, loss_graph=KLDivergenceLoss())

        # check attributes
        print(model3.user_repr_graph)
        print(model3.item_repr_graph)
        print(model3.loss_graph)

        # run training loop
        epochs = 30
        model3.fit(epochs, user_features, item_features, tf_interaction, lr=0.0001)

        # run prediction
        _, tf_predictions = model3.predict(A)

        # get scores

        recall_at_10_3 = model3.recall_at_k(A)
        precision_at_10_3 = model3.precision_at_k(A)
        f1_at_10_3 = model3.f1_at_k(A)
        print(f'Recall at 10: {tf.reduce_mean(recall_at_10_3)}')
        print(f'Precision at 10: {tf.reduce_mean(precision_at_10_3)}')
        print(f'F1 at 10: {f1_at_10_3}')

