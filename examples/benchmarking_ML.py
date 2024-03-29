# THIS FILE IS MEANT TO BE FOR BENCHMARKING PURPOSES OF VARIOUS CONFIGURATIONS OF MATRIX FACTORIZATION MODEL
# LOAD AND RUN MODEL ON: MOVIELENS 100K DATASET, link to dataset: https://grouplens.org/datasets/movielens/100k/, extract it and then put the csv's inside a directory called 'ml-latest-small' in the same folder as this one
# CREDIT: THE LOADING OF THIS DATASET IS TAKEN FROM TENSORREC (BY JAMES KIRK, THE FILE IS CALLED getting_started.py)

# base libraries

import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import sparse
import random

# our libraries

from src.teamoflow.mf.matrix_factorization import MatrixFactorization
from src.teamoflow.mf.initializer_graphs import NormalInitializer, UniformInitializer
from src.teamoflow.mf.loss_graphs import WMRBLoss
from src.teamoflow.mf.embedding_graphs import *
from src.teamoflow.mf.input_utils import *


# This method converts a list of (user, item, rating, time) to a sparse matrix
def interactions_list_to_sparse_matrix(interactions, n_users, n_items):
    users_column, items_column, ratings_column, _ = zip(*interactions)
    return sparse.coo_matrix((ratings_column, (users_column, items_column)),
                             shape=(n_users, n_items))

# flag for running benchmarks
run, load = True, True

# load data
if load == True:

    # load dataframe
    df = pd.read_csv('./ml-latest-small/ratings.csv').rename(columns={'userId': 'User ID', 'movieId': 'Items'}).drop(columns=['timestamp'], axis=1)
    sparse_train_ratings, sparse_test_ratings = df_to_sparse_pipeline(df, test_size=0.25)

    sparse_train_ratings_4plus = sparse_train_ratings.multiply(sparse_train_ratings >= 4.0)
    sparse_test_ratings_4plus = sparse_test_ratings.multiply(sparse_test_ratings >= 4.0)

# run benchmarks
if run == True:
    train_np = sparse_train_ratings.toarray()

    test_np = sparse_test_ratings.toarray()

    train_np_4plus = sparse_train_ratings_4plus.toarray()

    test_np_4plus = sparse_test_ratings_4plus.toarray()

    # get sparse tensor for fitting
    tf_train = convert_to_tf_sparse(train_np)

    # tensors used in scoring process
    A_train = tf.constant(train_np, dtype=tf.float32)
    A_test = tf.constant(test_np, dtype=tf.float32)
    A_train_4plus = tf.constant(train_np_4plus, dtype=tf.float32)
    A_test_4plus = tf.constant(test_np_4plus, dtype=tf.float32)

    # fit wmrb with 4+ ratings to push those positive ratings to the top
    tf_train_4plus = convert_to_tf_sparse(train_np_4plus)

    # initialize model
    n_users, n_items = A_train.shape
    n_sampled_items = n_items // 5
    n_components = 5
    epochs = 100

    # WE WILL BE TRAINING BOTH MSE AND WMRB MODELS WITH THE SAME FEATURES AND HYPERPARAMETERS TO COMPARE PERFORMANCE

    # mse model
    model_ml_100k = MatrixFactorization(n_components)

    # WMRB model
    model_ml_100k_wmrb = MatrixFactorization(n_components, user_weight_graph=UniformInitializer(), item_weight_graph=UniformInitializer(),
                                             loss_graph=WMRBLoss(), n_users=n_users, n_items=n_items,
                                             n_samples=n_sampled_items, generate_sample=True)

    # WMRB model w/ biased linear embedding
    model_ml_100k_wmrb_user_biased = MatrixFactorization(n_components, user_weight_graph=UniformInitializer(),
                                                    item_weight_graph=UniformInitializer(), loss_graph=WMRBLoss(),
                                                    user_repr_graph=BiasedLinearEmbedding(), item_repr_graph=BiasedLinearEmbedding(),
                                                    n_users=n_users, n_items=n_items, n_samples=n_sampled_items,
                                                    generate_sample=True)

    # WMRB model w/ ReLU embedding
    model_ml_100k_wmrb_relu_repr = MatrixFactorization(n_components, user_weight_graph=UniformInitializer(),
                                                       item_weight_graph=UniformInitializer(), loss_graph=WMRBLoss(),
                                                       user_repr_graph=ReLUEmbedding(), item_repr_graph=ReLUEmbedding(),
                                                       n_users=n_users, n_items=n_items, n_samples=n_sampled_items,
                                                       generate_sample=True)

    # initialize indicator features

    user_features = tf.eye(n_users)

    item_features = tf.eye(n_items)

    # fit model
    model_ml_100k.fit(epochs, user_features, item_features, tf_train, lr=1e-3)

    model_ml_100k_wmrb.fit(epochs, user_features, item_features, tf_train_4plus, lr=0.1)

    model_ml_100k_wmrb_user_biased.fit(epochs, user_features, item_features, tf_train_4plus, lr=0.1)

    # model_ml_100k_wmrb_relu_repr.fit(epochs, user_features, item_features, tf_train_4plus, lr=0.1)

    # score model
    k = 10
    recall_train = model_ml_100k.recall_at_k(A_train, k)
    recall_train_2 = model_ml_100k_wmrb.recall_at_k(A_train, k)
    recall_train_3 = model_ml_100k_wmrb_user_biased.recall_at_k(A_train)
    # recall_train_4 = model_ml_100k_wmrb_relu_repr.recall_at_k(A_train)
    # precision_train = model_ml_100k.precision_at_k(A_train, k)

    print(f'Recall @ 10 on training set w/ MSE: {tf.reduce_mean(recall_train).numpy()}')
    print(f'Recall @ 10 on training set w/ WMRB: {tf.reduce_mean(recall_train_2).numpy()}')
    print(f'Recall @ 10 on training set w/ WMRB, biased: {tf.reduce_mean(recall_train_3).numpy()}')
    # print(f'Recall @ 10 on training set w/ WMRB, ReLU representation: {tf.reduce_mean(recall_train_4).numpy()}')
    print('\n')

    recall_test = model_ml_100k.recall_at_k(A_test, k)
    recall_test_2 = model_ml_100k_wmrb.recall_at_k(A_test, k)
    recall_test_3 = model_ml_100k_wmrb_user_biased.recall_at_k(A_test)
    # recall_test_4 = model_ml_100k_wmrb_relu_repr.recall_at_k(A_test)
    # precision_test = model_ml_100k.precision_at_k(A_test, k)
    print(f'Recall @ 10 on testing set w/ MSE: {tf.reduce_mean(recall_test).numpy()}')
    print(f'Recall @ 10 on testing set w/ WMRB: {tf.reduce_mean(recall_test_2).numpy()}')
    print(f'Recall @ 10 on testing set w/ WMRB, biased: {tf.reduce_mean(recall_test_3).numpy()}')
    # print(f'Recall @ 10 on testing set w/ WMRB, ReLU representation: {tf.reduce_mean(recall_test_4).numpy()}')
    print('\n')

    recall_train_4plus = model_ml_100k.recall_at_k(A_train_4plus, k)
    recall_train_4plus_2 = model_ml_100k_wmrb.recall_at_k(A_train_4plus, k)
    recall_train_4plus_3 = model_ml_100k_wmrb_user_biased.recall_at_k(A_train_4plus, k)
    # recall_train_4plus_4 = model_ml_100k_wmrb_relu_repr.recall_at_k(A_train_4plus, k)
    # precision_train_4plus = model_ml_100k.precision_at_k(A_train_4plus, k)
    print(f'Recall @ 10 on training set (ratings >= 4) w/ MSE: {tf.reduce_mean(recall_train_4plus).numpy()}')
    print(f'Recall @ 10 on training set (ratings >= 4) w/ WMRB: {tf.reduce_mean(recall_train_4plus_2).numpy()}')
    print(f'Recall @ 10 on training set (ratings >= 4) w/ WMRB, biased: {tf.reduce_mean(recall_train_4plus_3).numpy()}')
    # print(f'Recall @ 10 on training set (ratings >= 4) w/ WMRB, ReLU representation: {tf.reduce_mean(recall_train_4plus_4).numpy()}')
    print('\n')

    recall_test_4plus = model_ml_100k.recall_at_k(A_test_4plus, k)
    recall_test_4plus_2 = model_ml_100k_wmrb.recall_at_k(A_test_4plus, k)
    recall_test_4plus_3 = model_ml_100k_wmrb_user_biased.recall_at_k(A_test_4plus, k)
    # recall_test_4plus_4 = model_ml_100k_wmrb_relu_repr.recall_at_k(A_test_4plus, k)
    # precision_test_4plus = model_ml_100k.precision_at_k(A_test_4plus, k)
    print(f'Recall @ 10 on testing set (ratings >= 4) w/ MSE: {tf.reduce_mean(recall_test_4plus).numpy()}')
    print(f'Recall @ 10 on testing set (ratings >= 4) w/ WMRB: {tf.reduce_mean(recall_test_4plus_2).numpy()}')
    print(f'Recall @ 10 on testing set (ratings >= 4) w/ WMRB, biased: {tf.reduce_mean(recall_test_4plus_3).numpy()}')
    # print(f'Recall @ 10 on testing set (ratings >= 4) w/ WMRB, ReLU representation: {tf.reduce_mean(recall_test_4plus_4).numpy()}')
    print('\n')

    recall_test_4plus_30 = model_ml_100k.recall_at_k(A_test_4plus, k=30)
    recall_test_4plus_2_30 = model_ml_100k_wmrb.recall_at_k(A_test_4plus, k=30)
    recall_test_4plus_3_30 = model_ml_100k_wmrb_user_biased.recall_at_k(A_test_4plus, k=30)
    # recall_test_4plus_4_30 = model_ml_100k_wmrb_relu_repr.recall_at_k(A_test_4plus, k=30)
    # precision_test_4plus = model_ml_100k.precision_at_k(A_test_4plus, k)
    print(f'Recall @ 30 on testing set (ratings >= 4) w/ MSE: {tf.reduce_mean(recall_test_4plus_30).numpy()}')
    print(f'Recall @ 30 on testing set (ratings >= 4) w/ WMRB: {tf.reduce_mean(recall_test_4plus_2_30).numpy()}')
    print(f'Recall @ 30 on testing set (ratings >= 4) w/ WMRB, biased: {tf.reduce_mean(recall_test_4plus_3_30).numpy()}')
    # print(f'Recall @ 30 on testing set (ratings >= 4) w/ WMRB, ReLU representation: {tf.reduce_mean(recall_test_4plus_4_30).numpy()}')
    print('\n')

    recall_test_4plus_50 = model_ml_100k.recall_at_k(A_test_4plus, k=50)
    recall_test_4plus_2_50 = model_ml_100k_wmrb.recall_at_k(A_test_4plus, k=50)
    recall_test_4plus_3_50 = model_ml_100k_wmrb_user_biased.recall_at_k(A_test_4plus, k=50)
    # recall_test_4plus_4_50 = model_ml_100k_wmrb_relu_repr.recall_at_k(A_test_4plus, k=50)
    # precision_test_4plus = model_ml_100k.precision_at_k(A_test_4plus, k)
    print(f'Recall @ 50 on testing set (ratings >= 4) w/ MSE: {tf.reduce_mean(recall_test_4plus_50).numpy()}')
    print(f'Recall @ 50 on testing set (ratings >= 4) w/ WMRB: {tf.reduce_mean(recall_test_4plus_2_50).numpy()}')
    print(f'Recall @ 50 on testing set (ratings >= 4) w/ WMRB, biased: {tf.reduce_mean(recall_test_4plus_3_50).numpy()}')
    # print(f'Recall @ 50 on testing set (ratings >= 4) w/ WMRB, ReLU representation: {tf.reduce_mean(recall_test_4plus_4_50).numpy()}')
    print('\n')

    # print('Checking Weights:')
    # print(model_ml_100k_wmrb_relu_repr.item_relu_bias)
    # print(model_ml_100k_wmrb_relu_repr.item_trainable[2])
