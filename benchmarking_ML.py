# THIS FILE IS MEANT TO BE FOR BENCHMARKING PURPOSES OF VARIOUS CONFIGURATIONS OF MATRIX FACTORIZATION MODEL
# LOAD AND RUN MODEL ON: MOVIELENS 100K DATASET, link to dataset: https://grouplens.org/datasets/movielens/100k/, extract it and then put the csv's inside a directory called 'ml-latest-small' in the same folder as this one
# CREDIT: THE LOADING OF THIS DATASET IS TAKEN FROM TENSORREC (BY JAMES KIRK, THE FILE IS CALLED getting_started.py)

# base libraries

import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import sparse
import csv
from collections import defaultdict
import random

# our libraries

from matrix_factorization import MatrixFactorization
from loss_graphs import WMRBLoss
from input_utils import *


# This method converts a list of (user, item, rating, time) to a sparse matrix
def interactions_list_to_sparse_matrix(interactions, n_users, n_items):
    users_column, items_column, ratings_column, _ = zip(*interactions)
    return sparse.coo_matrix((ratings_column, (users_column, items_column)),
                             shape=(n_users, n_items))

# flag for running benchmarks
run, load = True, True

# load data
if load == True:
    with open('./ml-latest-small/ratings.csv', 'r') as ratings_file:
        ratings_file_reader = csv.reader(ratings_file)
        raw_ratings = list(ratings_file_reader)
        raw_ratings_header = raw_ratings.pop(0)

    # Iterate through the input to map MovieLens IDs to new internal IDs
    # The new internal IDs will be created by the defaultdict on insertion
    movielens_to_internal_user_ids = defaultdict(lambda: len(movielens_to_internal_user_ids))
    movielens_to_internal_item_ids = defaultdict(lambda: len(movielens_to_internal_item_ids))
    for row in raw_ratings:
        row[0] = movielens_to_internal_user_ids[int(row[0])]
        row[1] = movielens_to_internal_item_ids[int(row[1])]
        row[2] = float(row[2])
    n_users = len(movielens_to_internal_user_ids)
    n_items = len(movielens_to_internal_item_ids)

    # Look at an example raw rating
    print("Raw ratings example:\n{}\n{}".format(raw_ratings_header, raw_ratings[0]))

    # Shuffle the ratings and split them in to train/test sets 80%/20%
    random.shuffle(raw_ratings)  # Shuffles the list in-place
    cutoff = int(.8 * len(raw_ratings))
    train_ratings = raw_ratings[:cutoff]
    test_ratings = raw_ratings[cutoff:]
    print("{} train ratings, {} test ratings".format(len(train_ratings), len(test_ratings)))

    # Create sparse matrices of interaction data
    sparse_train_ratings = interactions_list_to_sparse_matrix(train_ratings, n_users, n_items)
    sparse_test_ratings = interactions_list_to_sparse_matrix(test_ratings, n_users, n_items)

    # Create sets of train/test interactions that are only ratings >= 4.0
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

    # initialize model
    n_users, n_items = A_train.shape
    n_sampled_items = n_items // 2
    n_components = 5
    epochs = 100

    # WE WILL BE TRAINING BOTH MSE AND WMRB MODELS WITH THE SAME FEATURES AND HYPERPARAMETERS TO COMPARE PERFORMANCE

    # mse model
    model_ml_100k = MatrixFactorization(n_components)

    # WMRB model
    model_ml_100k_wmrb = MatrixFactorization(n_components, loss_graph=WMRBLoss(), n_users=n_users, n_items=n_items,
                                             n_samples=n_sampled_items, generate_sample=True)

    # initialize indicator features

    user_features = tf.eye(n_users)

    item_features = tf.eye(n_items)

    # fit model
    model_ml_100k.fit(epochs, user_features, item_features, tf_train, lr=1e-3)

    model_ml_100k_wmrb.fit(epochs, user_features, item_features, tf_train, is_sample_based=True, lr=1e-3)

    # score model
    k = 10
    recall_train = model_ml_100k.recall_at_k(A_train, k)
    recall_train_2 = model_ml_100k_wmrb.recall_at_k(A_train, k)
    # precision_train = model_ml_100k.precision_at_k(A_train, k)

    print(f'Recall @ 10 on training set w/ MSE: {tf.reduce_mean(recall_train).numpy()}')
    print(f'Recall @ 10 on training set w/ WMRB: {tf.reduce_mean(recall_train_2).numpy()}')
    print('\n')

    recall_test = model_ml_100k.recall_at_k(A_test, k)
    recall_test_2 = model_ml_100k_wmrb.recall_at_k(A_test, k)
    # precision_test = model_ml_100k.precision_at_k(A_test, k)
    print(f'Recall @ 10 on testing set w/ MSE: {tf.reduce_mean(recall_test).numpy()}')
    print(f'Recall @ 10 on testing set w/ WMRB: {tf.reduce_mean(recall_test_2).numpy()}')
    print('\n')

    recall_train_4plus = model_ml_100k.recall_at_k(A_train_4plus, k)
    recall_train_4plus_2 = model_ml_100k_wmrb.recall_at_k(A_train_4plus, k)
    # precision_train_4plus = model_ml_100k.precision_at_k(A_train_4plus, k)
    print(f'Recall @ 10 on training set (ratings >= 4) w/ MSE: {tf.reduce_mean(recall_train_4plus).numpy()}')
    print(f'Recall @ 10 on training set (ratings >= 4) w/ WMRB: {tf.reduce_mean(recall_train_4plus_2).numpy()}')
    print('\n')

    recall_test_4plus = model_ml_100k.recall_at_k(A_test_4plus, k)
    recall_test_4plus_2 = model_ml_100k_wmrb.recall_at_k(A_test_4plus, k)
    # precision_test_4plus = model_ml_100k.precision_at_k(A_test_4plus, k)
    print(f'Recall @ 10 on testing set (ratings >= 4) w/ MSE: {tf.reduce_mean(recall_test_4plus).numpy()}')
    print(f'Recall @ 10 on testing set (ratings >= 4) w/ WMRB: {tf.reduce_mean(recall_test_4plus_2).numpy()}')
    # print('\n')

    # recall_test_4plus_at_5 = model_ml_100k.recall_at_k(A_test_4plus, k=5)
    # precision_test_4plus_at_5 = model_ml_100k.precision_at_k(A_test_4plus, k=5)
    # print(f'Recall @ 5 on testing set (ratings >= 4): {tf.reduce_mean(recall_test_4plus_at_5).numpy()}')
    # print(f'Precision @ 5 on testing set (ratings >= 4): {tf.reduce_mean(precision_test_4plus_at_5).numpy()}')

