# THIS CONTAINS THE UTILITY FUNCTIONS NECESSARY FOR SOME OF OUR IMPLEMENTATIONS

import numpy as np

# def random_sampler(n_items, n_users, n_samples, replace=False):
#
#     """
#     Arguments:\n
#     - n_items: python int, the number of items
#     - n_users: python int, the number of users
#     - n_samples: python int, the number of user samples
#     - replace: python boolean, whether we sample with replacement or without\n
#     NOTE: we should almost always sample without replacement as we will get duplicate indices
#
#    Purpose:\n
#    - Randomly sample indices from our existing interaction table
#    - Returns a numpy array of sampled indices
#     """
#
#     items_per_user = [np.random.choice(a=n_items, size=n_samples, replace=replace) for _ in range(n_users)]
#
#     sample_indices = []
#
#     for user, user_items in enumerate(items_per_user):
#         for item in user_items:
#             sample_indices.append((user, item))
#
#     return np.array(sample_indices)
