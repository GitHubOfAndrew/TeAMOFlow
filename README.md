# TeAMOFlow - (Tensor And Matrix Operations (with) tensorFlow)

## Purpose

This library is created to quickly deploy matrix-based models using TensorFlow as the computational and programmatic framework. TeAMOFlow stands for "***Te***nsor ***A***nd ***M***atrix ***O***perations (with) Tensor***Flow***". In particular, TeAMOFlow, at its core, is a library meant to facilitate matrix factorization-based recommender systems. 

We have 3 objectives in developing this library:

1) Developing an extremely-user friendly workflow for quick model prototyping.

2) Giving a wide variety of configurations to play around with, while also allowing contributors to easily plug in their own ideas.

3) Updating valuable work done by previous libraries.

We achieve these as follows:

1) This is completely adapted to the current python 3.6+ ecosystem. Furthermore, this takes full advantage of tensorflow 2.X's eager execution and autograph system. Unlike tensorflow 1.X, which needed a lot of boilerplate code to build computational graphs and execute them, our library is built on top of the most user-friendly and intuitive version of two industry giants: Python 3.X, and TensorFlow 2.X.

2) We architected our code on a purely object-oriented manner, allowing for modularity in every step of the machine learning process (initialization, embedding, training, prediction, evaluation). This makes it easy to both: plug in and play, and to write custom implementations of various parts of the model.

3) We fully acknowledge the importance of those who came before us. They walked so we could fly (or at least walk faster). Much of TeAMOFlow was developed out of necessity to keep James Kirk's excellent library, TensorRec (https://github.com/jfkirk/tensorrec), alive. Kirk and TensorRec were my first introduction to recommender systems and the matrix factorization space, and while it has been long-inactive (I suspect this is largely due to the code being written in TensorFlow 1.X), there is no reason that this work has to become lost to time. I actively want people to know of TensorRec's existence and its ingenuity, and I hope to uphold its memory and continue active development upon those ideas in this library.

## Goals

While the primary object of focus is the matrix factorization component, I named it "TeAMOFlow" for a reason: I hope to expand this library to encompass all kinds of matrix/tensor-based machine learning models. I intend for this library to have implementations from machine learning literature that are not commonly found in other similar libraries. This investigative spirit and willingness to experiment is something I encourage to myself, and to everyone.

## Demonstration of Matrix Factorization Model

To start seeing how to utilize models, let us generate random interaction data calling our generate_random_interaction() method.

The following code snippet will generate a random interaction table between 50 users, 75 items, which is 5% dense (meaning about only 5% of the entries are nonzero). The nonzero entries in this matrix indicate a *positive interaction* of some sort (these can be explicit interactions like ratings, likes; or implicit interactions like clicks, views, etc.). **Note: As of now, TeAMOFlow only accomodates positive interactions. There are work-arounds to incorporate negative interactions that we may include in a future update.**

```
# generate interaction table
n_users, n_items = 50, 75

tf_sparse_interaction, A = generate_random_interaction(n_users, n_items, max_entry=5.0, density=0.05)
```

We utilize a tensorflow sparse matrix (using the tensorflow.sparse.SparseTensor class) as our primary input of choice. Then we can use some features for our model. We utilize indicator features (features that are encoded in only 1's and 0's) for both the users and items, more specifically, we will just use an identity matrix.

```
# initialize user and item features
user_features = tf.eye(n_users)
item_features = tf.eye(n_items)
```

We now initialize our model. The only parameter we must specify upon instantiation of our model is the *number of components* (this is the common dimension into which we embed our user and item features). We can also specify custom loss function, user/item embeddings, and the user/item initializations, but by default, we use MSE loss, un-biased linear embeddings, and normal initializations.

```
n_components = 3
model1 = MatrixFactorization(n_components)
```

We now fit this model.

```
epochs = 400
model1.fit(epochs, user_features, item_features, tf_sparse_interaction)

# obtain predictions
predictions = model1.predict()
```

We can see how our model performed by using the **precision @ k**, **recall @ k**, **f1 @ k** metrics. By default, we use k=10.

```
# k=10 by default
recall_at_10_1 = model1.recall_at_k(A)
precision_at_10_1 = model1.precision_at_k(A)
f1_at_10_1 = model1.f1_at_k(A)
```

As an example, for a single run, on our training data (always run on test data first), we get a recall @ 10, precision @ 10, and f1 @ 10 of .251, 0.070, 0.109, respectively.

**Recall at 10** can be interpreted in the following way: On average, there is a 25.1% chance that the top-10 predictions in our model will contain an item that the user likes.

**Precision at 10** can be interpreted in the following way: On average, out of our top-10 predictions, the model predicts that the user will like 7.01% of the predicted items.

In matrix factorization, since our main goal is to rank items according to user preferences, precision and recall are more appropriate to judging the effectiveness of our model. These will be our predominant metrics.

For more detail into what precision and recall are, visit: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)

## Benchmark Results

We ensure the validity of the matrix factorization model by evaluating its performance on benchmark datasets. 

The MovieLens 100k dataset is data containing movies, users, and features about the movies, users and their interactions (user ratings of movies). As the name suggests, the dataset has about 100k nonzero ratings of movies by users, which we deem to be interactions.

The code to test our model out on this dataset is given in the file 'benchmarking_ML.py'. We will not implement it on this page, but we will show the results. We trained the two (main) types of models available in TeAMOFlow: a regression-based rating model, and a ranking model. The former uses a vectorized Mean-Squared Error Loss on observed interactions to predict ratings of unobserved user-item interactions. The latter uses a loss function called the **Weighted Margin Rank Batch Loss** (WMRB) which ranks suitable items higher than less suitable items. Using the following configurations:

```
# NOTE: The following configurations are in benchmarking_ML.py
# MSE Rating Model

epochs = 100
learning_rate = 1e-3
Weight_Initialization = NormalInitializer()
Loss = MSELoss()
n_components = 5

# WMRB Ranking Model

epochs = 100
learning_rate = 0.1
Weight_Initialization = UniformInitializer()
Loss = WMRBLoss()
n_components = 5
n_samples = 1944
```

We obtain the following results:

<img width="444" alt="github_movielens100k_best_benchmark1" src="https://user-images.githubusercontent.com/85316690/178095761-13e4b317-c7e0-4dba-ab51-d65257488835.PNG">

Consider that we have the bare minimum features for both users and items. With a judicious choice of initialization and loss function, the improvements from the default rating model to the ranking model are staggering (over 2 Orders of Magnitude improvement). With more preprocessing, feature engineering, and different choices of predictions and embeddings, this could likely be improved to a good degree.

I check the consistency of these results by referring to a previous demonstration by James Kirk (creator of Tensorrec), run with [roughly] the same configurations as above, just look in this article: https://towardsdatascience.com/getting-started-with-recommender-systems-and-tensorrec-8f50a9943eef). 

**Note:** Results will vary due to differences in computer hardware/environments, and simply due to the random nature of our model (namely in the initializations and, in case of the WMRB, the random sampling. Consider setting a random seed in your running environment if you do not want this to be the case).

## Acknowledgments

We would like to acknowledge the efforts of James Kirk and his fantastic project **TensorRec** (link: https://github.com/jfkirk/tensorrec), from which this project took inspiration from. In fact, this project came out as an effort to adopt Tensorrec for the TensorFlow 2.x environment (as that library is written on TensorFlow 1.x and is no longer supported). By no means are the ideas behind this work 100% original, they are from many fantastic tutorials and academic research done in the field. I have added my own optimizations and workflow in this, however. Please contact me if I have violated any policies regarding fair use or plagiarism.

***Note:*** This project is extremely early in its development cycle and not nearly close to completed to where I would like. Please open a pull request or an issue if there are any changes/improvements that need to be suggested.
