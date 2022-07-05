# TeAMOFlow - (Tensor And Matrix Operations (with) tensorFlow)

## Purpose

This library is created to quickly deploy matrix-based models using TensorFlow as the computational and programmatic framework. TeAMOFlow stands for "***Te***nsor ***A***nd ***M***atrix ***O***perations (using) Tensor***Flow***". In particular, TeAMOFlow, at its core, is a library meant to facilitate matrix factorization-based recommender systems. 

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

The code to test our model out on this dataset is given in the file 'benchmarking_ML.py'. We will not implement it on this page, but we will show the results below for the following configuration:

```
# the dimension of the range of the embedding
n_components = 5

# initialize model with default embedding, loss, initializer graphs
model = MatrixFactorization(n_components)

# number of epochs to optimize the model
epochs = 100

# learning rate for the optimizer (Adam)
learning_rate = 1e-3

# user and item features; both set as identity matrices for baseline benchmarking purposes
user_features = tf.eye(n_users)

item_features = tf.eye(n_items)
```

<img width="407" alt="TeAMOFlow MF benchmark N_comp-5 lr-1e-3 pt 2 img 2" src="https://user-images.githubusercontent.com/85316690/177405951-67104365-2d27-4464-9293-3dde05baf9ce.PNG">

Although far from optimal (on all ratings >= 4, there's a 0.14% chance that a user gets an item that they like in the top 10 predictions [recall @ 10]), this does give us a baseline for the performance of this model. Given the scale of the data however, this 0.14% is more significant than it seems. The configuration of this model is the most basic possible configuration (MSE loss, un-biased linear embeddings, normal initialization).

I check the consistency of these results by referring to a previous benchmark demonstrated by James Kirk (creator of Tensorrec [look in *Acknowledgements*]), run with my exact configurations above, in this article: https://towardsdatascience.com/getting-started-with-recommender-systems-and-tensorrec-8f50a9943eef). 

**Note:** Results may vary due to differences in hardware and environment configurations.

## Acknowledgments

We would like to acknowledge the efforts of James Kirk and his fantastic project **Tensorrec** (link: https://github.com/jfkirk/tensorrec), from which this project took inspiration from. In fact, this project came out as an effort to adopt Tensorrec for the TensorFlow 2.x environment (as that library was written on TensorFlow 1.x). By no means are the ideas behind this work original, they are from many fantastic tutorials and academic research done in the field. Please contact me if I have violated any policies regarding fair use or plagiarism.

***Note:*** This project is extremely early in its development cycle and not nearly close to completed to where I would like. For more information, or if you would like to collaborate, please contact me at andrewjych@gmail.com.
