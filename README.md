![TeAMOFlow Logo v1 (Rectangle)](https://user-images.githubusercontent.com/85316690/182063301-622a1a85-323a-4643-b254-87445fe4d230.png)

# TeAMOFlow - (Tensor And Matrix Operations (with) tensorFlow)

If you want to skip the exposition and get started right away, please visit this [link](https://colab.research.google.com/drive/1JTRi81F3bgf_aai3Hz1eXnGVEToTpqZ-).

## Installation

**TeAMOFlow is now available to be installed.** Check it out on [PyPI](https://pypi.org/project/teamoflow/).

To install, please open your conda environment or python venv and type in the following:

```
pip install teamoflow
```

**Note: The dependencies do not get installed with this installation, please follow the following notes on minimum package dependencies.**

Dependencies:
TeAMOFlow has the following dependencies:

- Python 3.7+
- tensorflow (2.9.0+) (`pip install tensorflow`)
- tensorflow-probability (0.17.0+) (`pip install tensorflow-probability`)
- numpy (`pip install numpy`)
- scipy (`pip install scipy`)
- pandas (`pip install pandas`)

We put version numbers on the tensorflow-based libraries because the bulk of our code is built exclusively on tensorflow. The compatible version of numpy and scipy will be installed according to the version of tensorflow. Pandas is included for some of our optional utility functions, but it is not absolutely necessary for proper usage of our models. 

## Purpose

This library is created to quickly deploy matrix-based models using TensorFlow as the computational and programmatic framework. TeAMOFlow stands for "***Te***nsor ***A***nd ***M***atrix ***O***perations (with) Tensor***Flow***". In particular, TeAMOFlow, at its core, is a library meant to facilitate query-key matching through machine learning. However, we made sure that the techniques (if applicable) could be extended to multiple domains of interest, where applicable.

We have 3 objectives in developing this library:

1) Developing an extremely-user friendly workflow for quick model prototyping.

2) Giving a wide variety of configurations to play around with, while also allowing contributors to easily plug in their own ideas.

3) Updating valuable work done by previous libraries.

We achieve these as follows:

1) This is completely adapted to the current python 3.7+ ecosystem. Furthermore, this takes full advantage of tensorflow 2.X's eager execution and autograph system. Unlike tensorflow 1.X, which needed a lot of boilerplate code to build computational graphs and execute them, our library is built on top of the most user-friendly and intuitive version of two industry giants: Python 3.X, and TensorFlow 2.X.

2) We architected our code on a purely object-oriented manner, allowing for modularity in every step of the machine learning process (initialization, embedding, training, prediction, evaluation). This makes it easy to both: plug in and play, and to write custom implementations of various parts of the model.

3) We fully acknowledge the importance of those who came before us. They walked so we could fly (or at least walk faster). Much of TeAMOFlow was developed out of necessity to keep James Kirk's excellent library, TensorRec (https://github.com/jfkirk/tensorrec), alive. Kirk and TensorRec were my first introduction to recommender systems and the matrix factorization space, and while it has been long-inactive (I suspect this is largely due to the code being written in TensorFlow 1.X), there is no reason that this work has to become lost to time. I actively want people to know of TensorRec's existence and its ingenuity, and I hope to uphold its memory and continue active development upon those ideas in this library.

## Goals

While the primary object of focus is the matrix factorization component, we plan to expand the scope of this project to include a lot of relevant and interesting applications. We do not want TeAMOFlow to be a tensorflow-based scikit-learn clone, we will create a framework for various interesting and specialized models/operations. 

## Currently Available

TeAMOFlow currently has 1 library at minimum operational capacity: the Matrix Factorization library. Our next goal is to create a DNN specifically for query-key matching. This will circumvent having to load a keras instance from scratch, and it allows the user to prioritize feature engineering and preprocessing over drafting and prototyping a model.

## Concept Behind Matrix Factorization

We can view matrix factorization as a mathematical problem in the following manner:

Suppose we have a *sparse* matrix $A$. $A$ is known as the interaction table (matrix/tensor). The row headers of this interaction table represent **users** (query), the column headers represent **items** (value). This table encodes a *user* **interacting** with an *item*. For example, in the following table:

<img width="150" alt="github_readme_mf_example" src="https://user-images.githubusercontent.com/85316690/180573179-eaa9c5cd-4ef8-46a8-a3dd-dd5967323b2d.PNG">

This interaction table encodes a simple *passive* (implicit) interaction between user 1/item 1, user 2/item 3, and user 3/item 1 (this interaction could be that these users clicked on their respective items, this is typically what an "implicit" interaction entails).

The mathematical process behind matrix factorization is simple:

Let $U$ and $V$ be matrices with common rank $r$, such that $U \in M_{m\times r}(\mathbb{R})$ and $V \in M_{n\times r}(\mathbb{R})$.

Then, for *user embedding* and *item embedding*, $U$, $V$, respectively, and an appropriate loss function $\mathcal{L}$ (that computes the deviation between $A$ and $U\cdot V^{T}$), we want to find $U$, $V$ such that:

$$\mathcal{L}(A, U, V)_{ij} < \varepsilon \quad \forall i,j : A_{ij} \neq 0$$

TeAMOFlow computes the embeddings in the following way:

Let $f$, $g$ be user, item embeddings, respectively (an *embedding* is, for our purposes, **a continuous function that maps vectors into lower-dimensional vector spaces**). Then let $W_{u}$, $W_{v}$ be **user features**, **item features**, respectively (*features* are, for our purposes, data about the user/item beyond the scope of the interactions). We use our embedding functions $f$, $g$ to compute the user and item embeddings as follows:

$$U = f(W_{u}) \quad V = g(W_{v})$$

Therefore, the full workflow of matrix factorization, in TeAMOFlow, is summarizable as follows:

- Select user features and item features, $W_{u}$, $W_{v}$.
- Initialize trainable weights and compute user and item embeddings ($U$ and $V$), choosing our embedding functions $f$, $g$ appropriately. The result will be $U = f(W_{u})$, $V = g(W_{u})$.
- Compute our loss function $\mathcal{L}(A, U, V)$.
- Take the gradient and perform gradient descent/optimization on initialized weights.
- Repeat steps 3, 4.

## Getting Started with TeAMOFlow Matrix Factorization Library

**Interactions** are at the forefront of recommendations. There are usually two classes of interactions:

1) **Explicit Interactions**

These are encoded events in which users are active in describing their experience with an item. Things like 1-5 star ratings (like on Amazon), or a "like" (YouTube, Netflix, etc.) are all explicit interactions. The users are actively showing their experience with the item that they interacted with.

2) **Implicit Interactions**

These are encoded events in which users passively describe their experience with an item. Examples of this include: clicks on a webpage (almost every web-based business has this), views on a video (YouTube, TikTok, etc.). The users passively describe their experience with the item. These types of interactions are usually not as powerful as explicit interactions, but they are just as useful.

**NOTE:** Regarding ethics of privacy and data collection, it is usually advised to let users know that your data is being collected as transparency can save organizations from legal trouble and develop a sense of trust with clients/consumers. I am not a legal expert, this is just common sense.

With that said, let us write a sample workflow in teamoflow:

To start seeing how to utilize models in our Matrix Factorization library, let us generate random interaction data calling our generate_random_interaction() method.

The following code snippet will generate a random interaction table between 300 users, 1000 items, which is 0.5% dense (meaning only about 0.5% of the entries are nonzero). The nonzero entries in this matrix indicate an *interaction* between *user* and *item* of some sort. 

```
from teamoflow.mf import *
import tensorflow as tf

# initialize model

n_components = 3

mf_model1 = matrix_factorization.MatrixFactorization(n_components)

n_users, n_items = 300, 1000

# initialize indicator features

user_features = tf.eye(n_users)

item_features = tf.eye(n_items)

# generate random interaction table, note how we get a sparse representation and a dense representation

tf_sparse_toy, A_toy = utils.generate_random_interaction(n_users, n_items, min_val=0.0, max_val=3.0, density=0.005)

# fit our model using the default configuration

epochs=100

mf_model1.fit(epochs, user_features, item_features, tf_sparse_toy, lr=0.05)

# evaluate our model's performance using recall @ 10, 30, 50, 100 (up to 5 decimal places of precision - a typical C-style floating point precision)

recall_at_10_1 = mf_model1.recall_at_k(A_toy, k=10)

print(f'The recall @ 10 is {tf.reduce_mean(recall_at_10_1).numpy() * 100:.5f}%.')

recall_at_30_1 = mf_model1.recall_at_k(A_toy, k=30)

print(f'The recall @ 30 is {tf.reduce_mean(recall_at_30_1).numpy() * 100:.5f}%.')
recall_at_50_1 = mf_model1.recall_at_k(A_toy, k=50)

print(f'The recall @ 50 is {tf.reduce_mean(recall_at_50_1).numpy() * 100:.5f}%.')

recall_at_100_1 = mf_model1.recall_at_k(A_toy, k=100)

print(f'The recall @ 100 is {tf.reduce_mean(recall_at_100_1).numpy() * 100:.5f}%.')
```

The only parameter we must specify upon instantiation of our model is the *number of components* (this is the common dimension into which we embed our user and item features). We can also specify custom loss function, user/item embeddings, and the user/item initializations, but the default configuration is a model with the MSE loss, un-biased linear embeddings, and normal initializations. To see a more extensive tutorial, please go to this [link](https://colab.research.google.com/drive/1JTRi81F3bgf_aai3Hz1eXnGVEToTpqZ-).

We can see how our model performed by using the **precision @ k**, **recall @ k**, **f1 @ k** metrics. By default, we use k=10.

**Recall at k** can be interpreted in the following way: On average, there is a 'recall at k' chance that the top-k predictions in our model will contain an item that the user likes.

**Precision at k** can be interpreted in the following way: On average, out of our top-k predictions, the model predicts that the user will like 'precision at k' of the predicted items.

In recommender systems, since our main goal is to rank items according to user preferences, precision and recall are more appropriate to judging the effectiveness of our model. In other contexts, this may not be so appropriate and we can adopt different scoring methods.

There is more detail on what these metrics mean [here](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)).

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

<img width="650" alt="upload_best_setup1" src="https://user-images.githubusercontent.com/85316690/178170254-85cf68f7-a025-483b-ab8b-5ecd75f36c23.PNG">

Consider that we have the bare minimum features for both users and items (identity, indicator features). With a judicious choice of initialization and loss function, our recall @ 10 on our testing set (of interactions with ratings of 4+) increased from 0.23% to 7.53%! This means that, on average, has a 7.5% chance of recommending an item that a user would like, in its top 10 predictions. Furthermore, this model attains an 18% recall @ 30, and a 24% recall @ 50. This demonstrates the utility of a ranking loss, like the **WMRB**, in a task such as item recommendations. 

With more preprocessing, feature engineering, and different choices of predictions and embeddings, this could likely be improved to a good degree.

## Acknowledgments

We would like to acknowledge the efforts of James Kirk and his fantastic project [**TensorRec**](https://github.com/jfkirk/tensorrec), from which this project took inspiration from. In fact, this project came out as an effort to adopt Tensorrec for the TensorFlow 2.x environment (as that library is written on TensorFlow 1.x and is no longer supported). By no means are the ideas behind this work 100% original, they are from many fantastic tutorials and academic research done in the field. I have added my own optimizations and workflow in this, however. Please contact me if there are any questions or concerns.

***Note:*** This project is extremely early in its development cycle and not nearly close to completed to where I would like. Please open a pull request or an issue if there are any changes/improvements that need to be suggested.
