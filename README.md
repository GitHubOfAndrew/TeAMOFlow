# TeAMOFlow - (Tensor And Matrix Operations (with) tensorFlow)

## Purpose

This library is created to quickly deploy matrix-based models using TensorFlow as the computational and programmatic framework. TeAMOFlow stands for "***Te***nsor ***A***nd ***M***atrix ***O***perations (using) Tensor***Flow***". In particular, TeAMOFlow, at its core, is a library meant to facilitate matrix factorization-based recommender systems.

## Demonstration

To start seeing how to utilize models, let us generate random interaction data calling our generate_random_interaction() method.

The following code snippet will generate a random interaction table between 50 users, 75 items, which is 5% dense (meaning about only 5% of the entries are nonzero). The nonzero entries in this matrix indicate a *positive interaction* of some sort (these can be explicit interactions like ratings, likes; or implicit interactions like clicks, views, etc.). **Note: As of now, TeAMOFlow only accomodates positive interactions. There are work-arounds to incorporate negative interactions that we may include in a future update.**

```
n_users, n_items = 50, 75

tf_sparse_interaction, A = generate_random_interaction(n_users, n_items, max_entry=5.0, density=0.05)
```

This project is extremely early in its development cycle and not nearly close to completed. For more information, or if you would like to collaborate, please contact me at andrewjych@gmail.com.

## Acknowledgments

We would like to acknowledge the efforts of James Kirk and his fantastic project **Tensorrec** (link: https://github.com/jfkirk/tensorrec), from which this project took inspiration from. In fact, this project came out as an effort to adopt Tensorrec for the TensorFlow 2.x environment (as that library was written on TensorFlow 1.x). By no means are the ideas behind this work original, they are from many fantastic tutorials and academic research done in the field. Please contact me if I have violated any policies regarding fair use or plagiarism.
