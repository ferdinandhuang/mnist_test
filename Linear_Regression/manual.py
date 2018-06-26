import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../')

from loadmnist import load

# load data
train_images, train_labels, test_images, test_labels = load('../mnist').load_mnist()
print('train_images shape:%s' % str(train_images.shape))
print('train_labels shape:%s' % str(train_labels.shape))
print('test_images shape:%s' % str(test_images.shape))
print('test_labels shape:%s' % str(test_labels.shape))


def layer_size(X, Y):
    """
    Get number of input and output size, and set hidden layer size
    :param X: input dataset's shape(m, 784)
    :param Y: input labels's shape(m,1)
    :return:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """

    n_x = X.T.shape[0]
    n_h = 4
    n_y = Y.T.shape[0]

    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
    """
    Initialize parameters
    :param n_x: the size of the input layer
    :param n_h: the size of the hidden layer
    :param n_y: the size of the output layer
    :return: dictionary of parameters
    """

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2
                  }

    return parameters


def forward_propagation(X, parameters):
    """
    Compute the forword propagation
    :param X: input data (m, n_x)
    :param parameters: parameters from initialize_parameters
    :return:
        cache: caches of forword result
        A2: sigmoid output
    """

    X = X.T

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache

X = train_images
n_x, n_h, n_y = layer_size(X, train_labels)

parameters = initialize_parameters(n_x, n_h, n_y)

A2, cache = forward_propagation(X, parameters)
print(A2)