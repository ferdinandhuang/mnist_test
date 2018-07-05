import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../')

from loadmnist import load
loadfnc = load('../mnist')
# load data
# train_images, train_labels, test_images, test_labels = load('../mnist').load_mnist()
# print('train_images shape:%s' % str(train_images.shape))
# print('train_labels shape:%s' % str(train_labels.shape))
# print('test_images shape:%s' % str(test_images.shape))
# print('test_labels shape:%s' % str(test_labels.shape))


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
    n_h = 10
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


def forward_propagation(X, parameters, activation="tanh"):
    """
    Compute the forword propagation
    :param X: input data (m, n_x)
    :param parameters: parameters from initialize_parameters
    :param activation: activation function name, has "tanh" and "relu"
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
    if activation == "tanh":
        A1 = np.tanh(Z1)
    elif activation == "relu":
        A1 = relu(Z1)
    else:
        raise Exception('Activation function is not found!')
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return Z2, cache


def back_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    :param parameters:
    :param cache:
    :param X:
    :param Y:
    :return:
    """

    m = X.shape[0]
    X = X.T
    Y = Y.T

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    # dZ2 = d_mean_cross_entropy_softmax(A2, Y)
    dZ2 = A2 - Y

    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = np.dot(W2.T, dZ2) * d_relu(A1)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters, grads, learning_rate = 0.001):
    """
    Updates parameters using the gradient descent update rule given above

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients

    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def relu(z):
    return (abs(z) + z) / 2


def d_relu(a):
    "Compute the derivative of RELU given activation (a)."
    d = np.zeros_like(a)
    d[np.where(a > 0.0)] = 1.0
    return d


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def mean_cross_entropy(outputs, labels):
    n = labels.shape[0]
    return - np.sum(labels * np.log(outputs)) / n


def mean_cross_entropy_softmax(logits, labels):
    return mean_cross_entropy(softmax(logits), labels)


def d_mean_cross_entropy_softmax(logits, labels):
    return softmax(logits) - labels


def loss(outputs, labels):
    "Compute the cross entropy softmax loss."
    return mean_cross_entropy_softmax(outputs, labels)


def d_loss(outputs, labels):
    "Compute derivatives of the cross entropy softmax loss w.r.t the outputs."
    return d_mean_cross_entropy_softmax(outputs, labels)



X = loadfnc.train_images
Y = loadfnc.train_labels_onehot
n_x, n_h, n_y = layer_size(X, Y)
parameters = initialize_parameters(n_x, n_h, n_y)
for i in range(0, 20):

    activation = "relu"
    A2, cache = forward_propagation(X, parameters, activation)

    lost = loss(A2, Y.T)

    print("lost:%s" % str(lost))

    grads = back_propagation(parameters, cache, X, Y)

    parameters = update_parameters(parameters, grads, learning_rate = 0.1)

