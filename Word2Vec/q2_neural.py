#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(X, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.

    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Note: compute cost based on `sum` not `mean`.

    # initialization
    gradW1 = np.zeros(W1.shape)+0.
    gradW2 = np.zeros(W2.shape)+0.
    gradb1 = np.zeros(b1.shape)+0.
    gradb2 = np.zeros(b2.shape)+0.
    cost = 0.

    for i in range(X.shape[0]):


        x = X[i].reshape((1, -1))
        y = labels[i].reshape((1,-1))
        ### forward propagation
        z1 = np.dot(x, W1) + b1
        h = sigmoid(z1)
        z2 = np.dot(h, W2) + b2
        y_hat = softmax(z2)

        # entropy cost
        cost_i = - np.log(np.sum(y * y_hat))

        # gradients
        delta_1 = y_hat - y
        grad_w2 = np.dot(h.T, delta_1)
        grad_b2 = delta_1

        delta_2 = np.dot(delta_1, W2.T)
        delta_3 = delta_2 * h * (1-h)
        grad_w1 = np.dot(x.T, delta_3)
        grad_b1 = delta_3

        cost += cost_i
        gradW1 += grad_w1
        gradb1 += grad_b1
        gradW2 += grad_w2
        gradb2 += grad_b2

        grad_w1 = 0
        grad_b1 = 0
        grad_w1 = 0
        grad_b2 = 0

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


# def your_sanity_checks():
#     """
#     Use this space add any additional sanity checks by running:
#         python q2_neural.py
#     This function will not be called by the autograder, nor will
#     your additional tests be graded.
#     """
#     print "Running your sanity checks..."
#     ### YOUR CODE HERE
#     raise NotImplementedError
#     ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    # your_sanity_checks()
