"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  
"""
import random

import matplotlib.pyplot as plt

import mnist_loader

import numpy as np


def create_mini_batches(train_data, size):
    """
    shuffles the data and splits it to sections of constant given size
    :param train_data: the training data provided
    :param size: list of (x,y) tuples
    :return:
    """
    np.random.shuffle(train_data)
    mini_batches = [train_data[i:i + size] for i in range(0,len(train_data), size)]
    return mini_batches


def cost_derivative(output_activations, y):
    """
    Return the vector of partial derivatives partial C_x /
    partial a for the output activations.
    """
    return output_activations - y


class Network:

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.epochs = []  # x axis for later plots
        self.accuracies = []  # y axis for later plots

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for w, b in zip(self.weights, self.biases):
            a = sigmoid((w @ a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(list(test_data))
        for j in range(epochs):
            mini_batches = create_mini_batches(training_data,mini_batch_size)
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                eval = self.evaluate(test_data)
                print(f"Epoch {j}: {eval} / {n_test}")
                self.epochs.append(j)  # this will be used for later plots
                self.accuracies.append(eval / n_test)  # same
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch) * nw) for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch) * nb) for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # forward pass:
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):  # recalculating the z's and a's according to the new b's and w's
            z = w @ activation + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass : calculating the weights backwards as we've seen in class (derivatives)
        delta = sigmoid_prime(zs[-1])*cost_derivative(activations[-1], y)
        nabla_w[-1] = delta @ activations[-2].T
        nabla_b[-1] = delta
        for i in range(2, self.num_layers):  # iterating backwards
            delta = sigmoid_prime(zs[-i]) * (self.weights[-i + 1].T @ delta)
            nabla_w[-i] = delta @ activations[-i - 1].T
            nabla_b[-i] = delta
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        """
        network_ret = np.array([np.argmax(self.feedforward(x)) for (x, y) in test_data])
        all_y = np.array([y for (x, y) in test_data])
        return np.sum(network_ret == all_y)


# Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    exp = np.exp(-z)
    return exp / ((1 + exp) ** 2)


def set_network(train, test, sizes, epoch, mini_batch_size, learning_rate):
    """
    sets up the network with the wanted parameters and trains it
    :param num_layers: not including input layer and output layer
    :return: trained network
    """
    my_net = Network(sizes)
    my_net.SGD(train, epoch, mini_batch_size, learning_rate, test_data=test)
    return my_net


def plot(x_data, y_data, title, x_title="Epochs", y_title="Accuracy"):
    plt.plot(x_data, y_data)
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)


if __name__ == '__main__':
    sizes = [784, 30, 10]
    train, validate, test = mnist_loader.load_data_wrapper()
    accuracy = []
    epochs = [i for i in range(30)]
    for eta in [3, 5, 30]:
        nn = set_network(train, test, sizes, epoch=30, mini_batch_size=10, learning_rate=eta)
        plot(nn.epochs, nn.accuracies, title=r"Epoch vs Accuracy for various $\eta$'s")
    plt.legend([r"$\eta$=3", r"$\eta$=5", r"$\eta$=15", r"$\eta$=30"])
    plt.show()
