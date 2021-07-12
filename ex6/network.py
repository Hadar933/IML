"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  
"""
import mnist_loader
import numpy as np


def create_mini_batches(data, size):
    """
    shuffles the data and splits it to sections of with constant given size
    :param data:
    :param size: list of (x,y) tuples
    :return:
    """
    np.random.shuffle(data)
    mini_batches = np.split(data, size)
    return mini_batches


class Network(object):

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
        n_test = 0
        if test_data: n_test = len(list(test_data))
        n = len(list(training_data))
        for j in range(epochs):
            mini_batches = create_mini_batches(np.array(training_data), mini_batch_size)
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
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
            activation.append(activation)

        # backward pass : calculating the weights backwards as we've seen in class (derivatives)
        C_vec = self.cost_derivative(activations[-1], y)
        delta = C_vec * sigmoid(zs[-1])
        nabla_w[-1] = delta @ activations[-2].T
        nabla_b[-1] = delta
        for i in range(2, self.num_layers):
            delta = sigmoid_prime(zs[-i]) * (self.weights[-i + 1].T @ delta)
            nabla_w[-i] = delta @ activations[-i - 1].T
            nabla_b[-i] = delta
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        # -----------TODO------------##
        # -----------TODO------------##

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        # TODO
        return output_activations


# Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    exp = np.exp(-z)
    return exp / ((1 + exp) ** 2)


if __name__ == '__main__':
    train, validate, test = mnist_loader.load_data_wrapper()
    train = train[:500]
    validate = validate[:100]
    test = test[:100]
    nn = Network([2, 3, 1])
    nn.SGD(train, 5, 10, 0.1)