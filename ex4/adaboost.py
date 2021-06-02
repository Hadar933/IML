"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

"""
import matplotlib.pyplot as plt
import numpy as np
from ex4_tools import *


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None] * T  # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        m = X.shape[0]
        D = np.ones(m) / m  # uniform initial distribution
        for t in range(self.T):
            h = DecisionStump(D, X, y)
            self.h[t] = h
            y_hat = h.predict(X)  # current prediction
            disagree = y_hat - y
            disagree[disagree != 0] = 1  # = 1[yi != h(xi)]
            error_t = np.dot(D, disagree)  # weighted sum
            w = 0.5 * np.log(1 / error_t - 1)
            self.w[t] = w
            D = D * np.exp(-y * w * y_hat)  # element-wise
            D /= np.sum(D)  # normalize

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        return np.sign(np.sum([self.w[i] * self.h[i].predict(X) for i in range(max_t)]))

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the wrong predictions when predict only with max_t weak learners (float)
        """
        y_hat = self.predict(X, max_t)
        return len(y[y != y_hat]) / len(y)


def q13():
    """
    trains ada-boost and plots the training error and test error as a function of T
    """
    # training algorithm:
    train_samples, test_samples, noise, T = 500, 200, 0, 500
    X_train, y_train = generate_data(train_samples, noise)
    X_test, y_test = generate_data(test_samples, noise)
    m = X_train.shape[0]
    D = np.ones(m) / m
    ds = DecisionStump(D, X_train, y_train)
    ab = AdaBoost(ds, T)
    ab.train(X_train, y_train)

    # getting error rates:
    test_err = []
    train_err = []
    for max_t in range(T):
        test_err.append(ab.error(X_train, y_train, max_t))
        train_err.append(ab.error(X_test, y_test, max_t))

    # plotting
    x = [i for i in range(T)]
    plt.plot(x, test_err)
    plt.plot(x, train_err)
    plt.title("AdaBoost Error")
    plt.xlabel("T")
    plt.ylabel("Error")
    plt.legend(["Test Error", "Train Error"])
    plt.grid()
    plt.show()


if __name__ == '__main__':
    q13()
