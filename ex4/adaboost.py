"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

"""
import matplotlib.pyplot as plt
import numpy as np

import ex4_tools
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
            h = self.WL(D, X, y)
            self.h[t] = h
            y_hat = h.predict(X)  # current prediction
            disagree = y_hat - y
            disagree[disagree != 0] = 1  # = 1[yi != h(xi)]
            error_t = np.dot(D, disagree)  # weighted sum
            w = 0.5 * np.log(1 / error_t - 1)
            self.w[t] = w
            D = D * np.exp(-y * w * y_hat)  # element-wise
            D /= np.sum(D)  # normalize
        return D

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        predict = [self.w[i] * self.h[i].predict(X) for i in range(max_t)]
        return np.sign(sum(predict))

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


def q13(noise):
    """
    trains ada-boost and plots the training error and test error as a function of T
    """
    # training adaboost:
    train_samples, test_samples, T = 5000, 200, 500
    X_train, y_train = generate_data(train_samples, noise)
    X_test, y_test = generate_data(test_samples, noise)
    ab = AdaBoost(DecisionStump, T)
    ab.train(X_train, y_train)

    # getting error rates:
    test_err = []
    train_err = []
    for max_t in range(T):
        test_err.append(ab.error(X_train, y_train, max_t))
        train_err.append(ab.error(X_test, y_test, max_t))

    # plotting
    x = [i for i in range(T)]
    plt.plot(x, test_err), plt.plot(x, train_err)
    plt.title("AdaBoost Error"), plt.xlabel("T"), plt.ylabel("Error")
    plt.legend(["Test Error", "Train Error"])
    plt.grid()
    plt.show()


def q14(noise):
    """
    plot the decisions of the learned classifiers for some various T's,
    together with the test data
    :return:
    """
    X_train, y_train = generate_data(5000, noise)
    X_test, y_test = generate_data(200, noise)
    T = [5, 10, 50, 100, 200, 500]
    for ind, t in enumerate(T):
        ab = AdaBoost(DecisionStump, t)
        ab.train(X_train, y_train)
        plt.subplot(2, 3, ind + 1)
        decision_boundaries(ab, X_test, y_test, t)
        plt.title(f"{t} classifiers")
    plt.show()


def q15(noise):
    """
    find the T that minimizes the test error and plots
    the boundaries for such T
    """
    T = 500
    test_err_lst = []
    X_train, y_train = generate_data(5000, noise)
    X_test, y_test = generate_data(200, noise)
    for t in range(T):
        ab = AdaBoost(DecisionStump, t)
        ab.train(X_train, y_train)
        curr_error = ab.error(X_test, y_test, t)
        test_err_lst.append((curr_error, ab, t))
        print(f"{t}")
    best_val = min(test_err_lst, key=lambda t: t[0])  # get min by first element
    err, ab, t = best_val
    decision_boundaries(ab, X_train, y_train, t)
    plt.title(f"best T={t}, giving test error of {err}")
    plt.show()


def q16(noise):
    """
    plotting classification with weights
    :return:
    """
    T = 500
    X_train, y_train = generate_data(5000, noise)
    ab = AdaBoost(DecisionStump, T)
    D = ab.train(X_train, y_train)
    D = D / np.max(D) * 10  # normalizing
    decision_boundaries(ab, X_train, y_train, T, D)
    plt.show()


def q17():
    """
    repeat the process with error rate
    """
    for noise in [0.01, 0.4]:
        q13(noise)
        q14(noise)
        q15(noise)
        q16(noise)


if __name__ == '__main__':
    q17()
