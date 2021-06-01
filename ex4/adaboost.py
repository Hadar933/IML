"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

"""
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
        D = np.full((1, m), 1 / m)  # uniform initial distribution
        for t in range(self.T):
            h = self.WL(D, X, y)
            self.h[t] = h
            y_hat = h.predict(X)  # current prediction
            diff = y_hat - y
            disagree = diff[diff != 0] = 1  # = 1[yi != h(xi)]
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
        # TODO complete this function


if __name__ == '__main__':
    X, y = generate_data(10, 2)

    m = X.shape[0]
    D = np.full((m, 1), 1 / m)
    ds = DecisionStump(D, X, y)
    ab = AdaBoost(ds, 5)
    ab.train(X, y)
