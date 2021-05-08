import numpy as np


class Perceptron:

    def __init__(self):
        self.w = None  # module weights initialization

    def fit(self, X, y):
        """
        :param X: m rows (Samples) d columns (features)
        :param y: {+-1}^m
        :return: w (weights) d rows
        """
        new_X = np.insert(X, 0, 1, axis=1)  # add column of ones to the beginning of X (for the non-homogenous case)
        m, d = new_X.shape[0], new_X.shape[1]
        self.w = np.zeros((d, 1))
        entered_if = False  # if we didnt enter the inner if for all i in m, we can return
        while True:
            for i in range(m):
                if y[i] * np.dot(self.w[i], new_X[i]) <= 0:
                    entered_if = True
                    self.w += (y[i] * new_X[i])
            if not entered_if:
                break

    def predict(self, X):
        pass

    def score(self, X, y):
        pass


class LDA:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def score(self, X, y):
        pass


class SVM:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def score(self, X, y):
        pass


class Logistic:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def score(self, X, y):
        pass
