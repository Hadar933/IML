import numpy as np


def fit_linear_regression(X, y):
    """

    :param X: design matrix (m rows x d columns) = (m samples x d features)
    :param y: response vector (m rows)
    :return: 1. w: coefficient vector ()
             2. singular values of X
    """
