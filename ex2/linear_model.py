import numpy as np


def fit_linear_regression(X, y):
    """
    returns the weight and singular values that correspond to x=X^dag * y
    where X^dag is the pseudo inverse of X
    :param X: design matrix (m rows x d columns) = (m samples x d features)
    :param y: response vector (m rows)
    :return: 1. w: coefficient vector
             2. singular values of X
    """
    U, sigma, V_transpose = np.linalg.svd(X)
    x_dag = np.linalg.pinv(X)
    w = x_dag @ y
    return w, sigma


def predict(X, w):
    """
    :param X: design matrix
    :param w: weights matrix
    :return: returns the prediction y
    """
    return X @ w


def mse(y, y_hat):
    """
    :param y: response vector
    :param y_hat: prediction
    :return: mean square error between y_hat and y
    """
    return np.mean((y_hat - y) ** 2)


X_mat = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
y_vec = np.array([1, 2, 3])
w, sing_vals = fit_linear_regression(X_mat, y_vec)
