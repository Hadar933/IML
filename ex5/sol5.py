import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

"""
initializing some variables here:
"""
m = 1500  # num samples
X = np.random.uniform(-3.2, 2.2, m)
f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
mu, sigma = 0, 1
epsilon = np.random.normal(mu, sigma, m)  # noise
y = f(X) + epsilon  # labels with noise
poly_deg = 15  # maximal degree of polynomial


def fit_polynomial(x_S, y_S, deg):
    """
    trains a polynomial fitting hypothesis
    :param x_S: training samples
    :param y_S: labels
    :param deg: degree of polynomial
    :return: hypothesis
    """
    x_S, y_S = x_S.reshape(-1, 1), y_S.reshape(-1, 1)
    h_S = make_pipeline(PolynomialFeatures(deg), LinearRegression()).fit(x_S, y_S)
    return h_S


def set_2fold_data(x_D=X[:1000], y_D=y[:1000]):
    """
    separates the data to S,V,T for 2-fold
    """
    x_T, y_T = X[1000:], y[1000:]  # T for Test
    x_S, y_S = x_D[:500], y_D[:500]  # S for train
    x_V, y_V = x_D[500:], y_D[500:]  # V for validation
    return x_D, y_D, x_T, y_T, x_S, y_S, x_V, y_V


def plot_2fold_error():
    x_D, y_D, x_T, y_T, x_S, y_S, x_V, y_V = set_2fold_data()
    h_s = [fit_polynomial(x_S, y_S, d + 1) for d in range(poly_deg)]
    y_hat_V_arr = [h_s[d].predict(x_V.reshape(-1, 1)) for d in range(poly_deg)]
    V_err = [mean_squared_error(y_hat_V, y_V) for y_hat_V in y_hat_V_arr]
    plt.plot(range(1, poly_deg + 1), V_err)
    best_deg = np.argmin(V_err) + 1
    plt.plot(range(1, poly_deg + 1), V_err, label="Validation Error")
    plt.title(f"Fitting degrees of polynomial - best fit deg = {best_deg}")
    plt.legend(), plt.xlabel("degree-k"), plt.ylabel("MSE error")
    plt.grid(), plt.show()


def set_kfold_data(k, k_fold=5, x_D=X[:1000], y_D=y[:1000]):
    """
    separates the data to S,V,T for k-fold, initialized with 5
    """
    separator = np.remainder(np.arange(x_D.size), k_fold)
    x_S_fold, y_S_fold = x_D[separator != k], y_D[separator != k]
    x_V_fold, y_V_fold = x_D[separator == k], y_D[separator == k]
    return x_S_fold, y_S_fold, x_V_fold, y_V_fold


def kfold_error(kfold):
    """
    generates a list of validation error and train error for a given kFold parameter
    """
    valid_err = np.zeros(poly_deg)
    train_err = np.zeros(poly_deg)
    for k in range(kfold):
        x_S_fold, y_S_fold, x_V_fold, y_V_fold = set_kfold_data(k)
        h_s = [fit_polynomial(x_S_fold, y_S_fold, d + 1) for d in range(poly_deg)]
        y_hat_V_arr = [h_s[d].predict(x_V_fold.reshape(-1, 1)) for d in range(poly_deg)]
        y_hat_S_arr = [h_s[d].predict(x_S_fold.reshape(-1, 1)) for d in range(poly_deg)]
        V_err = [mean_squared_error(y_hat_V, y_V_fold) for y_hat_V in y_hat_V_arr]
        S_err = [mean_squared_error(y_hat_S, y_S_fold) for y_hat_S in y_hat_S_arr]
        valid_err += np.array(V_err) / kfold  # adding the respective normalized error for every d
        train_err += np.array(S_err) / kfold  # same here, just for S
    return valid_err, train_err


def plot_errors(valid_err, train_err):
    print(f"valid={valid_err}")
    print(f"train={train_err}")
    plt.plot(range(1, poly_deg + 1), valid_err, label="Validation Error")
    plt.plot(range(1, poly_deg + 1), train_err, label="Training Error")
    plt.title(f"Fitting degrees of polynomial")
    plt.legend(), plt.xlabel("degree-k"), plt.ylabel("MSE error")
    plt.grid(), plt.show()


if __name__ == '__main__':
    plot_2fold_error()
