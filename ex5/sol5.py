import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

m = 1500  # num samples
X = np.random.uniform(-3.2, 2.2, m)
f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
mu, sigma = 0, 1
epsilon = np.random.normal(mu, sigma, m)  # noise
y = f(X) + epsilon  # labels with noise
poly_deg = 15


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


def plot_kfold(kfold, x_S, y_S, x_V, y_V):
    """
    generates a list of validation error and train error for a given kFold parameter
    """
    valid_err = np.zeros(poly_deg)
    train_err = np.zeros(poly_deg)
    for k in range(kfold):
        h_s = [fit_polynomial(x_S, y_S, d + 1) for d in range(poly_deg)]
        y_hat_V_arr = [h_s[d].predict(x_V.reshape(-1, 1)) for d in range(poly_deg)]
        y_hat_S_arr = [h_s[d].predict(x_S.reshape(-1, 1)) for d in range(poly_deg)]
        V_err = [mean_squared_error(y_hat_V, y_V) for y_hat_V in y_hat_V_arr]
        S_err = [mean_squared_error(y_hat_S, y_S) for y_hat_S in y_hat_S_arr]

        valid_err += np.array(V_err) / kfold  # adding the respective normalized error for every d
        train_err += np.array(S_err) / kfold  # same here, just for S
    return valid_err, train_err


def plot_errors(valid_err, train_err):
    best_deg = np.argmin(valid_err) + 1
    print(f"valid={valid_err}")
    print(f"train={train_err}")
    plt.plot(range(1, poly_deg + 1), valid_err, label="Validation Error")
    plt.plot(range(1, poly_deg + 1), train_err, label="Training Error")
    plt.title(f"Fitting degrees of polynomial - best fit ={best_deg}")
    plt.legend(), plt.xlabel("degree-k"), plt.ylabel("MSE error")
    plt.grid(), plt.show()


if __name__ == '__main__':
    # 2-fold data:
    x_D, y_D = X[:1000], y[:1000]  # D for S and V
    x_T, y_T = X[1000:], y[1000:]  # T for Test
    x_S, y_S = x_D[:500], y_D[:500]  # S for train
    x_V, y_V = x_D[500:], y_D[500:]  # V for validatio
    v_err, t_err = plot_kfold(2, x_S, y_S, x_V, y_V)
    plot_errors(v_err, t_err)
