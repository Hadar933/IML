import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

"""

============  k-Fold Cross Validation on Polynomial Fitting  ============

"""
m = 1500  # num samples
X = np.random.uniform(-3.2, 2.2, m)
f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
mu, sigma = 0, 5
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
    x_S, y_S = x_D[:500], y_D[:500]  # S for train
    x_V, y_V = x_D[500:], y_D[500:]  # V for validation
    return x_D, y_D, x_S, y_S, x_V, y_V


def plot_2fold_error():
    """
    trains the data for 2-fold CV and plots the error
    :return:
    """
    x_D, y_D, x_S, y_S, x_V, y_V = set_2fold_data()
    h_s = [fit_polynomial(x_S, y_S, d + 1) for d in range(poly_deg)]
    y_hat_V_arr = [h_s[d].predict(x_V.reshape(-1, 1)) for d in range(poly_deg)]
    V_err = [mean_squared_error(y_hat_V, y_V) for y_hat_V in y_hat_V_arr]
    plt.plot(range(1, poly_deg + 1), V_err)
    plt.plot(range(1, poly_deg + 1), V_err, label="Validation Error")
    plt.title(f"Fitting degrees of polynomial")
    plt.legend(), plt.xlabel("degree-k"), plt.ylabel("MSE error")
    plt.grid(), plt.show()


def set_kfold_data(k, k_fold=5, x_D=X[:1000], y_D=y[:1000]):
    """
    separates the data to S,V,T for k-fold, initialized with 5
    """
    separator = np.remainder(np.arange(y_D.size), k_fold)
    x_S_fold, y_S_fold = x_D[separator != k], y_D[separator != k]
    x_V_fold, y_V_fold = x_D[separator == k], y_D[separator == k]
    return x_S_fold, y_S_fold, x_V_fold, y_V_fold


def kfold_error(kfold=5):
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


def plot_errors(valid_err, train_err, x_range, xtitle, title):
    """
    plots the validation and training error
    """
    plt.plot(x_range, valid_err, '-*', label="Validation Error")
    plt.plot(x_range, train_err, '-o', label="Training Error")
    plt.title(title)
    plt.legend(), plt.xlabel(xtitle), plt.ylabel("MSE error")
    plt.grid(), plt.show()


def fit_and_train_best_poly(best_deg, x_D=X[1000:], y_D=y[1000:], x_T=X[1000:], y_T=y[1000:]):
    """
    fits the polynomial with the best degree and find its error on the training set
    """
    x_S, y_S = x_D[:500], y_D[:500]
    h_best = fit_polynomial(x_S, y_S, best_deg)
    y_hat = h_best.predict(x_T.reshape(-1, 1))
    error = mean_squared_error(y_T, y_hat)
    print(f"test error is: {error}")


"""

============  k-Fold and Regularization  ============

"""

from sklearn import datasets
from sklearn import linear_model

X, y = datasets.load_diabetes(return_X_y=True)
m = 50
x_S, y_S, x_T, y_T = X[:m], y[:m], X[m:], y[m:]
num_regularization_terms = 50
regularization_values = np.linspace(0.00001, 3, num=num_regularization_terms)


def ridge_error(kfold=5):
    """
    generates a list of validation error and train error for a given kFold parameter
    """
    valid_err = np.zeros(num_regularization_terms)
    train_err = np.zeros(num_regularization_terms)
    for k in range(kfold):
        x_S_fold, y_S_fold, x_V_fold, y_V_fold = set_kfold_data(k, 5, x_S, y_S)
        h_s = [linear_model.Ridge(alpha=a, tol=1e-3).fit(x_S_fold, y_S_fold) for a in regularization_values]
        y_hat_V_arr = [item.predict(x_V_fold) for item in h_s]
        y_hat_S_arr = [item.predict(x_S_fold) for item in h_s]
        V_err = [mean_squared_error(y_hat_V, y_V_fold) for y_hat_V in y_hat_V_arr]
        S_err = [mean_squared_error(y_hat_S, y_S_fold) for y_hat_S in y_hat_S_arr]
        valid_err += np.array(V_err) / kfold  # adding the respective normalized error for every d
        train_err += np.array(S_err) / kfold  # same here, just for S
    return valid_err, train_err


def lasso_error(kfold=5):
    """
    generates a list of validation error and train error for a given kFold parameter
    """
    valid_err = np.zeros(num_regularization_terms)
    train_err = np.zeros(num_regularization_terms)
    for k in range(kfold):
        x_S_fold, y_S_fold, x_V_fold, y_V_fold = set_kfold_data(k, 5, x_S, y_S)
        h_s = [linear_model.Lasso(alpha=a, tol=1e-3).fit(x_S_fold, y_S_fold) for a in regularization_values]
        y_hat_V_arr = [item.predict(x_V_fold) for item in h_s]
        y_hat_S_arr = [item.predict(x_S_fold) for item in h_s]
        V_err = [mean_squared_error(y_hat_V, y_V_fold) for y_hat_V in y_hat_V_arr]
        S_err = [mean_squared_error(y_hat_S, y_S_fold) for y_hat_S in y_hat_S_arr]
        valid_err += np.array(V_err) / kfold  # adding the respective normalized error for every d
        train_err += np.array(S_err) / kfold  # same here, just for S
    return valid_err, train_err


def best_reg_term(valid_err_lasso, valid_err_ridge):
    """
    find the best regularization terms for both lasso and ridge (in terms of minimizer of the MSE)
    """
    reg_term_ridge = regularization_values[np.argmin(valid_err_ridge)]
    reg_term_lasso = regularization_values[np.argmin(valid_err_lasso)]
    print("Best Regularization Term:")
    print(f"Lasso: {reg_term_lasso}")
    print(f"Ridge: {reg_term_ridge}")
    return reg_term_lasso, reg_term_ridge


def train_error_for_best_h(best_lasso_reg_term, best_ridge_reg_term, x_S=x_S, x_T=x_T, y_S=y_S, y_T=y_T):
    """
    trains the hyphotesis with lambda that corresponds to the lowest MSE
    and calculates the error on the train test for it (both for LASSO and RIDGE)
    """
    h_best_lasso = linear_model.Lasso(alpha=best_lasso_reg_term).fit(x_S, y_S)
    h_best_ridge = linear_model.Ridge(alpha=best_ridge_reg_term).fit(x_S, y_S)
    h_best_linear = linear_model.LinearRegression().fit(x_S, y_S)
    y_hat_lasso = h_best_lasso.predict(x_T)
    y_hat_ridge = h_best_ridge.predict(x_T)
    y_hat_linear = h_best_linear.predict(x_T)
    error_lasso = mean_squared_error(y_T, y_hat_lasso)
    error_ridge = mean_squared_error(y_T, y_hat_ridge)
    error_linear = mean_squared_error(y_T, y_hat_linear)
    print(f"Error on Test Set:\n"
          f"Ridge: {error_ridge}\n"
          f"Lasso: {error_lasso}\n"
          f"Linear: {error_linear}")


if __name__ == '__main__':

    # v_err_lasso, t_err_lasso = lasso_error()
    # v_err_ridge, t_err_ridge = ridge_error()
    # reg_term_lasso, reg_term_ridge = best_reg_term(v_err_lasso, v_err_ridge)
    # train_error_for_best_h(reg_term_lasso, reg_term_ridge)
    # plot_errors(v_err_lasso, t_err_lasso, regularization_values, rf"Regularization Term $\lambda$",
    #             "Lasso Regression Polynomial Fitting")
    # plot_errors(v_err_ridge, t_err_ridge, regularization_values, rf"Regularization Term $\lambda$",
    #             "Ridge Regression Polynomial Fitting")
    pass