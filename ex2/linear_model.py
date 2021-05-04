import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


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


def remove_non_positive_values(df):
    """
    removes all non positive values for some relevant features
    such as price, sqft, etc...
    :param df: panda df
    :return: processed df
    """
    feature_lst = ['price', 'sqft_lot15']  # all instances of non positive values are removed when iterating on these
    # two features
    for feature in feature_lst:
        col = np.array(df[feature])
        bad_row_indexes = np.argwhere(col <= 0).T[0]
        df.drop(bad_row_indexes.tolist(), inplace=True)
    df.dropna(inplace=True)  # remove nan values


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    remove_non_positive_values(df)
    df.drop(columns=['lat', 'long', 'id', 'date'], inplace=True)  # lat/long is related to zipcode
    df = pd.get_dummies(df, columns=['zipcode'])  # generate dummies from zipcode
    price = df.pop('price')
    return df, price


def plot_singular_values(singular_values):
    """
    plots the singular values in descending order
    :param singular_values:
    :return:
    """
    descending = sorted(singular_values, reverse=True)
    plt.scatter(range(1, len(descending) + 1), descending)
    plt.title("Signular Values $\sigma_i$ in descending order")
    plt.xlabel("# of Singular Value (i)")
    plt.ylabel("Singular Value $\sigma_i$ - log scale")
    plt.yscale('log')
    plt.show()


def load_fit_plot():
    """
    code that loads the dataset, performs
    the preprocessing and plots the singular values plot
    """
    X, y = load_data('kc_house_data.csv')
    w, sigma = fit_linear_regression(X, y)
    plot_singular_values(sigma)


def mse_plot(df, y):
    """
    :param df:
    :return:
    """
    train, test, y_train, y_test = train_test_split(df, y)  # default is train = 75%, test=25%
    train_size = train.shape[0]
    mse_lst = []
    for p in range(1, 101):
        max_index = int((p / 100) * train_size)
        curr_training_set = train.iloc[:max_index, :]  # slicing the needed percentage
        curr_test_set = test.iloc[:max_index, :]
        curr_y_train = y_train[:max_index]
        curr_y_test = y_test[:max_index]
        w_hat, sig = fit_linear_regression(curr_training_set, curr_y_train)
        y_hat = predict(curr_test_set, w_hat)
        curr_error = mse(curr_y_test, y_hat)
        print(f'p={p}%, error={curr_error}')
        mse_lst.append(curr_error)

    plt.title("MSE values as function of p% - log scale")
    plt.xlabel('Percentage (p%)')
    plt.ylabel('log MSE')
    plt.yscale("log")
    plt.plot(range(1, 101), mse_lst)
    plt.show()


def pearson(feature1, feature2):
    """
    :return: pearson correlation between two features
    """
    # the resulting pearson matrix is a 2 by 2 in which the elements that are not on the diagonal
    # are the relevant return value
    return (np.cov(feature1, feature2) / (np.std(feature1) * np.std(feature2)))[0][1]


def feature_evaluation(X, y):
    """
    for every non-categorical feature, a graph (scatter plot) of the feature
    values and the response values. It then also computes and shows on the
    graph the Pearson Correlation between the feature and the response
    :param X: design matrix
    :param y: response vector
    :return:
    """
    non_categor_labels = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                          'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built',
                          'yr_renovated', 'sqft_living15', 'sqft_lot15']
    for i in range(len(non_categor_labels)):
        feature = X[non_categor_labels[i]]
        plt.title(f"Price as a function of {non_categor_labels[i]}, $\\rho$={round(pearson(feature, y), 2)}")
        plt.xlabel(f"{non_categor_labels[i]}")
        plt.ylabel('Price')
        plt.scatter(feature, y)
        plt.show()
