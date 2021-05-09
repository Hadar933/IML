from tensorflow.keras.datasets.mnist import load_data
import numpy as np
import matplotlib.pyplot as plt
from comparison import accuracy, plot_accuracies
from models import Logistic, SVM, DecisionTree
from sklearn.neighbors import KNeighborsClassifier
import plotly.graph_objects as go


def load_mnist():
    (x_train, y_train), (x_test, y_test) = load_data()
    train_images = np.logical_or((y_train == 0), (y_train == 1))
    test_images = np.logical_or((y_test == 0), (y_test == 1))
    x_train, y_train = x_train[train_images], y_train[train_images]
    x_test, y_test = x_test[test_images], y_test[test_images]
    return x_train, y_train, x_test, y_test


def q12():
    """
    plots three images of 0 and three images of 1
    """
    x_train, y_train, x_test, y_test = load_mnist()
    indexes = [0, 5, 8, 1, 2, 3]  # picked first three zero and one values from x_train
    for i in indexes:
        plt.imshow(x_train[i, :, :], cmap='gray')
        plt.show()


def rearrange_data(X):
    """
    :param X: array X of size n * 28 * 28
    :return: matrix of size n * 784 with the same data
    """
    n, m = X.shape[0], X.shape[1]
    return X.reshape(n, m * m)


def get_Xy_until_good(m, X_train, y_train):
    """
    loops until y has both 1 and -1
    :param m: number of sample
    :return: X,y
    """
    while True:  # must have both +1 and -1 in y
        indexes = np.random.choice(len(X_train), m)
        X, y = X_train[indexes], y_train[indexes]
        if 0 in y and -1 in y:
            break
    return X, y


def q10():
    m_vals = [50, 100, 300, 500]
    X_train, y_train, X_test, y_test = load_mnist()

    all_accuracies = []
    models = {"Logistic": Logistic(), "SVM": SVM(), "TREE": DecisionTree(), "KNN": KNeighborsClassifier(n_neighbors=6)}
    repeat = 50
    for m in m_vals:
        acc_dict = {"Logistic": [], "SVM": [], "TREE": [], "KNN": []}
        for i in range(repeat):
            X, y = get_Xy_until_good(m, X_train, y_train)
            for model in models:
                models[model].fit(X, y)
                y_hat = models[model].predict(X_test)
                acc_dict[model].append(accuracy(y_test, y_hat))
        all_accuracies.append(acc_dict)

    for dictionary in all_accuracies:
        for model in dictionary:
            dictionary[model] = np.mean(np.array(dictionary[model]))
    return all_accuracies


plot_accuracies(q10(), [50, 100, 300, 500])
