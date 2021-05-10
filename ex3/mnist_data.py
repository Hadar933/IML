import time
import numpy as np
from comparison import accuracy, plot_accuracies
from models import Logistic, SVM, DecisionTree
from sklearn.neighbors import KNeighborsClassifier


def load_save_mnist_data():
    """
    initialization function : loads the mnist data and saves as a numpy file for future use
    """

    train_data = np.loadtxt("mnist_train.csv", delimiter=",")
    np.save("train_data", train_data)
    test_data = np.loadtxt("mnist_test.csv", delimiter=",")
    np.save("test_data", test_data)


def load_mnist():
    """
    loads the saved mnist data and parses it according to 0 and 1 images
    :return: X_train, y_train, X_test, y_test
    """
    yX_train, yX_test = np.load('train_data.npy'), np.load('test_data.npy')
    X_train, y_train = yX_train[:, 1:], yX_train[:, 0]
    X_test, y_test = yX_test[:, 1:], yX_test[:, 0]
    train_images = np.logical_or((y_train == 0), (y_train == 1))
    test_images = np.logical_or((y_test == 0), (y_test == 1))
    X_train, y_train = X_train[train_images], y_train[train_images]
    X_test, y_test = X_test[test_images], y_test[test_images]
    return X_train, y_train, X_test, y_test


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
    while True:  # must have both 1 and 0 in y
        indexes = np.random.choice(len(X_train), m)
        X, y = X_train[indexes], y_train[indexes]
        if 0 in y and 1 in y:
            break
    return X, y


def q14():
    """
    plots accuracy as a function of samples for various classifiers, as well as returns the mean fit time
    :return: accuracies and times
    """
    m_vals = [50, 100, 300, 500]
    X_train, y_train, X_test, y_test = load_mnist()

    all_accuracies = []
    all_times = []
    models = {"Logistic": Logistic(), "SVM": SVM(), "TREE": DecisionTree(), "KNN": KNeighborsClassifier(n_neighbors=6)}
    repeat = 50
    for m in m_vals:
        acc_dict = {"Logistic": [], "SVM": [], "TREE": [], "KNN": []}
        time_dict = {"Logistic": [], "SVM": [], "TREE": [], "KNN": []}
        for i in range(repeat):
            X, y = get_Xy_until_good(m, X_train, y_train)
            for model in models:
                start = time.time()
                models[model].fit(X, y)
                end = time.time() - start
                y_hat = models[model].predict(X_test)
                time_dict[model].append(end)
                acc_dict[model].append(accuracy(y_test, y_hat))
        all_accuracies.append(acc_dict)
        all_times.append(time_dict)

    for d in all_times:
        for model in d:
            d[model] = np.mean(np.array(d[model]))
    for d in all_accuracies:
        for model in d:
            d[model] = np.mean(np.array(d[model]))
    return all_accuracies, all_times

