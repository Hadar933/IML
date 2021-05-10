import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from models import Perceptron, SVM, LDA

pio.renderers.default = "browser"


def true_labels_f(X):
    w = np.array([0.3, -0.5])
    b = 0.1
    return np.array([np.sign(np.dot(w, x) + b) for x in X])


def draw_points(m):
    """
    given an integer m, returns a pair X, y where X is a
    m over 2 matrix where each column represents an i.i.d sample from
    the distribution D=N(0,I_2),and y (m rows of +1 or -1) is its
    corresponding label, according to f(x) = sign(dot(0.3,-0,5).T,x)+1)
    :param m: num of points (int)
    :return: X,y
    """
    X = np.random.multivariate_normal(mean=[0, 0], cov=np.eye(2), size=m)
    y = true_labels_f(X)
    return X, y


def decision_surface(predict_func, xrange, yrange, density=100):
    xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange, density)
    xx, yy = np.meshgrid(xrange, yrange)
    X = np.c_[xx.ravel(), yy.ravel()]
    return go.Contour(x=xrange, y=yrange, z=predict_func(X).reshape(xx.shape), opacity=.4, showscale=False)


def plot_points():
    """
    question 9
    :return:
    """
    m_vals = [5, 10, 15, 25, 70]
    for m in m_vals:
        X, y = draw_points(m)
        lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-0.1, 0.1])
        model_names = ["True labels", "Perceptron", "SVM"]
        models = ["True label", Perceptron(), SVM()]
        fig = make_subplots(rows=1, cols=3, subplot_titles=[m for m in model_names])
        for i, model in enumerate(models):
            if i == 0:  # handling the true labels differently
                func = true_labels_f
            else:  # for perceptron and svm
                func = model.predict
                model.fit(X, y)
            fig.add_traces([decision_surface(func, lims[0], lims[1]),
                            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                       marker=dict(color=y, line=dict(color="black", width=1)))],
                           rows=(i // 3) + 1, cols=(i % 3) + 1)

        fig.update_layout(title=rf"$\text{{Decision Boundaries Of Models with {m} samples}}$", margin=dict(t=100))
        fig.show()


def get_Xy_until_good(m):
    """
    loops until y has both 1 and -1
    :param m: number of sample
    :return: X,y
    """
    while True:  # must have both +1 and -1 in y
        X, y = draw_points(m)
        if 1 in y and -1 in y:
            break
    return X, y


def accuracy(y, y_hat):
    """
    :param y: true values
    :param y_hat: response vector
    :return: the accuracy of y (ratio between number of corret values / total size)
    """
    return sum([1 for i in range(len(y)) if y[i] == y_hat[i]]) / len(y)


def q10():
    """
    calculate the accuracy rates for perceptron, svm and lda classifiers
    :return: accuracy data
    """
    m_vals = [5, 10, 15, 25, 70]
    all_accuracies = []
    models = {"Perceptron": Perceptron(), "SVM": SVM(), "LDA": LDA()}
    k = 10000
    repeat = 500
    for m in m_vals:
        acc_dict = {"Perceptron": [], "SVM": [], "LDA": []}
        for i in range(repeat):
            X_train, y_train = get_Xy_until_good(m)
            X_test, y_test = get_Xy_until_good(k)
            for model in models:
                models[model].fit(X_train, y_train)
                y_hat = models[model].predict(X_test)
                acc_dict[model].append(accuracy(y_test, y_hat))
        all_accuracies.append(acc_dict)

    for dictionary in all_accuracies:
        for model in dictionary:
            dictionary[model] = np.mean(np.array(dictionary[model]))
    return all_accuracies


def plot_accuracies(acc, x_data, names):
    y_data = [[acc[i][model_name] for i in range(len(acc))] for model_name in names]
    fig = go.Figure()
    for i in range(len(names)):
        fig.add_trace(go.Scatter(x=x_data, y=y_data[i], mode='lines+markers', name=names[i]))
    fig.update_layout(title=rf"$\text{{Mean Accuracy as a function of m samples}}$")
    fig.update_xaxes(title_text=rf"$\text{{samples (m)}}$")
    fig.update_yaxes(title_text=rf"$\text{{Mean Accuracy}}$")
    fig.show()

