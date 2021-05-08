import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from models import Perceptron, SVM

pio.renderers.default = "browser"


def true_labels_f(x, w, b):
    return np.sign(np.dot(w, x) + b)


def draw_points(m):
    """
    given an integer m, returns a pair X, y where X is a
    m over 2 matrix where each column represents an i.i.d sample from
    the distribution D=N(0,I_2),and y (m rows of +1 or -1) is its
    corresponding label, according to f(x) = sign(dot(0.3,-0,5).T,x)+1)
    :param m: num of points (int)
    :return: X,y
    """
    w = np.array([0.3, -0.5])
    b = 0.1
    X = np.random.multivariate_normal(mean=[0, 0], cov=np.eye(2), size=m)
    y = np.array([true_labels_f(x, w, b) for x in X])
    return X, y


def decision_surface(predict, xrange, yrange, density=120, dotted=False, showscale=True):
    xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange, density)
    xx, yy = np.meshgrid(xrange, yrange)
    pred = predict(np.c_[xx.ravel(), yy.ravel()])

    if dotted:
        return go.Scatter(x=xx.ravel(), y=yy.ravel(), opacity=1, mode="markers",
                          marker=dict(color=pred, size=1, reversescale=False), hoverinfo="skip",
                          showlegend=False)
    return go.Contour(x=xrange, y=yrange, z=pred.reshape(xx.shape), reversescale=False,
                      opacity=.7, connectgaps=True, hoverinfo="skip", showlegend=False, showscale=showscale)


def plot_points():
    """
    question 9
    :return:
    """
    m_vals = [70]
    for m in m_vals:
        X, y = draw_points(m)
        lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-0.1, 0.1])
        model_names = ["Perceptron", "SVM"]
        models = [Perceptron(), SVM()]
        fig = make_subplots(rows=1, cols=2, subplot_titles=[rf"$\text{{{m}}}$" for m in model_names])
        for i, m in enumerate(models):
            m.fit(X,y)
            fig.add_traces([decision_surface(m.predict, lims[0], lims[1], showscale=False),
                            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                       marker=dict(color=y,
                                                   line=dict(color="black", width=1)))],
                           rows=(i // 3) + 1, cols=(i % 3) + 1)

        fig.update_layout(title=rf"$\text{{Decision Boundaries Of Models}}$", margin=dict(t=100))
        fig.show()


plot_points()
