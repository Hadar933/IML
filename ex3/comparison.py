import numpy as np


def draw_points(m):
    """
    given an integer m, returns a pair X, y where X is a
    m over 2 matrix where each column represents an i.i.d sample from
    the distribution D=N(0,I_2),and y (m rows of +1 or -1) is its
    corresponding label, according to f(x) = sign(dot(0.3,-0,5).T,x)+1)
    :param m: num of points (int)
    :return: X,y
    """
    f = lambda x: np.sign(np.dot(np.array([0.3, -0.5]), x) + 0.1)
    X = np.random.multivariate_normal(mean=[0, 0], cov=np.eye(2), size=m)
    y = np.array([f(x) for x in X])[:, np.newaxis]
    return X, y


