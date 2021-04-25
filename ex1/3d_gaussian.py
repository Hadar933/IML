"""
Introduction to Machine Learning
Exercise 1
Hadar Sharvit - 208287599
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr

mean = [0, 0, 0]
cov = np.eye(3)
x_y_z = np.random.multivariate_normal(mean, cov, 50000).T


def get_orthogonal_matrix(dim):
    H = np.random.randn(dim, dim)
    Q, R = qr(H)
    return Q, R


def plot_3d(x_y_z):
    '''
    plot points in 3D
    :param x_y_z: the points. numpy array with shape: 3 X num_samples (first dimension for x, y, z
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_y_z[0], x_y_z[1], x_y_z[2], s=1, marker='.', depthshade=False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def plot_2d(x_y):
    '''
    plot points in 2D
    :param x_y_z: the points. numpy array with shape: 2 X num_samples (first dimension for x, y
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_y[0], x_y[1], s=1, marker='.')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def q11():
    plot_3d(x_y_z)
    plt.title("Normal distribution with $\mu=0,\sigma=1$")
    # plt.savefig("q11")


def q12():
    scaling_mat = np.diag([0.1, 0.5, 2])
    new_xyz = np.dot(scaling_mat, x_y_z)
    plot_3d(new_xyz)
    plt.title("Transformed Data")
    plt.show()
    cov_matrix = np.cov(new_xyz)
    print(cov_matrix)
    # plt.savefig("q12")


def q13():
    scaling_mat = np.diag([0.1, 0.5, 2])
    new_xyz = np.dot(scaling_mat, x_y_z)
    ortho = get_orthogonal_matrix(3)[0]
    rand = np.dot(ortho, new_xyz)
    cov_matrix = np.cov(rand)
    print(cov_matrix)
    plot_3d(rand)
    plt.title("Data multiplied with random Orthogonal matrix")
    # plt.savefig("q13")


def q14():
    projection = x_y_z[0:-1]  # removing z data
    plot_2d(projection)
    plt.title('Projection of 3D data to 2D plane')
    # plt.savefig("q14")


def q15():
    index_to_keep = (-0.4 < x_y_z[2]) & (x_y_z[2] < 0.1)
    data = x_y_z[:, index_to_keep]
    projection = data[0:-1]
    plot_2d(projection)
    plt.title('Projection of 3D data to 2D plane, given -0.4 < z < 0.1')
    # plt.savefig("q15")


def q16a():
    data = np.random.binomial(1, 0.25, (100000, 1000))
    for i in range(5):
        throws = data[i, :]
        tosses = np.arange(1, 1001)  # x-axis
        mean_lst = [np.mean(throws[0:m]) for m in tosses]  # y -axis
        x_y = [tosses, mean_lst]
        plt.scatter(x_y[0], x_y[1], s=2, marker='.')
    plt.legend([f'throws #{i}' for i in range(1, 6)], markerscale=10)
    plt.title("mean as a function of #throws")
    plt.xlabel("throws")
    plt.ylabel("mean")
    # plt.savefig("q16a")


def q16bc():
    epsilon = [0.5, 0.25, 0.1, 0.01, 0.001]
    data = np.random.binomial(1, 0.25, (100000, 1000))
    probability = 0.25
    Expectancy = probability
    Variance = probability * (1 - probability)
    for e in epsilon:
        mean_matrix = np.cumsum(data, axis=1) / np.arange(1, 1001)
        cheb, hoeff = [], []
        for m in range(1, 1001):
            hoeff_bound = Variance / (m * e * e)
            cheb_bound = 2 * np.exp(-2 * m * e * e)
            hoeff.append(min(hoeff_bound, 1))
            cheb.append(min(cheb_bound, 1))
        condition = abs(mean_matrix - Expectancy) >= e
        percentage = np.count_nonzero(condition, axis=0) / 100000
        plt.plot(hoeff, label="Hoeffding")
        plt.plot(cheb, label="Chebyshev")
        plt.plot(percentage, label="Percentage")
        plt.legend()
        plt.ylabel('Bound')
        plt.xlabel('Throws (m)')
        plt.title(f"$\epsilon$ = {e}")
        # plt.savefig(f"q16_e={e}.png")
        plt.show()
        plt.close()

q16bc()