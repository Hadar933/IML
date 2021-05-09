import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from abc import ABC, abstractmethod


class Classifier(ABC):
    """
    abstract Classifier class that all classes inherit from
    """

    def __init__(self):
        self.model = None

    @abstractmethod
    def fit(self, X, y):
        """
        this method learns the parameters of the model and stores the trained model (namely, the variables that
        define hypothesis chosen) in self.model. The method returns nothing.
        :param X: training set as X(m over d)
        :param y: (d rows of +-1)
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        predicts the label of each sample.
        :param X: unlabled test set with m' rows and d columns
        :return: vector of predicted labels y with m' rows of +1 or -1
        """
        pass

    def score(self, X, y):
        """
        :param X: unlabled test set with m' rows and d columns
        :param y: true labels - vector of predicted labels y with m' rows of +1 or -1
        :return: dictionary with relevant scores
        """
        self.fit(X, y)
        y_hat = self.predict(X)
        m = len(y_hat)
        positive = sum([1 for item in y if item == 1])
        negative = sum([1 for item in y if item == -1])
        false_positive = sum([1 for i in range(m) if y_hat[i] == 1 and y[i] == -1])
        false_negative = sum([1 for i in range(m) if y_hat[i] == -1 and y[i] == 1])
        true_positive = sum([1 for i in range(m) if y_hat[i] == 1 and y[i] == 1])
        true_negative = sum([1 for i in range(m) if y_hat[i] == -1 and y[i] == -1])

        return {
            'num_samples': X.shape[0],
            'error': (false_positive + false_negative) / (positive + negative),
            'accuracy': (true_positive + true_negative) / (positive + negative),
            'FPR': false_positive / negative,
            'TPR': true_positive / positive,
            'precision': true_positive / (true_positive + false_positive),
            'specificity': true_negative / negative
        }


class Perceptron(Classifier):

    def __init__(self):
        super().__init__()
        self._w = None  # module weights initialization

    def fit(self, X, y):
        new_X = np.insert(X, 0, 1, axis=1)  # add column of ones to the beginning of X (for the non-homogenous case)
        m, d = new_X.shape[0], new_X.shape[1]
        self._w = np.zeros(d)
        changed_w = True  # if we didnt enter the inner if for all i in m, we can return
        while changed_w:
            changed_w = False
            for i in range(m):
                if y[i] * np.dot(self._w, new_X[i]) <= 0:
                    changed_w = True
                    self._w += (y[i] * new_X[i])

    def predict(self, X):
        return np.sign(np.insert(X, 0, 1, axis=1) @ self._w)

    def score(self, X, y):
        return super().score(X, y)


class LDA(Classifier):
    def __init__(self):
        # two lists (one for +1 and the other for -1) of delta function for every row in X
        super().__init__()
        self.d_pos = None  # delta plus function
        self.d_neg = None  # delta neg function

    def fit(self, X, y):
        new_X = np.insert(X, 0, 1, axis=1)
        X_pos = new_X[y == 1]
        X_neg = new_X[y == -1]

        mu_pos = np.array([np.mean(row) for row in X_pos.T])
        mu_neg = np.array([np.mean(row) for row in X_neg.T])
        sigma_inv = np.linalg.pinv(np.cov(new_X.T))
        prob_pos = sum([1 for i in y if i == 1]) / len(y)
        prob_neg = sum([1 for i in y if i == -1]) / len(y)

        self.d_pos = lambda x: x.T @ sigma_inv @ mu_pos - 0.5 * mu_pos.T @ sigma_inv @ mu_pos + np.log(prob_pos)
        self.d_neg = lambda x: x.T @ sigma_inv @ mu_neg - 0.5 * mu_neg.T @ sigma_inv @ mu_neg + np.log(prob_neg)

    def predict(self, X):
        m = X.shape[0]
        new_X = np.insert(X, 0, 1, axis=1)

        d_pos_arr = [self.d_pos(x) for x in new_X]
        d_neg_arr = [self.d_neg(x) for x in new_X]
        argmax_index = [np.argmax([d_pos_arr[i], d_neg_arr[i]]) for i in range(m)]
        return [-1 if argmax_index[i] == 0 else 1 for i in range(m)]

    def score(self, X, y):
        return super().score(X, y)


class SVM(Classifier):
    def __init__(self):
        super().__init__()
        self.model = SVC(C=1e10, kernel='linear')

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)


class Logistic(Classifier):
    def __init__(self):
        super().__init__()
        self.model = LogisticRegression(solver='liblinear')

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)


class DecisionTree(Classifier):
    def __init__(self):
        super().__init__()
        self.model = DecisionTreeClassifier(max_depth=5)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
