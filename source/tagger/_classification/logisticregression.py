from sklearn.base import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression as SKLearnLR

class LogisticRegression(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        # TODO: Consider pure (no penalty) LR
        # TODO: Look into convergence issues with other (faster) solvers
        #       (Also running into some version-related issues.)
        self._clf = OneVsRestClassifier(SKLearnLR(solver='liblinear'))
        self._clf.fit(X, y)
        return self

    def predict(self, X):
        return self._clf.predict(X)