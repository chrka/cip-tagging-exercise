from sklearn.base import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression as SKLearnLR

class LogisticRegression(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        self._clf = OneVsRestClassifier(SKLearnLR())
        self._clf.fit(X, y)
        return self

    def predict(self, X):
        return self._clf.predict(X)