from sklearn.base import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB

class NaiveBayes(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        self._clf = OneVsRestClassifier(MultinomialNB())
        self._clf.fit(X, y)
        return self

    def predict(self, X):
        return self._clf.predict(X)