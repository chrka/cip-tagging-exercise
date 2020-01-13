from sklearn.base import BaseEstimator, TransformerMixin

class Lowercase(BaseEstimator, TransformerMixin):
    """Transform Pandas series of strings to lowercase"""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.str.lower()