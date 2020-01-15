from sklearn.base import BaseEstimator, TransformerMixin

class SparseToDense(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.todense()