from sklearn.base import BaseEstimator, TransformerMixin

def _ngrams(xs, n_min, n_max):
    ngrams = []
    for n in range(n_min, n_max + 1):
        for i in range(len(xs) - n + 1):
            ngrams.append(' '.join(xs[i:i + n]))
    return ngrams

class NGram(BaseEstimator, TransformerMixin):
    def __init__(self, n_min, n_max=None):
        self.n_min = n_min
        self.n_max = n_max

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        n_max = self.n_max if self.n_max is not None else self.n_min
        return X.apply(lambda xs: _ngrams(xs, self.n_min, n_max))

