from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import SnowballStemmer


def _stem_tokens(tokens, stemmer):
    return [stemmer.stem(token) for token in tokens]


class Stemming(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.stemmer = SnowballStemmer('swedish')

        return self

    def transform(self, X, y=None):
        return X.apply(lambda ts: _stem_tokens(ts, self.stemmer))
