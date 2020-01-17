from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords

def _filter_stopwords(tokens, stopwords):
    return [token for token in tokens if token not in stopwords]

class Stopwords(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.stopwords = set(stopwords.words('swedish'))
        return self

    def transform(self, X, y=None):
        return X.apply(lambda ts: _filter_stopwords(ts, self.stopwords))