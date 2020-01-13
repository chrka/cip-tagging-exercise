from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

def _identity(x):
    return x


class BagOfWords(BaseEstimator, TransformerMixin):
    """Transforms lists of tokens into vectors."""
    def __init__(self, binary=False):
        self.binary = binary
        pass

    def fit(self, X, y=None):
        self._vectorizer = CountVectorizer(
            binary=self.binary,
            tokenizer=_identity,
            token_pattern=None,
            preprocessor=_identity,
            ngram_range=(1, 1))
        self._vectorizer.fit(X)
        return self

    def transform(self, X, y=None):
        return self._vectorizer.transform(X)

