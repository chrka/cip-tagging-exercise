from sklearn.base import BaseEstimator, TransformerMixin

from nltk.tokenize import WordPunctTokenizer, WhitespaceTokenizer


class Tokenize(BaseEstimator, TransformerMixin):
    def __init__(self, method='word_punct'):
        self.method = method

    def fit(self, X, y=None):
        if self.method == 'whitespace':
            self._tokenizer = WhitespaceTokenizer()
        elif self.method == 'word_punct':
            self._tokenizer = WordPunctTokenizer()
        else:
            # TODO: Better error
            raise NotImplementedError('Unknown tokenizer')

        return self

    def transform(self, X, y=None):
        return X.apply(lambda s: self._tokenizer.tokenize(s))