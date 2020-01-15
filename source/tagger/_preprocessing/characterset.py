import re
from sklearn.base import BaseEstimator, TransformerMixin

def _strip_matching(s, regexp):
    s = regexp.sub(' ', s)
    s = re.sub('\s+', ' ', s).strip()
    return s

class CharacterSet(BaseEstimator, TransformerMixin):
    def __init__(self, punctuation=True, digits=False):
        # TODO: Add more classes of characters
        self.punctuation = punctuation
        self.digits = digits

    def fit(self, X, y=None):
        character_set = 'a-zåäöA-ZÅÄÖ_'
        if self.punctuation:
            character_set += '.,!?:;()'
        if self.digits:
            character_set += '0-9'

        self._regexp = re.compile(f"[^{character_set}]")

        return self

    def transform(self, X, y=None):
        return X.apply(lambda s: _strip_matching(s, self._regexp))
