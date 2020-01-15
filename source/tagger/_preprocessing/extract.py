from sklearn.base import BaseEstimator, TransformerMixin
from functools import reduce

class ExtractText(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, add_time_of_day=False):
        if columns is None:
            columns = ['description']

        self.columns = columns
        self.add_time_of_day = add_time_of_day

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cols = [X[col].astype(str) for col in self.columns]
        if self.add_time_of_day:
            # TODO: Consider rounding to even hours
            # Slightly iffy mixing of datetime.time and str in fillna
            cols = ['__TIME__' + X['time'].fillna('ALL-DAY').astype(str) + '__'] + cols
        return reduce(lambda x, y: x + ' ' + y, cols)