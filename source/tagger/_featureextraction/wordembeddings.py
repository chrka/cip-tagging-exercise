import fasttext
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MeanWordEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, model_path="/tmp/wordvectors.bin"):
        self.model_path = model_path

    def fit(self, X, y=None):
        # TODO: Consider generating word vectors from X
        self.model = fasttext.load_model(self.model_path)
        return self

    def transform(self, X, y=None):
        embeddings = np.zeros((len(X), self.model.get_dimension()))
        for i, tokens in enumerate(X):
            if len(tokens) > 0:
                embeddings[i] = np.mean(list(map(lambda t: self.model[t],
                                                 tokens)),
                                        axis=0)

        return embeddings


class SumWordEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, model_path="/tmp/wordvectors.bin"):
        self.model_path = model_path

    def fit(self, X, y=None):
        # TODO: Consider generating word vectors from X
        self.model = fasttext.load_model(self.model_path)
        return self

    def transform(self, X, y=None):
        embeddings = np.zeros((len(X), self.model.get_dimension()))
        for i, tokens in enumerate(X):
            if len(tokens) > 0:
                embeddings[i] = np.sum(list(map(lambda t: self.model[t],
                                                tokens)),
                                       axis=0)

        return embeddings
