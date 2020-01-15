from sklearn.base import BaseEstimator

import keras.backend as K
from keras.layers import Dense, BatchNormalization, Dropout
from keras.models import Sequential
from keras.optimizers import Nadam
from keras.losses import binary_crossentropy

# Hardcoded for now
N_TOP_TAGS = 72


class MultiLayerPerceptron(BaseEstimator):
    def __init__(self, layers=None, epochs=16, batch_size=64):
        if layers is None:
            layers = [128, 64, 32]

        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        # Reclaim any previously allocated memory — note that this will
        # break any other model using keras!
        K.clear_session()
        print("NB! Cleared Keras session — previously trained models will break")


        model = Sequential()

        for i, layer in enumerate(self.layers):
            if i == 0:
                # Make sure first layer has correct dimension
                model.add(Dense(layer, input_dim=X.shape[1], activation='relu'))
            else:
                model.add(Dense(layer, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

        # Output layer
        model.add(Dense(N_TOP_TAGS, activation='sigmoid'))

        # We don't use any early stopping or checkpointing in order to
        # keep it simple
        model.compile(loss=binary_crossentropy, optimizer=Nadam())
        self.model = model

        print("Fitting model:")
        model.summary()
        model.fit(X, y, epochs=self.epochs,batch_size=self.batch_size)


    def predict(self, X):
        y_pred =  self.model.predict(X)
        return y_pred > 0.5
