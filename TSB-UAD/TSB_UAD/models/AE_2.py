"""
AE_MLP2 with separate fit/predict — allows training once and scoring multiple times.

Identical architecture and logic to AE.py, only difference:
  - model saved as self.model_ (not local variable)
  - predict() method for scoring new data without retraining
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers


class AE_MLP2(object):
    """
    Implementation of AE_MLP2 with reusable predict.

    Parameters
    ----------
    slidingWindow : int
        Subsequence length to analyze.
    epochs : int, (default=10)
        Number of epochs for the training phase

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The anomaly score.
    model_ : keras Sequential
        The trained autoencoder (available after fit).
    """
    def __init__(self, slidingWindow=100, epochs=10, verbose=0):
        self.slidingWindow = slidingWindow
        self.epochs = epochs
        self.verbose = verbose
        self.model_name = 'AE_MLP2'

    def fit(self, X_clean, X_dirty, ratio=0.15):
        """Train on X_clean and score X_dirty.

        Parameters
        ----------
        X_clean : numpy array of shape (n_samples, )
            The input training samples (clean data).
        X_dirty : numpy array of shape (n_samples, )
            The input testing samples (corrupted data).
        ratio : float, ([0,1])
            The ratio for the train validation split

        Returns
        -------
        self : object
            Fitted estimator.
        """
        TIME_STEPS = self.slidingWindow

        X_train = self.create_dataset(X_clean, TIME_STEPS)
        X_train = MinMaxScaler().fit_transform(X_train.T).T

        self.model_ = Sequential()
        self.model_.add(layers.Dense(32, activation='relu'))
        self.model_.add(layers.BatchNormalization())
        self.model_.add(layers.Dense(16, activation='relu'))
        self.model_.add(layers.BatchNormalization())
        self.model_.add(layers.Dense(32, activation='relu'))
        self.model_.add(layers.Dense(TIME_STEPS, activation='relu'))

        self.model_.compile(optimizer='adam', loss='mse')

        self.model_.fit(X_train, X_train,
                        epochs=self.epochs,
                        batch_size=64,
                        shuffle=False,
                        validation_split=ratio,
                        verbose=self.verbose,
                        callbacks=[EarlyStopping(monitor="val_loss",
                                                 verbose=self.verbose,
                                                 patience=5, mode="min")])

        self.predict(X_dirty)
        return self

    def predict(self, X_dirty):
        """Score new data using the already-trained model.

        Parameters
        ----------
        X_dirty : numpy array of shape (n_samples, )
            The input samples to score.

        Returns
        -------
        self : object
            Updates decision_scores_ in place.
        """
        TIME_STEPS = self.slidingWindow
        X_test = self.create_dataset(X_dirty, TIME_STEPS)
        X_test = MinMaxScaler().fit_transform(X_test.T).T

        test_predict = self.model_.predict(X_test, verbose=0)
        test_mae_loss = np.mean(np.abs(test_predict - X_test), axis=1)
        nor_test_mae_loss = MinMaxScaler().fit_transform(
            test_mae_loss.reshape(-1, 1)).ravel()

        score = np.zeros(len(X_dirty))
        score[self.slidingWindow//2:self.slidingWindow//2+len(test_mae_loss)] = nor_test_mae_loss
        score[:self.slidingWindow//2] = nor_test_mae_loss[0]
        score[self.slidingWindow//2+len(test_mae_loss):] = nor_test_mae_loss[-1]

        self.decision_scores_ = score
        return self

    def create_dataset(self, X, time_steps):
        output = []
        for i in range(len(X) - time_steps + 1):
            output.append(X[i : (i + time_steps)])
        return np.stack(output)
