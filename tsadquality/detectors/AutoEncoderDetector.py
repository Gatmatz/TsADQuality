import numpy as np
from TSB_AD.model_wrapper import run_Semisupervise_AD

from tsadquality.detectors.BaseDetector import BaseDetector


class AutoEncoderDetector(BaseDetector):
    """Wraps TSB-AD's `run_Semisupervise_AD('AutoEncoder', ...)`.

    TSB-AD only ships a semisupervised AutoEncoder (fit on `data_train`, scored
    on `data_test`); we fit and score on the same series (train == test) to
    match this package's other, purely unsupervised detectors.
    """

    def __init__(self, window=100, hidden_neurons=None, n_jobs=1):
        super().__init__()
        self.window = window
        self.hidden_neurons = hidden_neurons or [64, 32]
        self.n_jobs = n_jobs

    def fit(self, X):
        data = np.asarray(X, dtype=float).reshape(-1, 1)
        self.decision_scores_ = run_Semisupervise_AD(
            "AutoEncoder",
            data,
            data,
            window_size=self.window,
            hidden_neurons=self.hidden_neurons,
            n_jobs=self.n_jobs,
        )
        return self
