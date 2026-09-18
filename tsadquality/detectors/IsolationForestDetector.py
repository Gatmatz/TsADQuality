import numpy as np
from TSB_AD.model_wrapper import run_Unsupervise_AD

from tsadquality.detectors.BaseDetector import BaseDetector


class IsolationForestDetector(BaseDetector):
    """Wraps TSB-AD's `run_Unsupervise_AD('IForest', ...)`."""

    def __init__(self, window=100, n_estimators=100, max_features=1, n_jobs=1):
        super().__init__()
        self.window = window
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.n_jobs = n_jobs

    def fit(self, X):
        data = np.asarray(X, dtype=float).reshape(-1, 1)
        self.decision_scores_ = run_Unsupervise_AD(
            "IForest",
            data,
            slidingWindow=self.window,
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
        )
        return self
