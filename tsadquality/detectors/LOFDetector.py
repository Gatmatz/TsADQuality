import numpy as np
from TSB_AD.model_wrapper import run_Unsupervise_AD

from tsadquality.detectors.BaseDetector import BaseDetector


class LOFDetector(BaseDetector):
    """Wraps TSB-AD's `run_Unsupervise_AD('LOF', ...)`."""

    def __init__(self, window=100, n_neighbors=20, metric="minkowski", n_jobs=1):
        super().__init__()
        self.window = window
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X):
        data = np.asarray(X, dtype=float).reshape(-1, 1)
        self.decision_scores_ = run_Unsupervise_AD(
            "LOF",
            data,
            slidingWindow=self.window,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            n_jobs=self.n_jobs,
        )
        return self
