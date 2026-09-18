import numpy as np
from TSB_AD.model_wrapper import run_Unsupervise_AD

from tsadquality.detectors.BaseDetector import BaseDetector


class MatrixProfileDetector(BaseDetector):
    """Wraps TSB-AD's `run_Unsupervise_AD('MatrixProfile', ...)`.

    Unlike the other wrappers, MatrixProfile takes a `periodicity` (an ACF-peak
    rank fed into `find_length_rank` internally) instead of an explicit window,
    so it uses our precomputed `ts_metadata.periodicity` rather than
    `ts_metadata.window_length`.
    """

    def __init__(self, periodicity=1, n_jobs=1):
        super().__init__()
        self.periodicity = periodicity
        self.n_jobs = n_jobs

    def fit(self, X):
        data = np.asarray(X, dtype=float).reshape(-1, 1)
        self.decision_scores_ = run_Unsupervise_AD(
            "MatrixProfile", data, periodicity=self.periodicity, n_jobs=self.n_jobs
        )
        return self
