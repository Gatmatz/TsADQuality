import numpy as np


class BaseDetector:
    """
    Shared alignment logic for the tsb-uad model wrappers in this package.

    Subclasses fit an unsupervised model on sliding-window subsequences of
    length `window` and hand back raw scores of length `n - window + 1`.
    `fit()` pads those raw scores back out to the original series length by
    repeating the first/last score, centering the window the same way
    TSB-UAD's own AE wrapper does.
    """

    def __init__(self, window=100):
        self.window = window
        self.decision_scores_ = None

    def _raw_scores(self, X):
        raise NotImplementedError

    def fit(self, X):
        X = np.asarray(X, dtype=float).ravel()
        raw_scores = np.asarray(self._raw_scores(X), dtype=float)

        n = len(X)
        head = self.window // 2
        scores = np.empty(n, dtype=float)
        scores[head:head + len(raw_scores)] = raw_scores
        scores[:head] = raw_scores[0]
        scores[head + len(raw_scores):] = raw_scores[-1]

        self.decision_scores_ = scores
        return self
