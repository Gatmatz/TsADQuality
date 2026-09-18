from TSB_UAD.models.feature import Window
from TSB_UAD.models.lof import LOF

from tsadquality.detectors.BaseDetector import BaseDetector


class LOFDetector(BaseDetector):
    """Wraps tsb-uad's sliding-window Local Outlier Factor detector."""

    def __init__(self, window=100, n_neighbors=20, contamination=0.1):
        super().__init__(window=window)
        self.n_neighbors = n_neighbors
        self.contamination = contamination

    def _raw_scores(self, X):
        subsequences = Window(window=self.window).convert(X).to_numpy()
        model = LOF(n_neighbors=self.n_neighbors, contamination=self.contamination)
        model.fit(subsequences)
        return model.decision_scores_
