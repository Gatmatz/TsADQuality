from TSB_UAD.models.feature import Window
from TSB_UAD.models.iforest import IForest

from tsadquality.detectors.BaseDetector import BaseDetector


class IsolationForestDetector(BaseDetector):
    """Wraps tsb-uad's sliding-window Isolation Forest detector."""

    def __init__(self, window=100, n_estimators=100, contamination=0.1, random_state=None):
        super().__init__(window=window)
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state

    def _raw_scores(self, X):
        subsequences = Window(window=self.window).convert(X).to_numpy()
        model = IForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        model.fit(subsequences)
        return model.decision_scores_
