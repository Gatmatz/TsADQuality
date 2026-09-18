from TSB_UAD.models.matrix_profile import MatrixProfile

from tsadquality.detectors.BaseDetector import BaseDetector


class MatrixProfileDetector(BaseDetector):
    """Wraps tsb-uad's STUMPY-based Matrix Profile detector."""

    def _raw_scores(self, X):
        model = MatrixProfile(window=self.window)
        model.fit(X)
        return model.decision_scores_
