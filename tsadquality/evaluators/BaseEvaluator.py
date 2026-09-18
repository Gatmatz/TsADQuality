import numpy as np


class BaseEvaluator:
    """Shared interface for scoring a detector's output against ground-truth labels.

    Subclasses receive the raw binary labels (`y_true`, 1 = anomaly) and the
    detector's continuous `decision_scores_` (higher = more anomalous) and
    return a single float score.
    """

    @classmethod
    def evaluate(cls, y_true, decision_scores) -> float:
        raise NotImplementedError

    @staticmethod
    def _as_arrays(y_true, decision_scores):
        return (
            np.asarray(y_true, dtype=int).ravel(),
            np.asarray(decision_scores, dtype=float).ravel(),
        )
