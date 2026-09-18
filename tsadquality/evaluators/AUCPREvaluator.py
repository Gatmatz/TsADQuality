import numpy as np

from tsadquality.evaluators.BaseEvaluator import BaseEvaluator


class AUCPREvaluator(BaseEvaluator):
    """Area under the Precision-Recall curve (average precision), computed directly
    on the continuous decision scores."""

    @classmethod
    def evaluate(cls, y_true, decision_scores) -> float:
        from sklearn.metrics import average_precision_score

        y_true, decision_scores = cls._as_arrays(y_true, decision_scores)
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(average_precision_score(y_true, decision_scores))
    