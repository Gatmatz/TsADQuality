import numpy as np

from tsadquality.evaluators.BaseEvaluator import BaseEvaluator


class ThreeSigmaPrecisionEvaluator(BaseEvaluator):
    """Precision after binarizing decision scores with the classic mu +/- 3*sigma rule:
    a point is flagged anomalous when it falls outside [mean - 3*std, mean + 3*std]
    of the decision scores' own distribution.
    """

    n_sigma: float = 3.0

    @classmethod
    def evaluate(cls, y_true, decision_scores) -> float:
        from sklearn.metrics import precision_score

        y_true, decision_scores = cls._as_arrays(y_true, decision_scores)

        mean = decision_scores.mean()
        std = decision_scores.std()
        lower, upper = mean - cls.n_sigma * std, mean + cls.n_sigma * std

        y_pred = ((decision_scores < lower) | (decision_scores > upper)).astype(int)
        return float(precision_score(y_true, y_pred, zero_division=0))
