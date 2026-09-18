class BaseDetector:
    """
    Shared minimal interface for the TSB-AD `model_wrapper`-based detectors in
    this package. TSB-AD's `run_Unsupervise_AD`/`run_Semisupervise_AD` already
    return scores aligned to the original series length internally, so
    subclasses only need to store the wrapper's output on `decision_scores_`.
    """

    def __init__(self):
        self.decision_scores_ = None

    def fit(self, X):
        raise NotImplementedError
