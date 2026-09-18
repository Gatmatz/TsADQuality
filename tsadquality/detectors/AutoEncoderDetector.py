from TSB_UAD.models.AE import AE_MLP2

from tsadquality.detectors.BaseDetector import BaseDetector


class AutoEncoderDetector(BaseDetector):
    """
    Wraps tsb-uad's AE_MLP2 autoencoder detector.

    Unlike the other wrappers, AE_MLP2 already aligns its output to the full
    series length internally, so `fit()` is overridden instead of going
    through `BaseDetector`'s subsequence padding.
    """

    def __init__(self, window=100, epochs=10, verbose=0):
        super().__init__(window=window)
        self.epochs = epochs
        self.verbose = verbose

    def fit(self, X):
        model = AE_MLP2(slidingWindow=self.window, epochs=self.epochs, verbose=self.verbose)
        model.fit(X, X)
        self.decision_scores_ = model.decision_scores_
        return self
