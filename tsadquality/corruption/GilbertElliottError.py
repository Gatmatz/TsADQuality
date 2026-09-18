import numpy as np

from tsadquality.reproducibility import ReproducibleOperations

from .BaseError import BaseCorruptor


class GilbertElliottInjector(BaseCorruptor):
    """
    Injects missing values using bursty gaps produced by a 2-state Markov chain
    (Gilbert-Elliott model). The model alternates between Good (no corruption) and
    Bad (values set to NaN) states. Transition probabilities control burst
    frequency and duration:
      - a (p_good_to_bad, α): probability of entering a burst. Higher = more frequent bursts.
      - b (p_bad_to_good, β): probability of leaving a burst. Lower = longer bursts.

    Expected corruption rate ≈ α / (α + β)
    Expected avg burst length ≈ 1 / β
    """

    def inject(self, a, b):
        n = len(self.df)
        target_indices = set(self.get_target_indices())

        # Simulate Markov chain
        state = 0  # 0 = Good, 1 = Bad
        bad_indices = []

        for i in range(n):
            if i not in target_indices:
                continue

            if state == 0:  # Good state
                if ReproducibleOperations.uniform(low=0.0, high=1.0) < a:
                    state = 1
            else:  # Bad state
                if ReproducibleOperations.uniform(low=0.0, high=1.0) < b:
                    state = 0

            if state == 1:
                bad_indices.append(i)

        if len(bad_indices) == 0:
            return self.df

        bad_indices = np.array(bad_indices)
        self.df.loc[bad_indices, self.value_col] = np.nan

        self.record_corruption('gilbert_elliott', bad_indices, {
            'a': a,
            'b': b,
        })

        return self.df
