import numpy as np
import pandas as pd

from tsadquality.reproducibility import ReproducibleOperations


class BaseCorruptor:
    """
    Core engine handling DataFrame tracking, history logging, and targeting logic.

    Subclasses implement a single injection technique by overriding `inject()`; every
    subclass shares the same targeting, bookkeeping, and reporting behavior defined here.

    Randomness is sourced from `ReproducibleOperations`, so callers must call
    `ReproducibleOperations.set_random_seed(...)` before injecting for reproducibility.

    corruption_target controls WHERE corruption is applied relative to anomalies:

        'global'              — All points (no position filtering)
        'only_normal'         — Only normal (non-anomaly) points
        'overlapping_anomaly' — Only the anomaly points themselves
        'near_anomaly'        — Normal points within ±window_size of any anomaly
        'before_anomaly'      — Normal points in [anomaly - window_size, anomaly - 1]
        'after_anomaly'       — Normal points in [anomaly + 1, anomaly + window_size]
        'far_from_anomaly'    — Normal points outside ±window_size of all anomalies

    Example with window_size=3 and an anomaly at index 10:

        ... [7 8 9] [10] [11 12 13] ...
             before  anom   after
             |----- near ---------|
        far                          far
    """

    def __init__(self, df, value_col='value', label_col="is_anomaly", corruption_target='global', window_size=50):
        self.df = df.copy()
        self.df_original = df.copy()
        self.value_col = value_col
        self.label_col = label_col
        self.corruption_target = corruption_target
        self.window_size = window_size
        self.history = []
        self.corruption_mask = pd.Series(False, index=self.df.index)
        self.std = self.df[self.value_col].std()
        if self.std == 0 or np.isnan(self.std):
            self.std = 1.0

    def inject(self, **params):
        """Apply this injector's corruption to `self.df` and record it. Must be overridden."""
        raise NotImplementedError

    def get_target_indices(self):
        """Returns indices where corruption can be applied based on corruption_target."""
        all_indices = self.df.index.tolist()
        anomaly_indices = self.df[self.df[self.label_col] == 1].index.tolist()
        normal_indices = self.df[self.df[self.label_col] == 0].index.tolist()
        n = len(self.df)

        if self.corruption_target == 'global':
            return all_indices
        elif self.corruption_target == 'only_normal':
            return normal_indices
        elif self.corruption_target == 'overlapping_anomaly':
            return anomaly_indices
        elif self.corruption_target == 'before_anomaly':
            # Normal points in [anomaly - window_size, anomaly - 1]
            before_indices = set()
            for idx in anomaly_indices:
                for i in range(max(0, idx - self.window_size), idx):
                    before_indices.add(i)
            return list(before_indices.intersection(normal_indices))
        elif self.corruption_target == 'after_anomaly':
            # Normal points in [anomaly + 1, anomaly + window_size]
            after_indices = set()
            for idx in anomaly_indices:
                for i in range(idx + 1, min(n, idx + self.window_size + 1)):
                    after_indices.add(i)
            return list(after_indices.intersection(normal_indices))
        elif self.corruption_target == 'near_anomaly':
            near_indices = set()
            for idx in anomaly_indices:
                for i in range(max(0, idx - self.window_size), min(n, idx + self.window_size + 1)):
                    near_indices.add(i)
            # Only corrupt normal points near anomalies
            return list(near_indices.intersection(normal_indices))
        elif self.corruption_target == 'far_from_anomaly':
            near_indices = set()
            for idx in anomaly_indices:
                for i in range(max(0, idx - self.window_size), min(n, idx + self.window_size + 1)):
                    near_indices.add(i)
            far_indices = set(all_indices) - near_indices
            return list(far_indices.intersection(normal_indices))

        return all_indices

    def get_injectable_indices(self, required_amount):
        target_indices = self.get_target_indices()
        required_amount = min(required_amount, len(target_indices))
        return ReproducibleOperations.sample_from(
            target_indices, how_many=required_amount, sampling_with_replacement=False
        )

    def record_corruption(self, corruption_type, indices, params):
        if len(indices) > 0:
            self.history.append({
                'type': corruption_type,
                'count': len(indices),
                'first_idx': int(min(indices)),
                'last_idx': int(max(indices)),
                'params': params
            })
            self.corruption_mask.loc[indices] = True

    def get_corrupted_df(self):
        return self.df
