import numpy as np

from .base import BaseCorruptor

MECHANISMS = ("MCAR", "MNAR_extreme", "MNAR_high")


class MissingValueInjector(BaseCorruptor):
    """
    Injects missing values (NaN) at scattered points based on the targeting criteria.

    mechanism controls how points are selected among the targetable indices:
      - 'MCAR' (Missing Completely At Random): uniformly random selection.
      - 'MNAR_extreme' (Missing Not At Random): weighted random selection without
        replacement, with probability proportional to |z-score|, i.e. extreme values
        in EITHER tail are more likely to go missing (e.g. sensor overload).
      - 'MNAR_high' (Missing Not At Random): weighted random selection without
        replacement, with probability proportional to max(z-score, 0) + 0.01, i.e.
        only high positive values are disproportionately likely to go missing
        (e.g. a sensor that caps out at the top); the small floor keeps every point
        with some chance of being selected.
    """

    def inject(self, fraction, mechanism="MCAR"):
        target_indices = self.get_target_indices()
        count = min(int(len(self.df) * fraction), len(target_indices))
        if count == 0:
            return self.df

        if mechanism in ('MNAR_extreme', 'MNAR_high'):
            target_indices = np.array(target_indices)
            values = self.df.loc[target_indices, self.value_col].to_numpy('float')
            z = (values - np.nanmean(values)) / (np.nanstd(values) + 1e-10)
            if mechanism == 'MNAR_extreme':
                weights = np.abs(z)
            else:
                weights = np.maximum(z, 0) + 0.01
            weights = np.nan_to_num(weights, nan=0.0)
            weights = weights / weights.sum()
            positions = self.rng.choice(len(target_indices), size=count, replace=False, p=weights)
            indices = target_indices[positions]
        else:
            indices = self.get_injectable_indices(count)

        if not np.issubdtype(self.df[self.value_col].dtype, np.floating):
            self.df[self.value_col] = self.df[self.value_col].astype(float)

        self.df.loc[indices, self.value_col] = np.nan
        self.record_corruption('point_missing', indices, {'fraction': fraction, 'mechanism': mechanism})
        return self.df


class BurstMissingInjector(BaseCorruptor):
    """
    Simulates packet loss by dropping contiguous chunks of data (NaN).

    mechanism controls how burst START positions are selected among the targetable
    indices that leave room for a full burst:
      - 'MCAR' (Missing Completely At Random): uniformly random selection.
      - 'MNAR_extreme' (Missing Not At Random): weighted random selection without
        replacement, with probability proportional to the mean |z-score| over the
        candidate burst window, i.e. bursts starting near extreme values in EITHER
        tail are more likely (e.g. sensor overload).
      - 'MNAR_high' (Missing Not At Random): weighted random selection without
        replacement, with probability proportional to the mean of max(z-score, 0)
        over the candidate burst window, plus a 0.01 floor, i.e. bursts starting near
        high positive values are disproportionately likely (e.g. a sensor that caps
        out at the top); the small floor keeps every window with some chance of
        being selected.
    """

    def inject(self, fraction, num_bursts, mechanism="MCAR"):
        total_points = int(len(self.df) * fraction)
        burst_length = max(1, total_points // num_bursts)

        if mechanism in ('MNAR_extreme', 'MNAR_high'):
            n = len(self.df)
            candidates = np.array([
                idx for idx in self.get_target_indices() if idx + burst_length <= n
            ])
            if len(candidates) == 0:
                return self.df
            count = min(num_bursts, len(candidates))

            values = self.df[self.value_col].to_numpy('float')
            z = (values - np.nanmean(values)) / (np.nanstd(values) + 1e-10)
            scores = np.abs(z) if mechanism == 'MNAR_extreme' else np.maximum(z, 0)
            scores = np.nan_to_num(scores, nan=0.0)

            # Mean score over each candidate burst window, via a prefix sum.
            cumsum = np.concatenate([[0.0], np.cumsum(scores)])
            weights = (cumsum[candidates + burst_length] - cumsum[candidates]) / burst_length
            if mechanism == 'MNAR_high':
                weights = weights + 0.01
            weights = weights / weights.sum()

            target_indices = self.rng.choice(candidates, size=count, replace=False, p=weights)
        else:
            target_indices = self.get_injectable_indices(num_bursts)

        if len(target_indices) == 0:
            return self.df

        expanded_indices = set()
        for start_idx in target_indices:
            for i in range(burst_length):
                if start_idx + i < len(self.df):
                    expanded_indices.add(start_idx + i)

        if not np.issubdtype(self.df[self.value_col].dtype, np.floating):
            self.df[self.value_col] = self.df[self.value_col].astype(float)

        burst_array = np.array(sorted(expanded_indices))
        self.df.loc[burst_array, self.value_col] = np.nan
        self.record_corruption('burst_missing', burst_array, {
            'fraction': fraction,
            'num_bursts': num_bursts,
            'burst_length': burst_length,
            'mechanism': mechanism,
        })
        return self.df
