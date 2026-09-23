import numpy as np

from .base import BaseCorruptor


class SpikeInjector(BaseCorruptor):
    """
    Injects sudden extreme spikes at isolated points: each selected point moves by
    ±multiplier standard deviations of the series.
    """

    def inject(self, fraction, multiplier):
        count = int(len(self.df) * fraction)
        target_indices = self.get_injectable_indices(count)
        if len(target_indices) == 0:
            return self.df

        if not np.issubdtype(self.df[self.value_col].dtype, np.floating):
            self.df[self.value_col] = self.df[self.value_col].astype(float)

        signs = self.rng.choice([-1, 1], size=len(target_indices))
        self.df.loc[target_indices, self.value_col] += signs * multiplier * self.std

        self.record_corruption('spikes', target_indices, {'fraction': fraction, 'multiplier': multiplier})
        return self.df
