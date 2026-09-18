from tsadquality.reproducibility import ReproducibleOperations

from .BaseError import BaseCorruptor


class SpikeInjector(BaseCorruptor):
    """
    Injects sudden extreme spikes at isolated points.
    """

    def inject(self, fraction, multiplier):
        count = int(len(self.df) * fraction)
        target_indices = self.get_injectable_indices(count)

        if len(target_indices) == 0:
            return self.df

        signs = ReproducibleOperations.choice([-1, 1], size=len(target_indices))
        self.df.loc[target_indices, self.value_col] += signs * multiplier * self.std

        params = {'fraction': fraction, 'multiplier': multiplier}
        self.record_corruption('spikes', target_indices, params)
        return self.df
